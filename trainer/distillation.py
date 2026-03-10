import gc
import logging
import numpy as np

from utils.dataset import cycle, I2VDataset, TextDataset
from einops import rearrange
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # Use gradient accumulation to match the total batch size
        # Override if gradient_accumulation_steps is provided
        self.grad_accumulation_steps = max(1, config.total_batch_size // (config.batch_size * self.world_size))
        if hasattr(config, "gradient_accumulation_steps"):
            self.grad_accumulation_steps = config.gradient_accumulation_steps
        print(f"Using gradient accumulation steps: {self.grad_accumulation_steps}")

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if getattr(config, "i2v", False) and self.model.clip_encoder is not None:
            self.model.clip_encoder = fsdp_wrap(
                self.model.clip_encoder,
                sharding_strategy="no_shard",
                mixed_precision=config.mixed_precision,
                wrap_strategy=getattr(config, "clip_encoder_fsdp_wrap_strategy", "size")
            )

        # Load VAE for I2V (needed for encoding images) or visualization
        if not config.no_visualize or config.load_raw_video or getattr(config, "i2v", False):
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = I2VDataset(config.data_path)
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        num_workers = getattr(config, "num_workers", 8)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            multiprocessing_context='spawn' if num_workers > 0 else None)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 4: Set up visualization
        self.vis_pipeline = None
        if not config.no_visualize:
            vis_sample = dataset[0]
            self.vis_prompt = vis_sample["prompts"]
            self.vis_image = vis_sample.get("image", None)
            with torch.no_grad():
                vis_cond = self.model.text_encoder(text_prompts=[self.vis_prompt])
                self.vis_conditional_dict = {k: v.detach().clone() for k, v in vis_cond.items()}
            if self.is_main_process:
                print(f"Visualization prompt: {self.vis_prompt[:80]}...")

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                gen_sd = state_dict["generator"]
            elif "model" in state_dict:
                gen_sd = state_dict["model"]
            else:
                gen_sd = state_dict
            self.model.generator.load_state_dict(
                gen_sd, strict=True
            )

            if "critic" in state_dict:
                self.model.fake_score.load_state_dict(
                    state_dict["critic"], strict=True
                )
                print("Loaded critic state dict")


        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

        return generator_state_dict

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        clean_latent = None
        image_latent = None
        conditional_dict_i2v = None
        unconditional_dict_i2v = None

        if self.config.i2v:
            # Get raw image from batch: [B, C, H, W] normalized to [-1, 1]
            raw_image = batch["image"].to(device=self.device, dtype=self.dtype)

            with torch.no_grad():
                # VAE encode raw image to get image_latent
                # image needs to be [B, C, T, H, W] for VAE
                image_for_vae = rearrange(raw_image, "b c h w -> b c 1 h w")
                image_latent = self.model.vae.encode_to_latent(image_for_vae)  # [B, 1, C, H, W]
                lb, lf, lc, lh, lw = list(self.config.image_or_video_shape)

                # CLIP encode image to get clip_fea
                clip_fea = self.model.clip_encoder(raw_image)

                # Create mask tensor for I2V conditioning
                # Following wan/image2video.py lines 207-214
                msk = torch.ones(lb, 4, lf, lh, lw, device=self.device, dtype=self.dtype)
                msk[:, :, 1:, :, :] = 0

                image_latent_padded = torch.zeros(lb, lc, lf, lh, lw, device=self.device, dtype=self.dtype)
                image_latent_padded[:, :, 0, :, :] = image_latent

                # y = concat along channel dim: [B, 4+16, 21, H, W] = [B, 20, 21, H, W]
                y = torch.cat([msk, image_latent_padded], dim=1)

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

            # Build I2V conditional dicts if in I2V mode
            # For I2V, real_score needs clip_fea and y in addition to prompt_embeds
            # For CFG, both cond and uncond use same clip_fea and y, only text differs
            if self.config.i2v:
                conditional_dict_i2v = {
                    "prompt_embeds": conditional_dict["prompt_embeds"],
                    "clip_fea": clip_fea,
                    "y": y
                }
                unconditional_dict_i2v = {
                    "prompt_embeds": unconditional_dict["prompt_embeds"],
                    "clip_fea": clip_fea,  # Same clip_fea for unconditional
                    "y": y  # Same y for unconditional
                }

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            return_predictions = (self.step % self.config.log_iters == 0) and self.is_main_process
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                conditional_dict_i2v=conditional_dict_i2v,
                unconditional_dict_i2v=unconditional_dict_i2v,
                return_predictions=return_predictions,
            )

            scaled_generator_loss = generator_loss / self.grad_accumulation_steps
            scaled_generator_loss.backward()

            generator_log_dict.update({"generator_loss": generator_loss})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        # Note: critic_loss always uses T2V model which doesn't need I2V conditioning
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        scaled_critic_loss = critic_loss / self.grad_accumulation_steps
        scaled_critic_loss.backward()

        critic_log_dict.update({"critic_loss": critic_loss})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def _init_vis_pipeline(self):
        """Create a lightweight inference pipeline for visualization (rank 0 only)."""
        from pipeline.causal_inference import CausalInferencePipeline
        from utils.wan_wrapper import WanDiffusionWrapper

        class _CachedTextEncoder(torch.nn.Module):
            def __init__(self, cached_dict):
                super().__init__()
                self._cached = cached_dict
            def forward(self, text_prompts=None):
                return self._cached

        vis_generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)
        vis_generator = vis_generator.cpu()

        self.vis_pipeline = CausalInferencePipeline(
            self.config,
            device=self.device,
            generator=vis_generator,
            text_encoder=_CachedTextEncoder(self.vis_conditional_dict),
            vae=self.model.vae
        )

    @torch.no_grad()
    def _decode_diagnostics(self, log_dict):
        """Decode teacher/critic/generator x0 predictions and save as images for diagnosis."""
        if not self.is_main_process:
            return
        keys = [("_pred_real", "teacher"), ("_pred_fake", "critic"), ("_generator_output", "generator")]
        diag_dir = os.path.join(self.output_path, "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        try:
            for key, label in keys:
                if key not in log_dict:
                    continue
                latent = log_dict[key][:1]  # first sample only: [1, F, C, H, W]
                frame_indices = [0, latent.shape[1] // 2, latent.shape[1] - 1]
                pixels = self.model.vae.decode_to_pixel(
                    latent.to(device=self.device, dtype=self.dtype))  # [1, F, C_rgb, H_px, W_px]
                for fi in frame_indices:
                    frame = pixels[0, fi]  # [C, H, W]
                    frame = frame.clamp(-1, 1).mul(0.5).add(0.5).mul(255).byte()
                    frame = frame.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                    from PIL import Image
                    img = Image.fromarray(frame)
                    img.save(os.path.join(diag_dir, f"step{self.step:06d}_{label}_f{fi}.png"))
                    if not self.disable_wandb:
                        wandb.log({
                            f"diag/{label}_f{fi}": wandb.Image(img)
                        }, step=self.step)
            print(f"[Diag] Saved diagnostic frames to {diag_dir}")
        except Exception as e:
            print(f"[Warning] Diagnostic decode failed at step {self.step}: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def _visualize(self, generator_state_dict):
        """Generate a video with the current generator weights and log it."""
        if not self.is_main_process:
            return
        try:
            if self.vis_pipeline is None:
                self._init_vis_pipeline()

            self.vis_pipeline.generator.load_state_dict(generator_state_dict)
            self.vis_pipeline.generator = self.vis_pipeline.generator.to(
                device=self.device, dtype=self.dtype)

            video = self.generate_video(
                self.vis_pipeline,
                [self.vis_prompt],
                self.vis_image
            )

            self.vis_pipeline.generator = self.vis_pipeline.generator.cpu()
            torch.cuda.empty_cache()

            video_uint8 = video[0].clip(0, 255).astype(np.uint8)

            vis_dir = os.path.join(self.output_path, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            video_path = os.path.join(vis_dir, f"step_{self.step:06d}.mp4")
            from torchvision.io import write_video
            write_video(video_path, torch.from_numpy(video_uint8), fps=16)
            print(f"[Vis] Saved video to {video_path}")

            if not self.disable_wandb:
                wandb.log({
                    "generated_video": wandb.Video(
                        video_uint8, caption=self.vis_prompt[:100], fps=16, format="mp4")
                }, step=self.step)

        except Exception as e:
            print(f"[Warning] Visualization failed at step {self.step}: {e}")
            import traceback
            traceback.print_exc()

    def train(self):
        start_step = self.step

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                for i in range(self.grad_accumulation_steps):
                    batch = next(self.dataloader)
                    if i < self.grad_accumulation_steps - 1:
                        with self.model.generator.no_sync():
                            extra = self.fwdbwd_one_step(batch, True)
                    else:
                        extra = self.fwdbwd_one_step(batch, True)
                    extras_list.append(extra)
                generator_grad_norm = self.model.generator.clip_grad_norm_(
                    self.max_grad_norm_generator)
                generator_log_dict = merge_dict_list(extras_list)
                generator_log_dict.update({"generator_grad_norm": generator_grad_norm})
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

                if self.is_main_process and self.step % self.config.log_iters == 0:
                    self._decode_diagnostics(generator_log_dict)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            for i in range(self.grad_accumulation_steps):
                batch = next(self.dataloader)
                if i < self.grad_accumulation_steps - 1:
                    with self.model.fake_score.no_sync():
                        extra = self.fwdbwd_one_step(batch, False)
                else:
                    extra = self.fwdbwd_one_step(batch, False)
                extras_list.append(extra)
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic)
            critic_log_dict = merge_dict_list(extras_list)
            critic_log_dict.update({"critic_grad_norm": critic_grad_norm})
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model and visualize
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                generator_state_dict = self.save()
                torch.cuda.empty_cache()

                if not self.config.no_visualize:
                    self._visualize(generator_state_dict)
                    del generator_state_dict
                    torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item(),
                        }
                    )
                    for diag_key in ("raw_grad_norm", "grad_normalizer", "pred_real_mean", "pred_fake_mean"):
                        if diag_key in generator_log_dict:
                            wandb_loss_dict[diag_key] = generator_log_dict[diag_key].mean().item()

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                current_time = time.time()
                if self.previous_time is not None:
                    iter_time = current_time - self.previous_time
                    wandb_loss_dict["iter_time"] = iter_time
                loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in wandb_loss_dict.items())
                print(f"[Step {self.step}] {loss_str}", flush=True)

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                self.previous_time = current_time
