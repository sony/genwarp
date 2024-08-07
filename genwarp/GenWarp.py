from os.path import join
from typing import Union, Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import inspect

from omegaconf import OmegaConf, DictConfig
from jaxtyping import Float

import torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from .models import (
    PoseGuider,
    UNet2DConditionModel,
    UNet3DConditionModel,
    ReferenceAttentionControl
)
from .ops import get_viewport_matrix, forward_warper

class GenWarp():
    @dataclass
    class Config():
        pretrained_model_path: str = ''
        checkpoint_name: str = ''
        half_precision_weights: bool = False
        height: int = 512
        width: int = 512
        num_inference_steps: int = 20
        guidance_scale: float = 3.5

    cfg: Config

    class Embedder():
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.create_embedding_fn()

        def create_embedding_fn(self) -> None:
            embed_fns = []
            d = self.kwargs['input_dims']
            out_dim = 0
            if self.kwargs['include_input']:
                embed_fns.append(lambda x : x)
                out_dim += d

            max_freq = self.kwargs['max_freq_log2']
            N_freqs = self.kwargs['num_freqs']

            if self.kwargs['log_sampling']:
                freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            else:
                freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

            for freq in freq_bands:
                for p_fn in self.kwargs['periodic_fns']:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                    out_dim += d

            self.embed_fns = embed_fns
            self.out_dim = out_dim

        def embed(self, inputs) -> Tensor:
            return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def __init__(
        self,
        cfg: Optional[Union[dict, DictConfig]] = None,
        device: Optional[str] = 'cuda:0'
    ) -> None:
        self.cfg = OmegaConf.structured(self.Config(**cfg))
        self.model_path = join(
            self.cfg.pretrained_model_path, self.cfg.checkpoint_name
        )
        self.device = device
        self.configure()

    def configure(self) -> None:
        print(f"Loading GenWarp...")

        # Configurations.
        self.dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        self.viewport_mtx: Float[Tensor, 'B 4 4'] = get_viewport_matrix(
            self.cfg.width, self.cfg.height,
            batch_size=1, device=self.device
        ).to(self.dtype)

        # Load models.
        self.load_models()

        # Timestep
        self.scheduler.set_timesteps(
            self.cfg.num_inference_steps, device=self.device)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        print(f"Loaded GenWarp.")

    def load_models(self) -> None:
        # VAE.
        self.vae = AutoencoderKL.from_pretrained(
            join(self.cfg.pretrained_model_path, 'sd-vae-ft-mse')
        ).to(self.device, dtype=self.dtype)

        # Image processor.
        self.vae_scale_factor = \
            2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.clip_image_processor = CLIPImageProcessor()

        # Image encoder.
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            join(self.cfg.pretrained_model_path, 'image_encoder')
        ).to(self.device, dtype=self.dtype)

        # Reference Unet.
        self.reference_unet = UNet2DConditionModel.from_config(
            UNet2DConditionModel.load_config(
                join(self.model_path, 'config.json')
        )).to(self.device, dtype=self.dtype)
        self.reference_unet.load_state_dict(torch.load(
            join(self.model_path, 'reference_unet.pth'),
            map_location= 'cpu'),
        )

        # Denoising Unet.
        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            join(self.model_path, 'config.json'),
            join(self.model_path, 'denoising_unet.pth')
        ).to(self.device, dtype=self.dtype)
        self.unet_in_channels = self.denoising_unet.config.in_channels

        # Pose guider.
        self.pose_guider = PoseGuider(
            conditioning_embedding_channels=320,
            conditioning_channels=11,
        ).to(self.device, dtype=self.dtype)
        self.pose_guider.load_state_dict(torch.load(
            join(self.model_path, 'pose_guider.pth'),
            map_location='cpu'),
        )

        # Noise scheduler
        sched_kwargs = OmegaConf.to_container(OmegaConf.create({
            'num_train_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'scaled_linear',
            'steps_offset': 1,
            'clip_sample': False
        }))
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing='trailing',
            prediction_type='v_prediction',
        )
        self.scheduler = DDIMScheduler(**sched_kwargs)

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.reference_unet.requires_grad_(False)
        self.denoising_unet.requires_grad_(False)
        self.pose_guider.requires_grad_(False)

        # Coordinates embedding.
        self.embedder = self.get_embedder(2)

    def get_embedder(self, multires):
        embed_kwargs = {
            'include_input' : True,
            'input_dims' : 2,
            'max_freq_log2' : multires-1,
            'num_freqs' : multires,
            'log_sampling' : True,
            'periodic_fns' : [torch.sin, torch.cos],
        }

        embedder_obj = self.Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        return embed

    def __call__(
        self,
        src_image: Float[Tensor, 'B C H W'],
        src_depth: Float[Tensor, 'B C H W'],
        rel_view_mtx: Float[Tensor, 'B 4 4'],
        src_proj_mtx: Float[Tensor, 'B 4 4'],
        tar_proj_mtx: Float[Tensor, 'B 4 4'],
    ) -> Dict[str, Tensor]:
        """ Perform NVS.
        """
        batch_size = src_image.shape[0]

        # Rearrange and resize.
        src_image = self.preprocess_image(src_image)
        src_depth = self.preprocess_image(src_depth)
        viewport_mtx = repeat(
            self.viewport_mtx, 'b h w -> (repeat b) h w',
            repeat=batch_size)

        pipe_args = dict(
            src_image=src_image,
            src_depth=src_depth,
            rel_view_mtx=rel_view_mtx,
            src_proj_mtx=src_proj_mtx,
            tar_proj_mtx=tar_proj_mtx,
            viewport_mtx=viewport_mtx
        )

        # Prepare inputs.
        conditions, renders = self.prepare_conditions(**pipe_args)

        # NVS.
        latents_clean = self.perform_nvs(
            **pipe_args,
            **conditions
        )

        # Decode to images.
        synthesized = self.decode_latents(latents_clean)

        inference_out = {
            'synthesized': synthesized,
            'warped': renders['warped'],
            'mask': renders['mask'],
            'correspondence': conditions['correspondence']
        }

        return inference_out

    def preprocess_image(
        self,
        image: Float[Tensor, 'B C H W']
    ) -> Float[Tensor, 'B C H W']:
        image = F.interpolate(
            image, (self.cfg.height, self.cfg.width)
        )
        return image

    def get_image_prompt(
        self,
        src_image: Float[Tensor, 'B C H W']
    ) -> Float[Tensor, '2 B L']:
        ref_image_for_clip = self.vae_image_processor.preprocess(
            src_image, height=224, width=224
        )
        ref_image_for_clip = ref_image_for_clip * 0.5 + 0.5

        clip_image = self.clip_image_processor.preprocess(
            ref_image_for_clip, return_tensors='pt'
        ).pixel_values

        clip_image_embeds = self.image_encoder(
            clip_image.to(self.device, dtype=self.image_encoder.dtype)
        ).image_embeds

        image_prompt_embeds = clip_image_embeds.unsqueeze(1)
        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)

        image_prompt_embeds = torch.cat(
            [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
        )

        return image_prompt_embeds

    def encode_images(
        self,
        rgb: Float[Tensor, 'B C H W']
    ) -> Float[Tensor, 'B C H W']:
        rgb = self.vae_image_processor.preprocess(rgb)
        latents = self.vae.encode(rgb).latent_dist.mean
        latents = latents * 0.18215
        return latents

    def decode_latents(
        self,
        latents: Float[Tensor, 'B C H W']
    ) -> Float[Tensor, 'B C H W']:
        latents = 1 / 0.18215 * latents
        rgb = []
        for frame_idx in range(latents.shape[0]):
            rgb.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        rgb = torch.cat(rgb)
        rgb = (rgb / 2 + 0.5).clamp(0, 1)
        return rgb.squeeze(2)

    def get_reference_controls(
        self,
        batch_size: int
    ) -> Tuple[ReferenceAttentionControl, ReferenceAttentionControl]:
        reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=True,
            mode='read',
            batch_size=batch_size,
            fusion_blocks='full',
            feature_fusion_type='attention_full_sharing'
        )
        writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=True,
            mode='write',
            batch_size=batch_size,
            fusion_blocks='full',
            feature_fusion_type='attention_full_sharing'
        )

        return reader, writer

    def prepare_extra_step_kwargs(
        self,
        generator,
        eta
    ) -> Dict[str, Any]:
        accepts_eta = 'eta' in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        # check if the scheduler accepts generator
        accepts_generator = 'generator' in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def get_pose_features(
        self,
        src_embed: Float[Tensor, 'B C H W'],
        trg_embed: Float[Tensor, 'B C H W'],
        do_classifier_guidance: bool = True
    ) -> Tuple[Tensor, Tensor]:
        pose_cond_tensor = src_embed.unsqueeze(2)
        pose_cond_tensor = pose_cond_tensor.to(
            device=self.device, dtype=self.pose_guider.dtype
        )
        pose_cond_tensor_2 = trg_embed.unsqueeze(2)
        pose_cond_tensor_2 = pose_cond_tensor_2.to(
            device=self.device, dtype=self.pose_guider.dtype
        )
        pose_fea = self.pose_guider(pose_cond_tensor)
        pose_fea_2 = self.pose_guider(pose_cond_tensor_2)

        if do_classifier_guidance:
            pose_fea = torch.cat([pose_fea] * 2)
            pose_fea_2 = torch.cat([pose_fea_2] * 2)

        return pose_fea, pose_fea_2

    @torch.no_grad()
    def prepare_conditions(
        self,
        src_image: Float[Tensor, 'B C H W'],
        src_depth: Float[Tensor, 'B C H W'],
        rel_view_mtx: Float[Tensor, 'B 4 4'],
        src_proj_mtx: Float[Tensor, 'B 4 4'],
        tar_proj_mtx: Float[Tensor, 'B 4 4'],
        viewport_mtx: Float[Tensor, 'B 4 4']
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        # Prepare inputs.
        B = src_image.shape[0]
        H, W = src_image.shape[2:4]
        src_scr_mtx = (viewport_mtx @ src_proj_mtx).to(src_proj_mtx)
        mvp_mtx = (tar_proj_mtx @ rel_view_mtx).to(rel_view_mtx)

        # Coordinate grids.
        grid: Float[Tensor, 'H W C'] = torch.stack(torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy'), dim=-1
        ).to(self.device, dtype=self.dtype)

        # Unproject depth.
        screen = F.pad(grid, (0, 1), 'constant', 0)  # z=0 (z doesn't matter)
        screen = F.pad(screen, (0, 1), 'constant', 1)  # w=1
        screen = repeat(screen, 'h w c -> b h w c', b=B)
        screen_flat = rearrange(screen, 'b h w c -> b (h w) c')
        # To eye coordinates.
        eye = screen_flat @ torch.linalg.inv_ex(
            src_scr_mtx.float()
        )[0].mT.to(self.dtype)
        # Overwrite depth.
        eye = eye * rearrange(src_depth, 'b c h w -> b (h w) c')
        eye[..., 3] = 1

        # Coordinates embedding.
        coords = torch.stack((grid[..., 0]/H, grid[..., 1]/W), dim=-1)
        embed = repeat(self.embedder(coords), 'h w c -> b c h w', b=B)

        # Warping.
        input_image: Float[Tensor, 'B C H W'] = torch.cat(
            [embed, src_image], dim=1
        )
        output_image = forward_warper(
            input_image, screen_flat[..., :2], eye,
            mvp_mtx=mvp_mtx, viewport_mtx=viewport_mtx
        )
        warped_embed = output_image['warped'][:, :embed.shape[1]]
        warped_image = output_image['warped'][:, embed.shape[1]:]
        # mask == 1 where there's no pixel
        mask = output_image['mask']
        correspondence = output_image['correspondence']

        # Conditions.
        src_coords_embed = torch.cat(
            [embed, torch.zeros_like(mask, device=mask.device)], dim=1)
        trg_coords_embed = torch.cat([warped_embed, mask], dim=1)

        # Outputs.
        conditions = dict(
            src_coords_embed=src_coords_embed,
            trg_coords_embed=trg_coords_embed,
            correspondence=correspondence
        )

        renders = dict(
            warped=warped_image,
            mask=1 - mask  # mask == 1 where there's a pixel
        )

        return conditions, renders

    def perform_nvs(
        self,
        src_image,
        src_coords_embed,
        trg_coords_embed,
        correspondence,
        eta: float=0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
        **kwargs,
    ) -> Float[Tensor, 'B C H W']:
        batch_size = src_image.shape[0]

        # For the cross attention.
        reference_control_reader, reference_control_writer = \
            self.get_reference_controls(batch_size)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(
            generator, eta
        )

        with torch.no_grad():
            # Create fake inputs. It'll be replaced by pure noise.
            latents = torch.randn(
                batch_size,
                self.unet_in_channels,
                self.cfg.height // self.vae_scale_factor,
                self.cfg.width // self.vae_scale_factor
            ).to(self.device, dtype=src_image.dtype)
            initial_t = torch.tensor(
                [self.num_train_timesteps - 1] * batch_size
            ).to(self.device, dtype=torch.long)

            # Add noise.
            noise = torch.randn_like(latents)
            latents_noisy_start = self.scheduler.add_noise(
                latents, noise, initial_t
            )
            latents_noisy_start = latents_noisy_start.unsqueeze(2)

            # Prepare clip image embeds.
            image_prompt_embeds = self.get_image_prompt(src_image)

            # Prepare ref image latents.
            ref_image_latents = self.encode_images(src_image)

            # Prepare pose condition image.
            pose_fea, pose_fea_2 = self.get_pose_features(
                src_coords_embed, trg_coords_embed
            )

            pose_fea = pose_fea[:, :, 0, ...]

            # Forward reference images.
            self.reference_unet(
                ref_image_latents.repeat(2, 1, 1, 1),
                torch.zeros(batch_size * 2).to(ref_image_latents),
                encoder_hidden_states=image_prompt_embeds,
                pose_cond_fea=pose_fea,
                return_dict=False,
            )
            # Update the denosing net with reference features.
            reference_control_reader.update(
                reference_control_writer,
                correspondence=correspondence
            )

            timesteps = self.scheduler.timesteps
            latents_noisy = latents_noisy_start
            for t in timesteps:
                # Prepare latents.
                latent_model_input = torch.cat([latents_noisy] * 2)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # Denoise.
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_prompt_embeds,
                    pose_cond_fea=pose_fea_2,
                    return_dict=False,
                )[0]

                # CFG.
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # t -> t-1
                latents_noisy = self.scheduler.step(
                    noise_pred, t, latents_noisy, **extra_step_kwargs,
                    return_dict=False
                )[0]

            # Noise disappears eventually
            latents_clean = latents_noisy

        reference_control_reader.clear()
        reference_control_writer.clear()

        return latents_clean.squeeze(2)