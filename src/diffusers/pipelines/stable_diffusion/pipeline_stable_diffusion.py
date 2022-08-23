import inspect
import warnings
from typing import List, Optional, Union

import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from einops import rearrange
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from ...models import AutoencoderKL, UNet2DConditionModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler


class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        seed = 42,
        n_iter=1,
        init_image=None,
        strength = 0.8
    ):      
        output_type = "np"
        if init_image is not None and not isinstance(self.scheduler, PNDMScheduler):
            raise ValueError("image-to-image only works with PNDMScheduler for now")
            
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset=0
        if accepts_offset:
            extra_set_kwargs["offset"] = 1  
            offset=1
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        init_latents = None
        init_timestep = None
        if init_image is not None:
            init_image = init_image.resize((width,height))
            init_latents = self.encode_image(init_image)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)
            timesteps = self.scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
            # prepare init_latents noise to latents
        
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the intial random noise
        seeds = np.zeros((n_iter,batch_size))
        all_images = []
        all_PIL = []
        counter = 0
        for ITER in range(n_iter):
            print("Dreaming iteration",ITER)
            latents = []
            for i in range(batch_size):
                seeds[ITER,i] = seed + counter
                counter+=1
                torch.manual_seed(seeds[ITER,i])  
                noise = torch.randn((self.unet.in_channels, height // 8, width // 8), device=self.device)
                if init_image is not None:
                    latents.append(self.scheduler.add_noise(init_latents, noise, timesteps)[0])
                else:    
                    latents.append(noise)
            
            latents = torch.stack(latents)

            self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

            # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = latents * self.scheduler.sigmas[0]

            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
            # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
            # and should be between [0, 1]
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
            
            
            t_start = 0 if init_image is None else max(num_inference_steps - init_timestep + offset, 0)
            for i, t in (enumerate(self.scheduler.timesteps[t_start:])):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    sigma = self.scheduler.sigmas[i]
                    latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

            image = self.decode_image(latents,return_pil=False)
            all_images.append(torch.from_numpy(image))
            all_PIL.append([Image.fromarray((image[i]*255).astype('uint8')) for i in range(batch_size)])
        
        grid = torch.stack(all_images, 0)
        grid = rearrange(grid, 'n b h w c -> (n b) h w c')
        grid = rearrange(grid, 'n h w c -> n c h w')
        grid = make_grid(grid, nrow=n_iter)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        grid = Image.fromarray(grid.astype(np.uint8))        
        return grid, all_PIL, seeds
    
    @torch.no_grad()
    def encode_image(self,image):
        numpy_image = np.array(image).astype(np.float32) / 255.0
        numpy_image = numpy_image[None].transpose(0, 3, 1, 2)
        numpy_image = torch.from_numpy(numpy_image).to(self.device)
        encoded = self.vae.encode(2*numpy_image-1)
        latent_mode = encoded.sample() * 0.18215
        return latent_mode
    
    @torch.no_grad()
    def decode_image(self,latent,return_pil=True):
        output_image = self.vae.decode(1 / 0.18215 * latent)
        output_image = (output_image / 2 + 0.5).clamp(0, 1)
        output_image = output_image.cpu().permute(0, 2, 3, 1).numpy()
        if return_pil:
            output_image = Image.fromarray((output_image[0]*255).astype('uint8'))
        return output_image        
    
    
    def make_grid(self,
                  prompt,
                  seed=42,
                  num_rows=4,
                  num_columns=4,
                  height=512,
                  width=512,
                  guidance_scale=7.5,
                  num_inference_steps=50,
                  init_image=None,
                  strength=0.8):
        return self(num_columns * [prompt],
                             guidance_scale=guidance_scale,
                             num_inference_steps=num_inference_steps,
                             n_iter = num_rows,
                             height=height,
                             width=width,
                             seed=seed,
                             init_image=init_image,
                             strength=strength)