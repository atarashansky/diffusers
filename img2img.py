"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


class I2I:
    def __init__(self,
                 model=None,
                 config="stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
                 ckpt="model.ckpt",
                 plms=False
                ):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")        
        if model is None:
            config = OmegaConf.load(f"{config}")
            model = load_model_from_config(config, f"{ckpt}")
            model = model.to(device)

        if plms:
            raise NotImplementedError("PLMS sampler not (yet) supported")
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)  
            
        self.sampler=sampler
        self.model=model
        self.device=device

    def run(self,
            init_img,
            prompt="a painting of a virus monster playing guitar",
            ddim_steps=50,
            fixed_code=False,
            ddim_eta=0.0,
            n_iter=2,
            C=4,
            f=8,
            n_samples=3,
            n_rows=0,
            scale=7.5,
            strength=0.75,
            from_file=None,
            seed=42,
            factor=0.5,
            precision="autocast"
           ):
        seed_everything(seed)
        
        device = self.device
        model = self.model
        sampler = self.sampler
        
        batch_size = n_samples
        n_rows = n_rows if n_rows > 0 else batch_size
        if not from_file:
            prompt = prompt
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {from_file}")
            with open(from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))



        assert os.path.isfile(init_img)
        init_image = load_img(init_img).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * ddim_steps)
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast if precision == "autocast" else nullcontext
        all_images=[]
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(n_iter, desc="Sampling"):
                        seed_everything(seed+n+1)
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                                
                            TEs=[]
                            for prompt in prompts:
                                if isinstance(prompt,list):
                                    c = model.get_learned_conditioning(prompt)        
                                    n = len(prompt)
                                    ff = min(factor,n-1)
                                    base = int(np.floor(max(0,ff-1e-6)))
                                    scaler = np.append(np.zeros(base),[1-(ff-base), ff-base])
                                    scaler = np.append(scaler,[0]*(n-len(scaler)))
                                    scaler=scaler.astype('float32')[:,None,None]
                                    scaler = torch.from_numpy(scaler).to("cuda")

                                    c = (c*scaler).sum(0) 
                                else:
                                    c = model.get_learned_conditioning([prompt])
                                TEs.append(c)
                            c = torch.stack(TEs)                                

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc,)

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)


                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                all_images.append(Image.fromarray(x_sample.astype(np.uint8)))
                            all_samples.append(x_samples)


                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    grid = Image.fromarray(grid.astype(np.uint8))

                    toc = time.time()
                    return grid, all_images
