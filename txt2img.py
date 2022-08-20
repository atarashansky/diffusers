import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
#from torch import autocast
from contextlib import contextmanager, nullcontext

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


class T2M:
  def __init__(self,
    ckpt="model.ckpt",
    config="stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
    plms=True,
  ):
    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)        
    

    self.model=model
    self.sampler=sampler
    self.device=device
  
  def run(self,
          prompt="a painting of a virus monster playing guitar",
          ddim_steps=50,
          laion400m=False,
          fixed_code=False,
          ddim_eta=0.0,
          n_iter=2,
          H=512,
          W=512,
          C=4,
          f=8,
          n_samples=3,
          n_rows=0,
          scale=7.5,
          factor=0.0,
          precision="autocast",
          seed=42):
        
    seed_everything(seed)
    sampler = self.sampler
    model = self.model
    device = self.device  

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size

    prompt = prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    start_code = None
    if fixed_code:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)

    precision_scope = nullcontext#autocast if precision=="autocast" else nullcontext
    images=[]    
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
                                c = model.get_learned_conditioning([prompt])[0]
                            TEs.append(c)
                        c = torch.stack(TEs)
                                
                        shape = [C, H // f, W // f]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)
                        

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        all_samples.append(x_samples_ddim)

                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            all_images.append(Image.fromarray(x_sample.astype(np.uint8)))
                        


                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                return Image.fromarray(grid.astype(np.uint8)), all_images