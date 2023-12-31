import argparse
import torch
from omegaconf import OmegaConf
from jieba import re
import os
from einops import repeat, rearrange
from basicsr.utils import tensor2img, img2tensor

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.adapter import Adapter, StyleAdapter, Adapter_light
from ldm.modules.extra_condition.api import ExtraCondition
from ldm.util import fix_cond_shapes, load_model_from_config, read_state_dict, load_img, resize_numpy_image, get_resize_shape

DEFAULT_NEGATIVE_PROMPT = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                          'fewer digits, cropped, worst quality, low quality'


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_base_argument_parser() -> argparse.ArgumentParser:
    """get the base argument parser for inference scripts"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outdir',
        type=str,
        help='dir to write results to',
        default=None,
    )

    parser.add_argument(
        '--prompt',
        type=str,
        nargs='?',
        default=None,
        help='positive prompt',
    )

    parser.add_argument(
        '--neg_prompt',
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help='negative prompt',
    )

    parser.add_argument(
        '--cond_path',
        type=str,
        default=None,
        help='condition image path',
    )
    
    parser.add_argument(
        '--init_img',
        type=str,
        default=None,
        help='your init image path'
    )

    parser.add_argument(
        '--cond_inp_type',
        type=str,
        default='image',
        help='the type of the input condition image, take depth T2I as example, the input can be raw image, '
        'which depth will be calculated, or the input can be a directly a depth map image',
    )

    parser.add_argument(
        '--sampler',
        type=str,
        default='ddim',
        choices=['ddim', 'plms'],
        help='sampling algorithm, currently, only ddim and plms are supported, more are on the way',
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='number of sampling steps',
    )

    parser.add_argument(
        '--sd_ckpt',
        type=str,
        default='models/sd-v1-4.ckpt',
        help='path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported',
    )

    parser.add_argument(
        '--vae_ckpt',
        type=str,
        default=None,
        help='vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded',
    )

    parser.add_argument(
        '--adapter_ckpt',
        type=str,
        default=None,
        help='path to checkpoint of adapter',
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/stable-diffusion/sd-v1-inference.yaml',
        help='path to config which constructs SD model',
    )

    parser.add_argument(
        '--max_resolution',
        type=float,
        default=512 * 512,
        help='max image height * width, only for computer with limited vram',
    )

    parser.add_argument(
        '--resize_short_edge',
        type=int,
        default=None,
        help='resize short edge of the input image, if this arg is set, max_resolution will not be used',
    )

    parser.add_argument(
        '--C',
        type=int,
        default=4,
        help='latent channels',
    )

    parser.add_argument(
        '--f',
        type=int,
        default=8,
        help='downsampling factor',
    )

    parser.add_argument(
        '--scale',
        type=float,
        default=7.5,
        help='unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))',
    )

    parser.add_argument(
        '--cond_tau',
        type=float,
        default=1.0,
        help='timestamp parameter that determines until which step the adapter is applied, '
        'similar as Prompt-to-Prompt tau',
    )

    parser.add_argument(
        '--style_cond_tau',
        type=float,
        default=1.0,
        help='timestamp parameter that determines until which step the adapter is applied, '
             'similar as Prompt-to-Prompt tau',
    )

    parser.add_argument(
        '--cond_weight',
        type=float,
        default=1.0,
        help='the adapter features are multiplied by the cond_weight. The larger the cond_weight, the more aligned '
        'the generated image and condition will be, but the generated quality may be reduced',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )

    parser.add_argument(
        '--n_samples',
        type=int,
        default=4,
        help='# of samples to generate',
    )

    return parser


def get_sd_models(opt):
    """
    build stable diffusion model, sampler
    """
    # SD
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, opt.sd_ckpt, opt.vae_ckpt)
    sd_model = model.to(opt.device)

    # sampler
    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(model)
    else:
        raise NotImplementedError

    return sd_model, sampler


def get_t2i_adapter_models(opt):
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, opt.sd_ckpt, opt.vae_ckpt)
    adapter_ckpt_path = getattr(opt, f'{opt.which_cond}_adapter_ckpt', None)
    if adapter_ckpt_path is None:
        adapter_ckpt_path = getattr(opt, 'adapter_ckpt')
    adapter_ckpt = read_state_dict(adapter_ckpt_path)
    new_state_dict = {}
    for k, v in adapter_ckpt.items():
        if not k.startswith('adapter.'):
            new_state_dict[f'adapter.{k}'] = v
        else:
            new_state_dict[k] = v
    m, u = model.load_state_dict(new_state_dict, strict=False)
    if len(u) > 0:
        print(f"unexpected keys in loading adapter ckpt {adapter_ckpt_path}:")
        print(u)

    model = model.to(opt.device)

    # sampler
    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(model)
    else:
        raise NotImplementedError

    return model, sampler


def get_cond_ch(cond_type: ExtraCondition):
    if cond_type == ExtraCondition.sketch or cond_type == ExtraCondition.canny:
        return 1
    return 3


def get_adapters(opt, cond_type: ExtraCondition):
    adapter = {}
    cond_weight = getattr(opt, f'{cond_type.name}_weight', None)
    if cond_weight is None:
        cond_weight = getattr(opt, 'cond_weight')
    adapter['cond_weight'] = cond_weight

    if cond_type == ExtraCondition.style:
        adapter['model'] = StyleAdapter(width=1024, context_dim=768, num_head=8, n_layes=3, num_token=8).to(opt.device)
    elif cond_type == ExtraCondition.color:
        adapter['model'] = Adapter_light(
            cin=64 * get_cond_ch(cond_type),
            channels=[320, 640, 1280, 1280],
            nums_rb=4).to(opt.device)
    else:
        adapter['model'] = Adapter(
            cin=64 * get_cond_ch(cond_type),
            channels=[320, 640, 1280, 1280][:4],
            nums_rb=2,
            ksize=1,
            sk=True,
            use_conv=False).to(opt.device)
    # point-e model: the same with depth model
    ckpt_path = getattr(opt, f'{cond_type.name}_adapter_ckpt', None)
    if ckpt_path is None:
        ckpt_path = getattr(opt, 'adapter_ckpt')
    state_dict = read_state_dict(ckpt_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('adapter.'):
            new_state_dict[k[len('adapter.'):]] = v
        else:
            new_state_dict[k] = v

    adapter['model'].load_state_dict(new_state_dict)

    return adapter


def diffusion_inference(opt, model, sampler, adapter_features, append_to_context=None):
    # get text embedding
    
    ori_c = model.get_learned_conditioning([opt.prompt])
   
    c_ = re.split('[,.!?]', opt.prompt)
    c_list = [cc for cc in c_ if cc != '' and cc!= None]
    c = [model.get_learned_conditioning([cc]) for cc in c_list]
    # gather prompts that are cut via list
        
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning([opt.neg_prompt])
    else:
        uc = None
        
    ori_c, ori_uc, c, uc = fix_cond_shapes(model, ori_c, c, uc)
    
    # print(f'list length after fixing: {len(c)}, {len(uc)}')
    
    # [0] if isinstance(c, list) else uc
    
    if not hasattr(opt, 'H'):
        opt.H = 512
        opt.W = 512
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    
    if opt.init_img != None:
        assert os.path.isfile(opt.init_img)
        init_image, h, w, opt = load_img(opt)
        init_image = init_image.to(opt.device)
        opt.H, opt.W = h , w
        
        # init_image = repeat(init_image, '1 ... -> b ...', b=opt.C)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space -> tensor
        print(f'init_latent.shape = {init_latent.shape}, shape = {shape}')
    
    samples_latents, _, ts = sampler.sample(
        S=opt.steps,
        batch_size=1,
        shape=shape,
        ori_conditioning=ori_c,
        conditioning=c,
        verbose=False,
        unconditional_guidance_scale=opt.scale,
        ori_unconditional_conditioning=ori_uc,
        unconditional_conditioning=uc,
        x_T=None if opt.init_img is None else init_latent,
        features_adapter=adapter_features,
        append_to_context=append_to_context,
        cond_tau=opt.cond_tau,
        style_cond_tau=opt.style_cond_tau,
        overlay=opt.overlay, t1=opt.time_t1, t2=opt.time_t2
    )

    x_samples = model.decode_first_stage(samples_latents)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    
    

    return x_samples, ts
