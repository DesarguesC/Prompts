import importlib
import math

import cv2
import torch
from einops import rearrange
import numpy as np

import os
from safetensors.torch import load_file

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
import PIL


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('assets/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


checkpoint_dict_replacements = {
    'cond_stage_model.transformer.text_model.embeddings.': 'cond_stage_model.transformer.embeddings.',
    'cond_stage_model.transformer.text_model.encoder.': 'cond_stage_model.transformer.encoder.',
    'cond_stage_model.transformer.text_model.final_layer_norm.': 'cond_stage_model.transformer.final_layer_norm.',
}


def transform_checkpoint_dict_key(k):
    for text, replacement in checkpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k


def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


def read_state_dict(checkpoint_file, print_global_state=False):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        pl_sd = load_file(checkpoint_file, device='cpu')
    else:
        pl_sd = torch.load(checkpoint_file, map_location='cpu')

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    sd = read_state_dict(ckpt)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if 'anything' in ckpt.lower() and vae_ckpt is None:
        vae_ckpt = 'models/anything-v4.0.vae.pt'

    if vae_ckpt is not None and vae_ckpt != 'None':
        print(f"Loading vae model from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")
        if "global_step" in vae_sd:
            print(f"Global Step: {vae_sd['global_step']}")
        sd = vae_sd["state_dict"]
        m, u = model.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

    model.cuda()
    model.eval()
    return model

def load_img(opt):
    path = opt.init_img
    resize_short_edge = opt.resize_short_edge
    max_resolution = opt.max_resolution
    
    image = Image.open(path).convert("RGB")
    # image = cv2.imread('images/dog.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")

    image = np.asarray(image, dtype=np.float32)
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    
    assert opt.fac != None
    assert opt.fac >= 0
    
    h *= opt.fac
    w *= opt.fac
    
    image = cv2.resize(image, (w,h), interpolation=cv2.INTER_LANCZOS4)
    
    image = np.array(image).astype(np.float32) / 255.0
    # print('before transpose: ', image.shape)
    image = image[None].transpose(0, 3, 2, 1)
    image = torch.from_numpy(image)
    print(image.shape)
    
    return 2.*image - 1., h, w, opt



def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None, opt=None):
    h, w = image.shape[:2]
    image = np.ndarray(image, dtype=np.float32)
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    
    if opt is not None:
        try:
            h //= opt.fac
            w //= opt.fac
        except:
            raise NotImplementedError
    
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image



def get_resize_shape(image_shape, max_resolution=512 * 512, resize_short_edge=None, resize_method=cv2.INTER_LANCZOS4) -> tuple:
    # print('resize: ', image_shape)
    h, w = image_shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = ((h * k) // 64) * 64
    w = ((w * k) // 64) * 64
    return h, w


# make uc and prompt shapes match via padding for long prompts
null_cond = None

def fix_cond_shapes(model, ori_prompt_condition, prompt_condition, uc, overlay=True, use_weights=False, dim=1) \
    -> (torch.tensor, torch.tensor, list, list):
    # uc here will never be a list
    
    if uc is None:
        return ori_prompt_condition, prompt_condition, uc
    global null_cond
    if null_cond is None:
        null_cond = model.get_learned_conditioning([""])
    
    ori_uc = uc
    while ori_prompt_condition.shape[1] > ori_uc.shape[1]:
        ori_uc = torch.cat((ori_uc, null_cond.repeat((ori_uc.shape[0], 1, 1))), axis=1)
    while ori_prompt_condition.shape[1] < ori_uc.shape[1]:
        ori_prompt_condition = torch.cat((ori_prompt_condition, null_cond.repeat((ori_prompt_condition.shape[0], 1, 1))), axis=1)
    # print('no to cut: ', prompt_condition.shape, uc.shape)
        
    
    assert isinstance(prompt_condition, list), 'list type error'
    # a cut prompt
    uc_ = []
    condition_ = []

    for i in range(len(prompt_condition)):
        prompt = prompt_condition[i]
        x = uc
        # print('op: ', prompt)
        while prompt.shape[1] > x.shape[1]:
            x = torch.cat((x, null_cond.repeat((x.shape[0], 1, 1))), axis=1)
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        uc_.append(x)
        while prompt.shape[1] < x.shape[1]:
            prompt = torch.cat((prompt, null_cond.repeat((prompt.shape[0], 1, 1))), axis=1)
        condition_.append(prompt)
    assert len(uc_) == len(condition_), 'length error when fixing'
        
        # prompt_condition = torch.cat([pp for pp in condition_], dim) if not use_weights else \
        #                                             torch.cat([i*1.*condition_[i] for i in range(len(condition_))], dim)
        
        # uc = torch.cat([uu for uu in uc_], dim)
        
    return ori_prompt_condition, ori_uc, condition_, uc_
