import os

import cv2
import torch
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models, str2bool)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)

torch.set_grad_enabled(False)


def main():
    supported_cond = [e.name for e in ExtraCondition]
    parser = get_base_argument_parser()
    parser.add_argument(
        '--which_cond',
        type=str,
        default=None,
        choices=supported_cond,
        help='which condition modality you want to test',
    )
    parser.add_argument(
        '--fac',
        type=int,
        default=1,
        help='extand imput image size',
    )
    parser.add_argument(
        '--overlay',
        type=str2bool,
        default=0,
        help='whether to overlay prompts that are cut',   # cut symbol: '|'
    )
    # parser.add_argument(
    #     '--o_dim',
    #     type=int,
    #     default=0,
    #     help='torch cat dim'
    # )
    # parser.add_argument(
    #     '--wei',
    #     type=str2bool,
    #     default=0,
    #     help='whether to add weights to each layer when using overlay mode'
    # )
    parser.add_argument(
        '--time_t1',
        type=int,
        default=10,
        help='clip choice 1'
    )
    parser.add_argument(
        '--time_t2',
        type=int,
        default=20,
        help='clip choice 2'
    )
    
    
    opt = parser.parse_args()
    which_cond = opt.which_cond
    if opt.outdir is None:
        opt.outdir = f'outputs/test-{which_cond}/' + opt.adapter_ckpt.split('/')[-1].strip('.pth')
    os.makedirs(opt.outdir, exist_ok=True)
    if opt.resize_short_edge is None:
        print(f"you don't specify the resize_short_edge, so the maximum resolution is set to {opt.max_resolution}")
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    assert max(opt.time_t1, opt.time_t2) <= opt.steps and opt.time_t1 <= opt.time_t2, 'time step error'
    
    # support two test mode: single image test, and batch test (through a txt file)
    if opt.prompt.endswith('.txt'):
        assert opt.prompt.endswith('.txt')
        image_paths = []
        prompts = []
        with open(opt.prompt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                image_paths.append(line.split('; ')[0])
                prompts.append(line.split('; ')[1])
    else:
        image_paths = [opt.cond_path]  # can be [None]
        prompts = [opt.prompt]
    print(image_paths)

    # prepare models
    sd_model, sampler = get_sd_models(opt)
    
    adapter = get_adapters(opt, getattr(ExtraCondition, which_cond)) if opt.cond_path != None else None
    cond_model = None
    if opt.cond_inp_type == 'image' and opt.cond_path != None:
        cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond))
    else:
        cond_model = None
    
    process_cond_module = getattr(api, f'get_cond_{which_cond}') if opt.cond_path != None else None

    # inference
    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):
        for test_idx, (cond_path, prompt) in enumerate(zip(image_paths, prompts)):
            seed_everything(opt.seed)
            for v_idx in range(opt.n_samples):
                # seed_everything(opt.seed+v_idx+test_idx)
                cond = process_cond_module(opt, cond_path, opt.cond_inp_type, cond_model) if cond_model != None else None

                base_count = len(os.listdir(opt.outdir)) // 2
                if which_cond != None:
                    cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_{which_cond}.png'), tensor2img(cond))

                adapter_features, append_to_context = get_adapter_feature(cond, adapter) if cond != None else (None, None)
                opt.prompt = prompt
                result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)
                cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_{opt.steps}_({opt.time_t1},{opt.time_t2}).png'), tensor2img(result))


if __name__ == '__main__':
    main()
