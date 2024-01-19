from __future__ import annotations

import math
import os
import random
import sys
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from consistencydecoder import ConsistencyDecoder
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

from stable_diffusion.ldm.util import instantiate_from_config
from wavelet_color_fix import adaptive_instance_normalization

sys.path.append("./stable_diffusion")

artifact_category = [
    'noise', 'blur', 'rain', 'underexposure', 'haze', 'low resolution',
    'raindrop', 'no'
]


class CFGDenoiser(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [
                torch.cat([
                    cond["c_crossattn"][0], uncond["c_crossattn"][0],
                    uncond["c_crossattn"][0]
                ])
            ],
            "c_concat": [
                torch.cat([
                    cond["c_concat"][0], cond["c_concat"][0],
                    uncond["c_concat"][0]
                ])
            ],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(
            cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (
                out_cond - out_img_cond) + image_cfg_scale * (out_img_cond -
                                                              out_uncond)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=True)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def predict_logit(x, text, model):
    batch_size = x.size(0)
    text_number = len(text)
    num_patch = 1

    logits_per_image, full_x = model.forward(x, text)

    logits_per_image = logits_per_image.t()

    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)

    logits_per_image = logits_per_image.mean(1)

    logits_distortion_gumbel = F.softmax(logits_per_image, dim=1)

    text_distortion = torch.mm(logits_distortion_gumbel, full_x.reshape(text_number, -1))

    text_arg = text_distortion.reshape(batch_size, 77, 768)

    main_artifact_index = torch.argmax(logits_distortion_gumbel)

    print(f"A photo needs {artifact_category[main_artifact_index]} artifact reduction")

    main_artifact = artifact_category.pop(main_artifact_index)

    return logits_distortion_gumbel, text_arg, main_artifact


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=768, type=int)
    parser.add_argument("--need-resize", action='store_true')
    parser.add_argument("--color-correct", action='store_true')
    parser.add_argument("--customize", default=None, type=str)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt",
                        default="checkpoints/autodir.ckpt",
                        type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--gt", required=False, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--decoder-consistency", action='store_true')
    parser.add_argument("--cfg-text", default=1.0, type=float)
    parser.add_argument("--cfg-image", default=1.0, type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)

    null_prompt_tokens = model.model_assessment.tokenizer(
        '',
        truncation=True,
        max_length=77,
        return_length=False,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt").to(model.device)
    tokens = null_prompt_tokens['input_ids']
    null_token = model.model_assessment.clipmodel.text_model(
        input_ids=tokens).last_hidden_state
    seed = random.randint(0, 100000) if args.seed is None else args.seed
    input = args.input
    if args.decoder_consistency:
        decoder_consistency = ConsistencyDecoder(device="cuda:0")

    name = os.path.basename(input).split('.')[0]
    print(name)
    input_image = Image.open(input).convert("RGB")
    input_image_save = input_image

    if args.gt:
        gt = Image.open(args.gt).convert("RGB")


    width, height = input_image.size
    if args.need_resize:
        if width < args.resolution or height < args.resolution:
            factor = args.resolution / min(width, height)
            factor = math.ceil(min(width, height) * factor / 64) * 64 / min(
                width, height)
            width = int((width * factor) // 64) * 64
            height = int((height * factor) // 64) * 64
            input_image = ImageOps.fit(input_image, (width, height),
                                       method=Image.Resampling.LANCZOS)
            if args.gt:
                gt = ImageOps.fit(gt, (width, height),
                                  method=Image.Resampling.LANCZOS)

            position_x = np.random.randint(0, width - args.resolution + 1)
            position_y = np.random.randint(0, height - args.resolution + 1)

            input_image = input_image.crop(
                (position_x, position_y, position_x + args.resolution,
                 position_y + args.resolution))
            input_image_save = input_image
            if args.gt:
                gt = gt.crop(
                    (position_x, position_y, position_x + args.resolution,
                     position_y + args.resolution))
        else:
            position_x = np.random.randint(0, width - args.resolution + 1)
            position_y = np.random.randint(0, height - args.resolution + 1)

            input_image = input_image.crop(
                (position_x, position_y, position_x + args.resolution,
                 position_y + args.resolution))
            input_image_save = input_image
            if args.gt:
                gt = gt.crop(
                    (position_x, position_y, position_x + args.resolution,
                     position_y + args.resolution))

    input_image_save.save(
        os.path.join(args.output, name  + '_input.png'))
    input_image = 2 * torch.tensor(
        np.array(input_image)).float() / 255 - 1

    input_image = rearrange(input_image,
                            "h w c -> 1 c h w").to(model.device)


    step_max_number = 1 if args.customize else 7

    for step_number in range(step_max_number):
        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            cond = {}
            cond["c_concat"] = [model.encode_first_stage(input_image).mode()]
            model.model_assessment.device = model.device
            if args.customize:
                customize_prompt_tokens = model.model_assessment.tokenizer(
                    args.customize,
                    truncation=True,
                    max_length=77,
                    return_length=False,
                    return_overflowing_tokens=False,
                    padding="max_length",
                    return_tensors="pt").to(model.device)
                tokens = customize_prompt_tokens['input_ids']
                customize_token = model.model_assessment.clipmodel.text_model(
                    input_ids=tokens).last_hidden_state
                cond["c_crossattn"] = [customize_token]
            else:
                joint_texts = [f"A photo needs {d} artifact reduction" for d in artifact_category]
                logits_per_image, text_arg, main_artifact = predict_logit(input_image,
                                                                          joint_texts,
                                                                          model.model_assessment)
                cond["c_crossattn"] = [text_arg]
                if main_artifact == 'no':
                    print("No artifact detected. End of the Image Restoration Process.")
                    break
            uncond = {}
            uncond["c_crossattn"] = [null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = model_wrap.get_sigmas(args.steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": args.cfg_text,
                "image_cfg_scale": args.cfg_image,
            }
            torch.manual_seed(seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg,
                                                  z,
                                                  sigmas,
                                                  extra_args=extra_args)
            if args.decoder_consistency:
                z = z / 0.18215
                x = decoder_consistency(z, schedule=[1.0])
            else:
                x = model.decode_first_stage(z)
            recon_stack = torch.cat((input_image, x), dim=1)
            result = model.NAFNet(recon_stack)
            if args.color_correct:
                correct_stable = adaptive_instance_normalization(x, input_image)
                correct_final = adaptive_instance_normalization(
                    result, input_image)
            if not args.customize:
                if main_artifact == 'low resolution':
                    input_image = x
                else:
                    input_image = result

            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")

            result = torch.clamp((result + 1.0) / 2.0, min=0.0, max=1.0)
            result = 255.0 * rearrange(result, "1 c h w -> h w c")

            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            edited_image_final = Image.fromarray(
                result.type(torch.uint8).cpu().numpy())

            if args.color_correct:
                correct_stable = torch.clamp((correct_stable + 1.0) / 2.0,
                                             min=0.0,
                                             max=1.0)
                correct_stable = 255.0 * rearrange(correct_stable,
                                                   "1 c h w -> h w c")
                correct_stable = Image.fromarray(
                    correct_stable.type(torch.uint8).cpu().numpy())

                correct_final = torch.clamp((correct_final + 1.0) / 2.0,
                                            min=0.0,
                                            max=1.0)
                correct_final = 255.0 * rearrange(correct_final,
                                                  "1 c h w -> h w c")
                correct_final = Image.fromarray(
                    correct_final.type(torch.uint8).cpu().numpy())

            edited_image.save(
                os.path.join(args.output, name +'_step_'+str(step_number)+ '_result.png'))
            edited_image_final.save(
                os.path.join(args.output, name + '_step_'+str(step_number)+'_result_w_SCM.png'))
            if args.gt:
                gt.save(
                    os.path.join(args.output, name +'_step_'+str(step_number)+ '_gt.png'))
            input_image_save.save(
                os.path.join(args.output, name +'_step_'+str(step_number)+ '_input.png'))
            if args.color_correct:
                correct_stable.save(
                    os.path.join(args.output, name +'_step_'+str(step_number)+ '_result_colorcorrect.png'))
                correct_final.save(
                    os.path.join(args.output, name +'_step_'+str(step_number)+ '_result_w_SCM_colorcorrect.png'))


if __name__ == "__main__":
    main()
