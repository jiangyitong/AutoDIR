import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from ldm.models.autoencoder import (AutoencoderKL, IdentityFirstStage,
                                    VQModelInterface)
from ldm.modules.diffusionmodules.util import (extract_into_tensor,
                                               make_beta_schedule, noise_like)
from ldm.modules.encoders.modules import AbstractEncoder
from ldm.util import (count_params, default, exists, instantiate_from_config,
                      isimage, ismap, log_txt_as_img, mean_flat)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer

from stable_diffusion.ldm.models.diffusion.ddpm_edit import LatentDiffusion
from stable_diffusion.ldm.util import default

artifact_category = [
    'noise', 'blur', 'rain', 'underexposure', 'haze', 'low resolution',
    'raindrop', 'no'
]

def cliploss(p, g):
    loss = 0
    for i in range(p.size(1)):
        p_i = p[:, i]
        g_i = g[:, i]
        g_i = g_i.view(-1, 1)
        p_i = p_i.view(-1, 1)
        loss_i = torch.sqrt(p_i * g_i + 1e-8)
        loss = loss + loss_i
    loss = 1 - loss
    loss = loss / p.size(1)
    return torch.mean(loss)


def predict_logit(x, text, gt_logits, model):
    batch_size = x.size(0)
    num_patch = 1

    logits_per_image, full_x = model.forward(x, text)

    logits_per_image = logits_per_image.t()

    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)

    logits_per_image = logits_per_image.mean(1)

    logits_distortion_gumbel = F.softmax(logits_per_image, dim=1)

    text_distortion = torch.mm(gt_logits, full_x.reshape(8, -1))

    text_arg = text_distortion.reshape(batch_size, 77, 768)

    return logits_distortion_gumbel, text_arg


class CLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self,
                 version="openai/clip-vit-large-patch14",
                 device="cuda",
                 max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.clipmodel = CLIPModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.processor = CLIPFeatureExtractor.from_pretrained(version)
        self.clipmodel.eval()

    def forward(self, image, text):
        tokens = self.tokenizer(text,
                                truncation=True,
                                max_length=self.max_length,
                                return_length=False,
                                return_overflowing_tokens=False,
                                padding="max_length",
                                return_tensors="pt")

        image = (image + 1) * 255. / 2
        images = self.processor(images=image,
                                do_resize=False,
                                return_tensors="pt")
        tokens.data['pixel_values'] = images.data['pixel_values']
        for k in tokens:
            if hasattr(tokens[k], 'device'):
                tokens[k] = tokens[k].to(self.device)
        outputs = self.clipmodel(**tokens)

        logits_per_text = outputs.logits_per_text
        text_outputs = outputs.text_model_output.last_hidden_state
        return logits_per_text.float(), text_outputs


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


from NAFNet.basicsr.models.archs.NAFNet_arch import NAFNet

class NAFNet_Combine(NAFNet):
    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        # weight of SCM
        weight = 1
        x = x * weight + inp[:, 3:6, :, :]

        return x[:, :, :H, :W]


class CycleLatentDiffusion(LatentDiffusion):

    def __init__(self,  *args, **kwargs):
        super().__init__(if_init_cond=False, *args, **kwargs)
        self.model_assessment = CLIPEmbedder(device=self.device)
        self.NAFNet = NAFNet_Combine(img_channel=6,
                                     width=64,
                                     enc_blk_nums=[2, 2, 4, 8],
                                     middle_blk_num=12,
                                     dec_blk_nums=[2, 2, 2, 2])

        for name, para in self.model_assessment.named_parameters():
            if 'text_model' in name:
                para.requires_grad = False
        self.model_assessment.eval()
        self.model.eval()
        for param in self.model_assessment.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):

        params_NANet = self.NAFNet.parameters()

        if self.cond_stage_trainable:
            print(
                f"{self.__class__.__name__}: Also optimizing conditioner params!"
            )
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW([
            # {'params': params_model, 'lr': lr},
            # {'params': params_assessment, 'lr': 3e-6},
            {
                'params': params_NANet,
                'lr': 1e-3
            },
            # {'params':params_combine, 'lr': 1e-3}
        ])
        if self.use_scheduler:
            assert 'target' in self.scheduler_config

            print("Setting up LambdaLR scheduler...")
            scheduler = [{
                'scheduler':
                CosineAnnealingLR(opt, T_max=4000, eta_min=1e-7),
                'interval':
                'step',
                'frequency':
                1
            }]
            return [opt], scheduler
        return opt

    def training_step(self, batch, batch_idx):

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict,
                      prog_bar=True,
                      logger=True,
                      on_step=True,
                      on_epoch=True)

        self.log("global_step",
                 self.global_step,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs',
                     lr,
                     prog_bar=True,
                     logger=True,
                     on_step=True,
                     on_epoch=False)

        return loss

    def shared_step(self, batch):
        x, c, clip_loss, xc, x_edited = self.get_input(
            batch,
            self.first_stage_key,
            return_original_edited=True,
            return_original_cond=True,
            if_cliploss=True)
        loss = self(x, c, clip_loss, x_edited, xc)
        return loss

    def forward(self, x, c, clip_loss, x_edited, xc, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0,
                          self.num_timesteps, (x.shape[0], ),
                          device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c,
                                  t=tc,
                                  noise=torch.randn_like(c.float()))

        return self.p_losses(x, c, x_edited, t, clip_loss, xc, *args, **kwargs)

    def get_input(self,
                  batch,
                  k,
                  return_first_stage_outputs=False,
                  force_c_encode=False,
                  cond_key=None,
                  return_original_cond=False,
                  bs=None,
                  return_original_edited=False,
                  uncond=0.05,
                  if_cliploss=False):
        self.model_assessment.device = self.device
        with torch.no_grad():
            x = batch[k]
            if bs is not None:
                x = x[:bs]
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            cond_key = cond_key or self.cond_stage_key
            xc = batch[cond_key]
            if bs is not None:
                # xc["c_crossattn"] = xc["c_crossattn"][:bs]
                xc["c_concat"] = xc["c_concat"][:bs]
            cond = {}

            # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
            random = torch.rand(x.size(0), device=x.device)
            input_mask = 1 - rearrange(
                (random >= uncond).float() *
                (random < 3 * uncond).float(), "n -> n 1 1 1")
            cond["c_concat"] = [
                input_mask * self.encode_first_stage(
                    (xc["c_concat"].to(self.device))).mode().detach()
            ]
        joint_texts = [
            f"A photo needs {d} artifact reduction" for d in dists_map
        ]
        edit_x = batch['edit']['c_concat']
        gt_logits = []
        for bs in range(len(batch['edited'])):
            if 'noise' in batch['edit']['c_crossattn'][bs].split(' '):
                gt_logits.append([1., 0., 0., 0., 0., 0., 0., 0.])
            elif 'blur' in batch['edit']['c_crossattn'][bs].split(' '):
                gt_logits.append([0., 1., 0., 0., 0., 0., 0., 0.])
            elif 'rain' in batch['edit']['c_crossattn'][bs].split(' '):
                gt_logits.append([0., 0., 1., 0., 0., 0., 0., 0.])
            elif 'underexposure' in batch['edit']['c_crossattn'][bs].split(
                    ' '):
                gt_logits.append([0., 0., 0., 1., 0., 0., 0., 0.])
            elif 'haze' in batch['edit']['c_crossattn'][bs].split(' '):
                gt_logits.append([0., 0., 0., 0., 1., 0., 0., 0.])
            elif 'resolution' in batch['edit']['c_crossattn'][bs].split(' '):
                gt_logits.append([0., 0., 0., 0., 0., 1., 0., 0.])
            elif 'raindrop' in batch['edit']['c_crossattn'][bs].split(' '):
                gt_logits.append([0., 0., 0., 0., 0., 0., 1., 0.])
            elif 'no' in batch['edit']['c_crossattn'][bs].split(' '):
                gt_logits.append([0., 0., 0., 0., 0., 0., 0., 1.])
        gt_logits = torch.FloatTensor(gt_logits).to(self.device)
        logits_per_image, text_arg_feature = predict_logit(
            edit_x, joint_texts, gt_logits, self.model_assessment)
        clip_loss = cliploss(logits_per_image, gt_logits)
        random = torch.rand(x.size(0), device=x.device)
        prompt_mask = rearrange(random < 2 * 0.05, "n -> n 1 1")
        null_prompt_tokens = self.model_assessment.tokenizer(
            "",
            truncation=True,
            max_length=77,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt").to(self.device)
        null_prompt_feature = self.model_assessment.clipmodel.text_model(
            **null_prompt_tokens).last_hidden_state
        cond["c_crossattn"] = [
            torch.where(prompt_mask, null_prompt_feature,
                        text_arg_feature.float())
        ]
        if if_cliploss:
            out = [z, cond, clip_loss]
        else:
            out = [z, cond]
        if return_first_stage_outputs:
            with torch.no_grad():
                xrec = self.decode_first_stage(z)
                out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        if return_original_edited:
            out.append(batch[k])
        return out

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_start_alpha = extract_into_tensor(self.sqrt_alphas_cumprod, t,
                                            x_start.shape)
        noise_beta = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t,
                                         x_start.shape)
        return (x_start_alpha * x_start +
                noise_beta * noise), x_start_alpha, noise_beta

    def p_losses(self, x_start, cond, x_edited, t, clip_loss, xc, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy, x_start_alpha, noise_beta = self.q_sample(x_start=x_start,
                                                           t=t,
                                                           noise=noise)

        model_output = self.apply_model(x_noisy, t, cond)
        model_recon = self.decode_first_stage(
            (x_noisy - noise_beta * model_output) / x_start_alpha)
        recon = x_edited

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        recon_stack = torch.cat((xc['c_concat'], model_recon), dim=1)
        result = self.NAFNet(recon_stack)
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_dict.update({f'{prefix}/clip_loss': clip_loss.mean()})

        loss_simple = self.get_loss(model_output, target,
                                    mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        loss_final = self.get_loss(result, recon, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_final': loss_final.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target,
                                 mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss += clip_loss
        # loss += loss_NafNet.mean()
        loss += loss_final.mean()
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def log_images(self,
                   batch,
                   N=4,
                   n_row=4,
                   sample=True,
                   ddim_steps=200,
                   ddim_eta=1.,
                   return_keys=None,
                   quantize_denoised=True,
                   inpaint=False,
                   plot_denoise_rows=False,
                   plot_progressive_rows=False,
                   plot_diffusion_rows=False,
                   **kwargs):

        use_ddim = False

        log = dict()
        z, c, x, xrec, xc, edited = self.get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            return_original_edited=True,
            bs=N,
            uncond=0)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reals"] = xc["c_concat"]
        log["reconstruction"] = xrec
        log["gt"] = edited
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]),
                                    batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(
                diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid,
                                       'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid,
                                       nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,
                                                         batch_size=N,
                                                         ddim=use_ddim,
                                                         ddim_steps=ddim_steps,
                                                         eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            with torch.no_grad():
                stack = torch.cat((xc["c_concat"], x_samples), dim=1)
                recon_final = self.NAFNet(stack)
            log["recon_final"] = recon_final

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(
                    self.first_stage_model, AutoencoderKL) and not isinstance(
                        self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(
                        cond=c,
                        batch_size=N,
                        ddim=use_ddim,
                        ddim_steps=ddim_steps,
                        eta=ddim_eta,
                        quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):
                    samples, _ = self.sample_log(cond=c,
                                                 batch_size=N,
                                                 ddim=use_ddim,
                                                 eta=ddim_eta,
                                                 ddim_steps=ddim_steps,
                                                 x0=z[:N],
                                                 mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c,
                                                 batch_size=N,
                                                 ddim=use_ddim,
                                                 eta=ddim_eta,
                                                 ddim_steps=ddim_steps,
                                                 x0=z[:N],
                                                 mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(
                    c,
                    shape=(self.channels, self.image_size, self.image_size),
                    batch_size=N)
            prog_row = self._get_denoise_row_from_list(
                progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {
                key + '_ema': loss_dict_ema[key]
                for key in loss_dict_ema
            }
        self.log_dict(loss_dict_no_ema,
                      prog_bar=False,
                      logger=True,
                      on_step=False,
                      on_epoch=True)
        self.log_dict(loss_dict_ema,
                      prog_bar=False,
                      logger=True,
                      on_step=False,
                      on_epoch=True)
