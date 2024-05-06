import os

import numpy as np
import cv2

import torch
import torch.nn as nn

from .image_encoder import ImageEncoder
from .generator import Generator
from .discriminator import Discriminator
from .stap_discriminator import STAPDiscriminator
from .unet_discriminator import UnetDiscriminator

import torch.nn.functional as F
from modules.geneva.loss import HingeAdversarial, gradient_penalty, kl_penalty
from modules.geneva.utils import get_grad_norm
from utils.truncnorm import truncated_normal

from logging import getLogger

logger = getLogger(__name__)


class GeNeVAPropV1ScainModel:
    def __init__(
            self,
            # generator
            image_feat_dim=512,
            generator_sn=True,
            generator_norm="bn",
            embedding_dim=768,
            condaug_out_dim=256,
            cond_kl_reg=1.0,
            noise_dim=100,
            gen_fusion="tirg-spade",
            sta="concat",
            nhead=8,
            res_mask_post=True,
            multi_channel_gate=True,
            use_relnet=False,
            rel_enhancement=False,
            # discriminator
            discriminator_arch="standard",
            discriminator_sn=True,
            num_objects=58,
            disc_fusion="all",
            use_stap_disc=False,
            use_gate_for_stap=False,
            use_co_attention=False,
            use_fake_txt=False,
            negative_samples=False,
            negative_cp=False,
            negative_loss=False,
            negative_loss_weight=1,
            negative_auto=False,
            negative_combo=False,
            split_combo=False,
            negative_select=False,
            substitute_rate=0,

            #BT
            BT_link = False,

            # misc
            generator_lr=1e-4,
            generator_beta1=0.0,
            generator_beta2=0.9,
            discriminator_lr=4e-4,
            discriminator_beta1=0.0,
            discriminator_beta2=0.9,
            wrong_fake_ratio=0.5,
            aux_reg=10.0,
            gp_reg=10.0,
    ):

        # modules
        self.noise_dim = noise_dim
        # image encoder with eval model
        self.image_encoder = nn.DataParallel(
            ImageEncoder(
                image_feat_dim=image_feat_dim,
                norm=generator_norm,
                use_spectral_norm=generator_sn,
            )
        ).cuda()
        self.eval_image_encoder = nn.DataParallel(
            ImageEncoder(
                image_feat_dim=image_feat_dim,
                norm=generator_norm,
                use_spectral_norm=generator_sn,
            )
        ).cuda()
        self.eval_image_encoder.load_state_dict(
            self.image_encoder.state_dict())

        # generator with eval model
        self.generator = nn.DataParallel(
            Generator(
                condition_dim=embedding_dim,
                condaug_out_dim=condaug_out_dim,
                cond_kl_reg=cond_kl_reg,
                noise_dim=noise_dim,
                norm=generator_norm,
                generator_sn=generator_sn,
                fusion=gen_fusion,
                image_feat_dim=image_feat_dim,
                sta=sta,
                nhead=nhead,
                res_mask_post=res_mask_post,
                multi_channel_gate=multi_channel_gate,
                use_relnet=use_relnet,
                rel_enhancement=rel_enhancement,
                BT_link=BT_link,
            )
        ).cuda()
        self.eval_generator = nn.DataParallel(
            Generator(
                condition_dim=embedding_dim,
                condaug_out_dim=condaug_out_dim,
                cond_kl_reg=cond_kl_reg,
                noise_dim=noise_dim,
                norm=generator_norm,
                generator_sn=generator_sn,
                fusion=gen_fusion,
                image_feat_dim=image_feat_dim,
                sta=sta,
                nhead=nhead,
                res_mask_post=res_mask_post,
                multi_channel_gate=multi_channel_gate,
                use_relnet=use_relnet,
                rel_enhancement=rel_enhancement,
                BT_link=BT_link
            )
        ).cuda()
        self.eval_generator.load_state_dict(
            self.generator.state_dict())

        # discriminator with eval model
        self.discriminator_arch = discriminator_arch
        self.use_stap_disc = use_stap_disc
        if self.discriminator_arch == "standard":
            if self.use_stap_disc:
                self.discriminator = nn.DataParallel(
                    STAPDiscriminator(
                        condition_dim=embedding_dim,
                        discriminator_sn=discriminator_sn,
                        aux_detection_dim=num_objects,
                        fusion=disc_fusion,
                        d_model=512,
                        nhead=8,
                        use_gate_for_stap=use_gate_for_stap,
                        use_co_attention=use_co_attention,
                    )
                ).cuda()
                self.eval_discriminator = nn.DataParallel(
                    STAPDiscriminator(
                        condition_dim=embedding_dim,
                        discriminator_sn=discriminator_sn,
                        aux_detection_dim=num_objects,
                        fusion=disc_fusion,
                        d_model=512,
                        nhead=8,
                        use_gate_for_stap=use_gate_for_stap,
                        use_co_attention=use_co_attention,
                    )
                ).cuda()
            else:
                self.discriminator = nn.DataParallel(
                    Discriminator(
                        condition_dim=embedding_dim,
                        discriminator_sn=discriminator_sn,
                        aux_detection_dim=num_objects,
                        fusion=disc_fusion,
                    )
                ).cuda()
                self.eval_discriminator = nn.DataParallel(
                    Discriminator(
                        condition_dim=embedding_dim,
                        discriminator_sn=discriminator_sn,
                        aux_detection_dim=num_objects,
                        fusion=disc_fusion,
                    )
                ).cuda()
        elif self.discriminator_arch == "unet":
            # ToDo
            assert not self.use_stap_disc
            assert disc_fusion == "subtract"
            self.discriminator = nn.DataParallel(
                UnetDiscriminator(
                    condition_dim=embedding_dim,
                    discriminator_sn=discriminator_sn,
                    aux_detection_dim=num_objects,
                )
            ).cuda()
            self.eval_discriminator = nn.DataParallel(
                UnetDiscriminator(
                    condition_dim=embedding_dim,
                    discriminator_sn=discriminator_sn,
                    aux_detection_dim=num_objects,
                )
            ).cuda()
        else:
            raise ValueError
        self.eval_discriminator.load_state_dict(
            self.discriminator.state_dict())

        # optimizers
        parameters = list(self.image_encoder.parameters())
        parameters += list(self.generator.parameters())
        self.generator_optimizer = torch.optim.Adam(
            parameters,
            lr=generator_lr,
            betas=(generator_beta1, generator_beta2),
            weight_decay=0.0,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            betas=(discriminator_beta1, discriminator_beta2),
            weight_decay=0.0,
        )

        # criterion
        self.wrong_fake_ratio = wrong_fake_ratio
        self.aux_reg = aux_reg
        self.gp_reg = gp_reg
        self.cond_kl_reg = cond_kl_reg
        self.criterion = HingeAdversarial()
        self.aux_criterion = nn.DataParallel(nn.BCEWithLogitsLoss()).cuda()

        # flags
        self.use_fake_txt = use_fake_txt
        self.negative_samples = negative_samples
        self.negative_cp = negative_cp
        self.negative_loss = negative_loss
        self.negative_loss_weight = negative_loss_weight
        self.negative_auto = negative_auto
        self.negative_combo = negative_combo
        self.split_combo = split_combo
        self.negative_select = negative_select
        self.substitute_rate = substitute_rate
        self.rel_enhancement = rel_enhancement
        self.BT_link = BT_link

    def predict_batch(self, batch, save_path, num_parallel=10, rel_mask=None):
        # NOTE: eval mode of GAN is sometimes disabled as a technique.
        # eval mode made images collapsed, but score is good...

        batch_size = batch["turns_image"].size(0)
        max_dialog_length = batch["turns_image"].size(1)

        prev_image = batch["background"]
        C, H, W = prev_image.size()
        prev_image = prev_image.unsqueeze(0)
        prev_image = prev_image.repeat(batch_size * num_parallel, 1, 1, 1)

        gen_images_parallel = []
        gen_realities_parallel = []
        gt_images = []

        for t in range(max_dialog_length):
            text_embedding = batch["turns_text_embedding"][:, t]
            word_embeddings = batch["turns_word_embeddings"][:, t]
            text_length = batch["turns_text_length"][:, t]

            # copy N tensors for parallel search of SCAIN
            # (B * N) = [x1_1, x1_2, ..., x1_N, x2_1, ...]
            # <=>
            # (B, N) = [[x1_1, x1_2, ..., x1_N], [x2_1, ...]]
            # (B, N)[0] = [x1_1, x1_2, ..., x1_N]
            text_embedding = torch.repeat_interleave(
                text_embedding, num_parallel, dim=0)
            word_embeddings = torch.repeat_interleave(
                word_embeddings, num_parallel, dim=0)
            text_length = torch.repeat_interleave(
                text_length, num_parallel, dim=0)

            if "turns_rel_enhancement" in batch:
                rel_mask = batch["turns_rel_enhancement"][:, t]
                rel_mask = torch.repeat_interleave(
                    rel_mask, num_parallel, dim=0)
            else:
                rel_mask = None

            with torch.no_grad():
                # add small noise for stability
                prev_image += torch.rand_like(prev_image) / 64.0

                # extract feature map
                # img_feat_map.shape = (B * N, 512, 16, 16)
                img_feat_map = self.image_encoder(prev_image)

                # truncation trick when prediction
                # SCAIN: generate N candidates for parallel search
                # z.shape = (B * N, self.noise_dim)
                z = truncated_normal(
                    (batch_size * num_parallel, self.noise_dim),
                    threshold=2.0,
                ).astype(np.float32)
                z = torch.from_numpy(z)

                # fake_image.shape = (B * N, 3, 128, 128)
                # [x1_1, x1_2, ..., x1_N, x2_1, ...]
                # logger.debug(
                #     f"[input shapes of turn {t}] "
                #     f"z: {z.size()}, "
                #     f"text_embedding: {text_embedding.size()}, "
                #     f"img_feat_map: {img_feat_map.size()}, "
                #     f"word_embeddings: {word_embeddings.size()}, "
                #     f"text_length: {text_length.size()}, "
                #     f"prev_image: {prev_image.size()}"
                # )
                fake_image, _, _, _ = self.eval_generator(
                    z,
                    text_embedding,
                    img_feat_map,
                    word_embeddings,
                    text_length,
                    rel_mask
                )
                # d_real.shape = (B * N, 1)
                # [d1_1, d1_2, ..., d1_N, d2_1, ...]
                if self.use_stap_disc:
                    d_real = self.eval_discriminator(
                        prev_image,
                        fake_image,
                        word_embeddings,
                        text_length,
                        text_embedding,
                    )[0]
                else:
                    d_real = self.eval_discriminator(
                        prev_image,
                        fake_image,
                        text_embedding,
                    )[0]
                # logger.debug(
                #     f"[output shapes of turn {t}] "
                #     f"fake_image: {fake_image.size()}, "
                #     f"d_real: {d_real.size()}"
                # )

                # to next step
                prev_image = fake_image

                # (B, N, ...)
                # (B, N) = [[x1_1, x1_2, ..., x1_N], [x2_1, ...]]
                # (B, N)[0] = [x1_1, x1_2, ..., x1_N]
                fake_image = fake_image.view(batch_size, num_parallel, C, H, W)
                d_real = d_real.view(batch_size, num_parallel)

                # L of (B, N, ...)
                gen_images_parallel.append(fake_image)
                gen_realities_parallel.append(d_real)

                # L of (B, C, H, W)
                gt_images.append(batch["turns_image"][:, t])

        # SCAIN: select most realistic dialogs from num_parallel candidates.
        # (L, B, N, C, H, W)
        gen_images_parallel = torch.stack(gen_images_parallel, dim=0)
        gen_realities_parallel = torch.stack(gen_realities_parallel, dim=0)
        # --> images.shape = (B, L, N, C, H, W)
        # --> realities.shape = (B, L, N)
        gen_images_parallel = gen_images_parallel.permute(1, 0, 2, 3, 4, 5)
        gen_realities_parallel = gen_realities_parallel.permute(1, 0, 2)

        # B of (L, C, H, W)
        gen_images = []
        dialogs = batch["dialogs"]  # (B, l) no zero padding
        for b in range(len(dialogs)):
            gen_image = gen_images_parallel[b]  # (L, N, C, H, W)
            gen_reality = gen_realities_parallel[b]  # (L, N)

            # len(dialogs[b]) = dialog length of a sample
            idx = len(dialogs[b]) - 1
            final_choice = gen_reality[idx].argmax()
            gen_image = gen_image[:, final_choice, ...]
            gen_images.append(gen_image)

        # (B, L, C, H, W) --> (L, B, C, H, W)
        gen_images = torch.stack(gen_images, dim=0)
        gen_images = gen_images.permute(1, 0, 2, 3, 4)

        self._save_predictions(
            gen_images,  # (L, B, C, H, W)
            gt_images,  # (L, B, C, H, W)
            batch["dialogs"],  # (B, l) no zero padding
            batch["scene_id"],  # (B)
            save_path,
        )

    def _save_predictions(self, gen_images, gt_images, dialogs, scene_ids, save_path):
        # i = index in batch
        for i, scene in enumerate(scene_ids):
            # save_gen_subpath: images_{split}/{scene_id}/{turn_id}.png
            # save_gt_subpath: images_{split}/{scene_id}_gt/{turn_id}.png
            save_gen_subpath = os.path.join(save_path, str(scene))
            save_gt_subpath = os.path.join(save_path, str(scene) + "_gt")
            os.makedirs(save_gen_subpath, exist_ok=True)
            os.makedirs(save_gt_subpath, exist_ok=True)

            with open(os.path.join(save_path, str(scene) + ".txt"), "w") as f:
                for j, text in enumerate(dialogs[i]):
                    f.write(f"{j}: {text}\n")

            for t in range(len(gen_images)):
                # len(dialogs[i]) = dialog length of a sample
                if t >= len(dialogs[i]):
                    continue

                # gen_images.shape: (L, B, C, H, W)
                gen = (gen_images[t][i].data.cpu().numpy() + 1) * 128
                gen = np.clip(gen.astype(np.uint8), 0, 255)
                # gen.shape: (C, H, W) -> (H, W, C)
                gen = gen.transpose(1, 2, 0)[..., ::-1]  # convert to bgr

                gt = (gt_images[t][i].data.cpu().numpy() + 1) * 128
                gt = np.clip(gt.astype(np.uint8), 0, 255)
                gt = gt.transpose(1, 2, 0)[..., ::-1]

                cv2.imwrite(
                    os.path.join(save_gen_subpath, '{}.png'.format(t)), gen)
                cv2.imwrite(
                    os.path.join(save_gt_subpath, '{}.png'.format(t)), gt)

    def train_batch(self, batch):
        batch_size = batch["source_image"].size(0)

        source_image = batch["source_image"]
        target_image = batch["target_image"]
        text_embedding = batch["text_embedding"]
        word_embeddings = batch["word_embeddings"]
        text_length = batch["text_length"]
        added_objects = batch["added_objects"]

        if self.negative_samples:
            text_neg_embedding = batch["text_neg_embedding"]
            word_neg_embeddings = batch["word_neg_embeddings"]
            text_neg_length = batch["text_neg_length"]

        if self.negative_cp:
            rand_idx = torch.randperm(batch_size)
            text_cp_embedding = batch["text_neg_embedding"][rand_idx]
            word_cp_embeddings = batch["word_neg_embeddings"][rand_idx]
            text_cp_length = batch["text_neg_length"][rand_idx]

        rand_select = torch.rand(1)[0]
        if self.negative_combo and self.split_combo == False:
            if rand_select > 0.5:
                rand = torch.rand(word_embeddings.shape)
                mask_arr = rand < self.substitute_rate  # [batch_size, 91, 768] this time ignore out of bounding part
                word_neg_embeddings = word_embeddings
                word_neg_embeddings[mask_arr == True] = 0
                text_neg_embedding = text_embedding
                text_neg_length = text_length  # [batch_size, ] [87, 81,..., 81, 79,
            else:
                rand_idx = torch.randperm(batch_size)
                word_neg_embeddings = word_embeddings[rand_idx]
                text_neg_embedding = text_embedding[rand_idx]
                text_neg_length = text_length[rand_idx]

        elif self.negative_combo and self.split_combo:
            rand_idx = torch.randperm(batch_size)
            word_neg_embeddings = word_embeddings[rand_idx]
            text_neg_embedding = text_embedding[rand_idx]
            text_neg_length = text_length[rand_idx]

            rand = torch.rand(word_embeddings.shape)
            mask_arr = rand < self.substitute_rate  # [batch_size, 91, 768] this time ignore out of bounding part
            word_msk_embeddings = word_embeddings
            word_msk_embeddings[mask_arr == True] = 0
            text_msk_embedding = text_embedding
            text_msk_length = text_length  # [batch_size, ] [87, 81,..., 81, 79,

        else:
            if self.negative_auto:
                rand = torch.rand(word_embeddings.shape)
                mask_arr = rand < self.substitute_rate  # [batch_size, 91, 768] this time ignore out of bounding part
                word_neg_embeddings = word_embeddings
                word_neg_embeddings[mask_arr == True] = 0
                text_neg_embedding = text_embedding
                text_neg_length = text_length  # [batch_size, ] [87, 81,..., 81, 79,
            if self.negative_select:
                rand_idx = torch.randperm(batch_size)
                word_neg_embeddings = word_embeddings[rand_idx]
                text_neg_embedding = text_embedding[rand_idx]
                text_neg_length = text_length[rand_idx]

        if self.rel_enhancement:
            rel_mask = batch["rel_enhancement"]
        else:
            rel_mask = None

        # data augmentation (add slight noise)
        source_image += torch.rand_like(source_image) / 64.0
        target_image += torch.rand_like(target_image) / 64.0

        # generate predict target image
        src_img_feat_map = self.image_encoder(source_image)
        z = torch.randn(
            (batch_size, self.noise_dim),
            dtype=torch.float32,
        )
        fake_image, mu, logvar, _ = self.generator(
            z,
            text_embedding,
            src_img_feat_map,
            word_embeddings,
            text_length,
            rel_mask
        )
        if self.negative_loss and self.split_combo == False:
            fake_image_negative, _, _, _ = self.generator(
                z,
                text_neg_embedding,
                src_img_feat_map,
                word_neg_embeddings,
                text_neg_length,
                rel_mask
            )
        elif self.negative_loss and self.split_combo:
            fake_image_negative, _, _, _ = self.generator(
                z,
                text_msk_embedding,
                src_img_feat_map,
                word_msk_embeddings,
                text_msk_length,
                rel_mask
            )

        if self.negative_cp:
            fake_image_cp, _, _, _ = self.generator(
                z,
                text_cp_embedding,
                src_img_feat_map,
                word_cp_embeddings,
                text_cp_length,
                rel_mask
            )
        else:
            fake_image_cp = None


        # discriminator & backward & step
        if self.negative_samples or self.negative_auto or self.negative_select or self.negative_combo or self.negative_cp:
            d_loss, aux_loss, d_grad = self._optimize_discriminator(
                target_image,
                fake_image.detach(),
                source_image,
                text_embedding,
                word_embeddings,
                text_length,
                added_objects,
                text_neg_embedding,
                word_neg_embeddings,
                text_neg_length,
            )
            # generator through discriminator & backward & step
            g_loss, g_grad, ie_grad = self._optimize_generator(
                fake_image,
                source_image,
                text_embedding,
                word_embeddings,
                text_length,
                added_objects,
                mu,
                logvar,
                fake_image_negative,
                fake_image_cp
            )
        else:
            d_loss, aux_loss, d_grad = self._optimize_discriminator(
                target_image,
                fake_image.detach(),
                source_image,
                text_embedding,
                word_embeddings,
                text_length,
                added_objects,
            )
            # generator through discriminator & backward & step
            g_loss, g_grad, ie_grad = self._optimize_generator(
                fake_image,
                source_image,
                text_embedding,
                word_embeddings,
                text_length,
                added_objects,
                mu,
                logvar,
            )

        # update eval image_encoder
        for param_src, param_trg in zip(self.image_encoder.parameters(), self.eval_image_encoder.parameters()):
            param_trg.data.mul_(0.999).add_(0.001 * param_src.data)
        for buffer_src, buffer_trg in zip(self.image_encoder.buffers(), self.eval_image_encoder.buffers()):
            buffer_trg.data.mul_(0.999).add_(0.001 * buffer_src.data)

        # update eval generator
        for param_src, param_trg in zip(self.generator.parameters(), self.eval_generator.parameters()):
            param_trg.data.mul_(0.999).add_(0.001 * param_src.data)
        for buffer_src, buffer_trg in zip(self.generator.buffers(), self.eval_generator.buffers()):
            buffer_trg.data.mul_(0.999).add_(0.001 * buffer_src.data)

        # update eval discriminator (for scain)
        for param_src, param_trg in zip(self.discriminator.parameters(), self.eval_discriminator.parameters()):
            param_trg.data.mul_(0.999).add_(0.001 * param_src.data)
        for buffer_src, buffer_trg in zip(self.discriminator.buffers(), self.eval_discriminator.buffers()):
            buffer_trg.data.mul_(0.999).add_(0.001 * buffer_src.data)

        # make outputs
        outputs = {
            "scalar": {
                "d_loss": d_loss,
                "g_loss": g_loss,
                "aux_loss": aux_loss,
                "discriminator_gradient": d_grad,
                "generator_gradient": g_grad,
                "image_gradient": ie_grad,
            },
            "image": {
                "source_image": source_image[:4].detach().cpu().numpy(),
                "teller_image": target_image[:4].detach().cpu().numpy(),
                "drawer_image": fake_image[:4].detach().cpu().numpy(),
            },
            "dialog": batch["utter"][:4],
            "others": {}
        }
        return outputs

    def _optimize_discriminator(
            self,
            real_image,
            fake_image,
            prev_image,
            text_embedding,
            word_embeddings,
            text_length,
            added_objects,
            text_neg_embedding=None,
            word_neg_embeddings=None,
            text_neg_length=None,

    ):
        '''

        Parameters
        ----------
        real_image [64, 3, 128, 128] #[batch, channel, dim, dim] single img
        fake_image [64, 3, 128, 128]
        prev_image [64, 3, 128, 128]
        text_embedding [64, 768]
        word_embeddings [64, 91, 768]
        text_length
        added_objects

        Returns
        -------

        '''

        # make wrong image-text pair
        # slide image data to left (0, 1, ..., N-1) -> (1, .., N-1, 0)
        # ToDo
        if self.use_fake_txt and not self.negative_samples:
            fake_text_embedding = torch.cat((text_embedding[1:], text_embedding[0:1]), dim=0)
            wrong_image = real_image
            wrong_prev_image = prev_image
        elif self.negative_samples or self.negative_auto or self.negative_select or self.negative_combo:
            fake_text_embedding = text_neg_embedding
            wrong_image = real_image
            wrong_prev_image = prev_image
        else:
            fake_text_embedding = text_embedding
            wrong_image = torch.cat((real_image[1:], real_image[0:1]), dim=0)
            wrong_prev_image = torch.cat((prev_image[1:], prev_image[0:1]), dim=0)

        self.discriminator.zero_grad()
        real_image.requires_grad_()

        # feed discriminator
        if self.discriminator_arch == "standard":
            if self.use_stap_disc:
                d_real, aux_real = self.discriminator(
                    prev_image, real_image, word_embeddings, text_length, text_embedding)
                d_fake, _ = self.discriminator(
                    prev_image, fake_image, word_embeddings, text_length, text_embedding)
                d_wrong, _ = self.discriminator(
                    wrong_prev_image, wrong_image, word_embeddings, text_length, text_embedding)
            else:
                d_real, aux_real = self.discriminator(
                    prev_image, real_image, text_embedding)
                d_fake, _ = self.discriminator(
                    prev_image, fake_image, text_embedding)
                d_wrong, _ = self.discriminator(
                    wrong_prev_image, wrong_image, text_embedding)
        elif self.discriminator_arch == "unet":
            # ToDo
            d_real, du_real, aux_real = self.discriminator(
                prev_image, real_image, text_embedding)
            d_fake, du_fake, _ = self.discriminator(
                prev_image, fake_image, text_embedding)
            d_wrong, _, _ = self.discriminator(
                wrong_prev_image, wrong_image, fake_text_embedding)
        # loss
        d_loss = self.criterion.discriminator(
            d_real,
            d_fake,
            d_wrong,
            self.wrong_fake_ratio,
        )
        if self.discriminator_arch == "unet":
            d_loss += self.criterion.discriminator(
                du_real,
                du_fake,
                wrong=None,  # unet-decoder for image-reality
                wrong_weight=None,
            )

        if self.aux_reg > 0.0:
            # ToDo
            # aux_real [64, 24]
            # added_objects [64, 24] [batch, total obj]
            aux_loss = self.aux_criterion(aux_real, added_objects).mean()
            d_loss += self.aux_reg * aux_loss
        else:
            aux_loss = 0

        d_loss.backward(retain_graph=True)
        if self.gp_reg > 0.0:
            reg = self.gp_reg * gradient_penalty(d_real, real_image)
            reg.backward(retain_graph=True)

        d_grad = get_grad_norm(self.discriminator.parameters())
        self.discriminator_optimizer.step()

        d_loss = d_loss.item()
        aux_loss = aux_loss.item() if self.aux_reg > 0.0 else aux_loss
        d_grad = d_grad.item()

        return d_loss, aux_loss, d_grad

    def _optimize_generator(
            self,
            fake_image,
            prev_image,
            text_embedding,
            word_embeddings,
            text_length,
            added_objects,
            mu,
            logvar,
            fake_image_negative=None,
            fake_image_cp=None,

    ):
        self.image_encoder.zero_grad()
        self.generator.zero_grad()

        # feed discriminator
        if self.discriminator_arch == "standard":
            if self.use_stap_disc:
                d_fake, aux_fake = self.discriminator(
                    prev_image, fake_image, word_embeddings, text_length, text_embedding)
            else:
                d_fake, aux_fake = self.discriminator(
                    prev_image, fake_image, text_embedding)
        elif self.discriminator_arch == "unet":
            d_fake, du_fake, aux_fake = self.discriminator(
                prev_image, fake_image, text_embedding)

        # loss
        g_loss = self.criterion.generator(d_fake)
        if self.discriminator_arch == "unet":
            g_loss += self.criterion.generator(du_fake)

        if self.aux_reg > 0.0:
            aux_loss = self.aux_criterion(aux_fake, added_objects).mean()
            g_loss += self.aux_reg * aux_loss

        if self.cond_kl_reg > 0.0:
            kl_loss = kl_penalty(mu, logvar)
            g_loss += self.cond_kl_reg * kl_loss

        if self.negative_loss:
            # Fake image [64, 3, 128, 128] -> [64, 3, 128*128]
            g_loss += self.negative_loss_weight * F.cosine_similarity(
                fake_image.reshape(fake_image.size(0), fake_image.size(1), fake_image.size(2) * fake_image.size(3)),
                fake_image_negative.reshape(fake_image.size(0), fake_image.size(1),
                                            fake_image.size(2) * fake_image.size(3)), dim=-1).mean()

        if self.negative_cp:
            g_loss += self.negative_loss_weight * F.cosine_similarity(
                fake_image.reshape(fake_image.size(0), fake_image.size(1), fake_image.size(2) * fake_image.size(3)),
                fake_image_cp.reshape(fake_image.size(0), fake_image.size(1),
                                      fake_image.size(2) * fake_image.size(3)), dim=-1).mean()

        g_loss.backward(retain_graph=True)


        g_grad = get_grad_norm(self.generator.parameters())
        ie_grad = get_grad_norm(self.image_encoder.parameters())
        self.generator_optimizer.step()

        g_loss = g_loss.item()
        g_grad = g_grad.item()
        ie_grad = ie_grad.item()

        return g_loss, g_grad, ie_grad

    def get_snapshot(self):
        snapshot = {
            "eval_image_encoder": self.eval_image_encoder.state_dict(),
            "eval_generator": self.eval_generator.state_dict(),
            "eval_discriminator": self.eval_discriminator.state_dict(),
        }
        return snapshot
