import os
import uuid
import shutil

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
#In metrics
from modules.propv1.model_scain import GeNeVAPropV1ScainModel
from modules.metrics.inception_localizer import calculate_inception_objects_accuracy
from data.codraw_propv1_dataset import CoDrawPropV1TrainDataset, codraw_propv1_train_collate_fn
from data.codraw_propv1_dataset import CoDrawPropV1EvalDataset, codraw_propv1_eval_collate_fn
from data.iclevr_propv1_dataset import ICLEVRPropV1TrainDataset, iclevr_propv1_train_collate_fn
from data.iclevr_propv1_dataset import ICLEVRPropV1EvalDataset, iclevr_propv1_eval_collate_fn
# from utils.plotter import plot_multilabel_confusion_matrix
# from utils.make_grid import make_grid_from_numpy
# from utils.consts import CODRAW_OBJS

# import wandb

from logging import getLogger
logger = getLogger(__name__)


SAVE_ROOT_DIR = "./results/experiments/"
SAVE_DIR = None


# NOTE: faster training instead of reproducibility
# torch.backends.cudnn.benchmark = True


class PropV1ScainTrainer:
    def __init__(self, cfg):
        if ("sta_concat" in cfg) and ("sta" not in cfg):
            if cfg.sta_concat:
                cfg.sta = "concat"
            else:
                cfg.sta = "none"
        if "use_stap_disc" not in cfg:
            cfg.use_stap_disc = False
        if "use_relnet" not in cfg:
            cfg.use_relnet = False
        if "use_gate_for_stap" not in cfg:
            cfg.use_gate_for_stap = False
        if "use_co_attention" not in cfg:
            cfg.use_co_attention = False
        if "discriminator_arch" not in cfg:
            cfg.discriminator_arch = "standard"
        #ToDo
        if "use_fake_txt" not in cfg:
            cfg.use_fake_txt = False
        if "negative_samples" not in cfg:
            cfg.negative_samples = False
        if "negative_cp" not in cfg:
            cfg.negative_cp = False
        if "negative_loss" not in cfg:
            cfg.negative_loss = False
        if "negative_loss_weight" not in cfg:
            cfg.negative_loss_weight = 1
        if "negative_combo" not in cfg:
            cfg.negative_combo = False
        if "split_combo" not in cfg:
            cfg.split_combo = False
        if "negative_auto" not in cfg:
            cfg.negative_auto = False
        if "negative_select" not in cfg:
            cfg.negative_select = False
        if "substitute_rate" not in cfg:
            cfg.substitute_rate = 0
        if "rel_enhancement" not in cfg:
            cfg.rel_enhancement = False
        if "BT_link" not in cfg:
            cfg.BT_link = False
        self.cfg = cfg

        # result path
        self.save_path = os.path.join(SAVE_ROOT_DIR, cfg.name)
        self.save_snapshot_dir = os.path.join(self.save_path, "snapshots/")
        os.makedirs(self.save_snapshot_dir, exist_ok=True)
        # models
        self.model = GeNeVAPropV1ScainModel(
            # generator
            image_feat_dim=cfg.image_feat_dim,
            generator_sn=cfg.generator_sn,
            generator_norm=cfg.generator_norm,
            embedding_dim=cfg.embedding_dim,
            condaug_out_dim=cfg.condaug_out_dim,
            cond_kl_reg=cfg.cond_kl_reg,
            noise_dim=cfg.noise_dim,
            gen_fusion=cfg.gen_fusion,
            sta=cfg.sta,
            nhead=cfg.nhead,
            res_mask_post=cfg.res_mask_post,
            multi_channel_gate=cfg.multi_channel_gate,
            use_relnet=cfg.use_relnet,
            rel_enhancement=cfg.rel_enhancement,
            BT_link=cfg.BT_link,
            # discriminator
            discriminator_arch=cfg.discriminator_arch,
            discriminator_sn=cfg.discriminator_sn,
            num_objects=cfg.num_objects,
            disc_fusion=cfg.disc_fusion,
            use_stap_disc=cfg.use_stap_disc,
            use_gate_for_stap=cfg.use_gate_for_stap,
            use_co_attention=cfg.use_co_attention,
            use_fake_txt=cfg.use_fake_txt,
            negative_samples=cfg.negative_samples,
            negative_cp=cfg.negative_cp,
            negative_loss=cfg.negative_loss,
            negative_loss_weight=cfg.negative_loss_weight,
            negative_auto=cfg.negative_auto,
            negative_combo=cfg.negative_combo,
            split_combo=cfg.split_combo,
            negative_select=cfg.negative_select,
            substitute_rate=cfg.substitute_rate,

            # misc
            generator_lr=cfg.generator_lr,
            generator_beta1=cfg.generator_beta1,
            generator_beta2=cfg.generator_beta2,
            discriminator_lr=cfg.discriminator_lr,
            discriminator_beta1=cfg.discriminator_beta1,
            discriminator_beta2=cfg.discriminator_beta2,
            wrong_fake_ratio=cfg.wrong_fake_ratio,
            aux_reg=cfg.aux_reg,
            gp_reg=cfg.gp_reg,
        )
        checkpoint = torch.load('/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-base2/snapshots/model_e87_i16000.pth', map_location='cpu')
        # ours '/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new/snapshots/model_e142_i26000.pth'
        # base1 '/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-base1/snapshots/model_e142_i26000.pth'
        # base2 '/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-base2/snapshots/model_e98_i18000.pth'
        # base3 '/home/peixixio/LatteGAN/results/results/experiments/exp183-geneva-LRT2I-new/snapshots/model_e131_i24000.pth'
        # latte '/home/peixixio/LatteGAN/results/experiments/exp183-geneva-LRT2I-new-base2/snapshots/model_e87_i16000.pth'
        self.model.eval_image_encoder.load_state_dict(checkpoint['eval_image_encoder'], strict=True)
        self.model.eval_image_encoder.eval()
        self.model.eval_generator.load_state_dict(checkpoint['eval_generator'], strict=True)
        self.model.eval_generator.eval()
        self.model.eval_discriminator.load_state_dict(checkpoint['eval_discriminator'], strict=True)
        self.model.eval_discriminator.eval()
        self.model.image_encoder.load_state_dict(checkpoint['eval_image_encoder'], strict=True)
        self.model.image_encoder.eval()
        self.model.generator.load_state_dict(checkpoint['eval_generator'], strict=True)
        self.model.generator.eval()
        self.model.discriminator.load_state_dict(checkpoint['eval_discriminator'], strict=True)
        self.model.discriminator.eval()

        if cfg.print_model:
            print(self.model.image_encoder)
            print(self.model.generator)
            print(self.model.discriminator)
            # for name, param in self.model.eval_image_encoder.named_parameters():
            #     print(name, param.data)
            # for name, param in self.model.image_encoder.named_parameters():
            #     print(name, param.data)

        # dataset
        if cfg.dataset == "codraw":
            # train
            self.dataset = CoDrawPropV1TrainDataset(
                dataset_path=cfg.dataset_path,
                embed_dataset_path=cfg.embed_dataset_path,
                image_size=cfg.image_size,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            self.dataloader.collate_fn = codraw_propv1_train_collate_fn
            # valid
            self.valid_dataset = CoDrawPropV1EvalDataset(
                dataset_path=cfg.valid_dataset_path,
                embed_dataset_path=cfg.valid_embed_dataset_path,
                # small batch size (due to SCAIN)
                batch_size=cfg.eval_batch_size,
                image_size=cfg.image_size,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                # small batch size (due to SCAIN)
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.valid_dataloader.collate_fn = codraw_propv1_eval_collate_fn
            # test
            self.test_dataset = CoDrawPropV1EvalDataset(
                dataset_path=cfg.test_dataset_path,
                embed_dataset_path=cfg.test_embed_dataset_path,
                # small batch size (due to SCAIN)
                batch_size=cfg.eval_batch_size,
                image_size=cfg.image_size,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                # small batch size (due to SCAIN)
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.test_dataloader.collate_fn = codraw_propv1_eval_collate_fn
        elif "iclevr" in cfg.dataset :
            # # train
            # self.dataset = ICLEVRPropV1TrainDataset(
            #     dataset_path=cfg.dataset_path,
            #     embed_dataset_path=cfg.embed_dataset_path,
            #     image_size=cfg.image_size,
            # )
            # self.dataloader = DataLoader(
            #     self.dataset,
            #     batch_size=cfg.batch_size,
            #     shuffle=True,
            #     num_workers=cfg.num_workers,
            #     pin_memory=True,
            #     drop_last=True,
            # )
            # self.dataloader.collate_fn = iclevr_propv1_train_collate_fn

            # # valid
            # self.valid_dataset = ICLEVRPropV1EvalDataset(
            #     dataset_path=cfg.valid_dataset_path,
            #     embed_dataset_path=cfg.valid_embed_dataset_path,
            #     image_size=cfg.image_size,
            # )
            # self.valid_dataloader = DataLoader(
            #     self.valid_dataset,
            #     # small batch size (due to SCAIN)
            #     batch_size=cfg.eval_batch_size,
            #     shuffle=False,
            #     num_workers=cfg.num_workers,
            #     drop_last=False,  # evaluate all data
            # )
            # self.valid_dataloader.collate_fn = iclevr_propv1_eval_collate_fn


            # test
            self.test_dataset_all = ICLEVRPropV1EvalDataset(
                dataset_path=cfg.test_dataset_path,
                embed_dataset_path=cfg.test_embed_dataset_path,
                image_size=cfg.image_size,
            )
            self.test_dataloader_all = DataLoader(
                self.test_dataset_all,
                # small batch size (due to SCAIN)
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.test_dataloader_all.collate_fn = iclevr_propv1_eval_collate_fn

            #########
            self.test_dataset_all_rsim = ICLEVRPropV1EvalDataset(
                dataset_path=cfg.test_dataset_path,
                embed_dataset_path=cfg.test_embed_dataset_path,
                image_size=cfg.image_size,
                category='general_all_rsim',
            )
            self.test_dataloader_all_rsim = DataLoader(
                self.test_dataset_all_rsim,
                # small batch size (due to SCAIN)
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.test_dataloader_all_rsim.collate_fn = iclevr_propv1_eval_collate_fn
            ################

            self.test_dataset_manipulation = ICLEVRPropV1EvalDataset(
                dataset_path=cfg.test_dataset_path,
                embed_dataset_path=cfg.test_embed_dataset_path,
                image_size=cfg.image_size,
                category='manipulation',
            )
            self.test_dataloader_manipulation = DataLoader(
                self.test_dataset_manipulation,
                # small batch size (due to SCAIN)
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.test_dataloader_manipulation.collate_fn = iclevr_propv1_eval_collate_fn

            self.test_dataset_ambre = ICLEVRPropV1EvalDataset(
                dataset_path=cfg.test_dataset_path,
                embed_dataset_path=cfg.test_embed_dataset_path,
                image_size=cfg.image_size,
                category='ambre',
            )
            self.test_dataloader_ambre = DataLoader(
                self.test_dataset_ambre,
                # small batch size (due to SCAIN)
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.test_dataloader_ambre.collate_fn = iclevr_propv1_eval_collate_fn

            self.test_dataset_infer = ICLEVRPropV1EvalDataset(
                dataset_path=cfg.test_dataset_path,
                embed_dataset_path=cfg.test_embed_dataset_path,
                image_size=cfg.image_size,
                category='infer',
            )
            self.test_dataloader_infer = DataLoader(
                self.test_dataset_infer,
                # small batch size (due to SCAIN)
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.test_dataloader_infer.collate_fn = iclevr_propv1_eval_collate_fn


            self.test_dataset_ambrr = ICLEVRPropV1EvalDataset(
                dataset_path=cfg.test_dataset_path,
                embed_dataset_path=cfg.test_embed_dataset_path,
                image_size=cfg.image_size,
                category='ambrr',
            )
            self.test_dataloader_ambrr = DataLoader(
                self.test_dataset_ambrr,
                # small batch size (due to SCAIN)
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.test_dataloader_ambrr.collate_fn = iclevr_propv1_eval_collate_fn

            self.test_dataset_detail = ICLEVRPropV1EvalDataset(
                dataset_path=cfg.test_dataset_path,
                embed_dataset_path=cfg.test_embed_dataset_path,
                image_size=cfg.image_size,
                category='detail',
            )
            self.test_dataloader_detail = DataLoader(
                self.test_dataset_detail,
                # small batch size (due to SCAIN)
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.test_dataloader_detail.collate_fn = iclevr_propv1_eval_collate_fn

            self.test_dataset_numb = ICLEVRPropV1EvalDataset(
                dataset_path=cfg.test_dataset_path,
                embed_dataset_path=cfg.test_embed_dataset_path,
                image_size=cfg.image_size,
                category='numb',
            )
            self.test_dataloader_numb = DataLoader(
                self.test_dataset_numb,
                # small batch size (due to SCAIN)
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,  # evaluate all data
            )
            self.test_dataloader_numb.collate_fn = iclevr_propv1_eval_collate_fn

        else:
            raise ValueError


    def evaluation(self, split="valid", cate='general_all'):
        if split == "valid":
            eval_dataloader = self.valid_dataloader
            eval_dataset_path = self.cfg.valid_dataset_path
        elif split == "test":
            if cate == 'general_all':
                eval_dataloader = self.test_dataloader_all
            elif cate == 'general_all_rsim':
                eval_dataloader = self.test_dataloader_all_rsim
            elif cate == 'manipulation':
                eval_dataloader = self.test_dataloader_manipulation
            elif cate == 'numb':
                eval_dataloader = self.test_dataloader_numb
            elif cate == 'detail':
                eval_dataloader = self.test_dataloader_detail
            elif cate == 'infer':
                eval_dataloader = self.test_dataloader_infer
            elif cate == 'ambre':
                eval_dataloader = self.test_dataloader_ambre
            elif cate == 'ambrr':
                eval_dataloader = self.test_dataloader_ambrr
            else:
                raise ValueError
            eval_dataset_path = self.cfg.test_dataset_path

        else:
            raise ValueError

        # create directory for latest (not best!) images
        dirname = os.path.join(SAVE_DIR, f"images_{split}/", cate)
        os.makedirs(dirname, exist_ok=True)

        # generate images
        logger.debug(f"generate {split} outputs...")
        for batch in tqdm(eval_dataloader):
            self.model.predict_batch(
                batch,
                dirname,
                num_parallel=self.cfg.num_parallel_search,
            )

        # submission
        ap, ar, f1, _, rsim, cmat, = \
            calculate_inception_objects_accuracy(
                dirname,
                self.cfg.detector_localizer_path,
                eval_dataset_path,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                category=cate,
            )

        if cate == 'general_all' and split == 'test':
            outputs = {
                "scalar": {
                    f"{split}_AP": ap,
                    f"{split}_AR": ar,
                    f"{split}_F1": f1,
                }
            }
        elif cate == 'general_all_rsim' and split == 'test':
            outputs = {
                "scalar": {
                    f"{split}_RSIM": rsim,
                }
            }
        else:
            outputs = {
                "scalar": {
                    f"{split}_AP": ap,
                    f"{split}_AR": ar,
                    f"{split}_F1": f1,
                    f"{split}_RSIM": rsim,
                },
                "others": {
                    f"{split}_CMAT": cmat,
                }
            }
        return outputs

    def eval(self):
        # logger.debug("evaluate by valid data...")
        # outputs = self.evaluation(split="valid")

        logger.debug("evaluate by test data...")
        outputs_all_rsim = self.evaluation(split="test", cate='general_all_rsim')
        outputs_all = self.evaluation(split="test", cate='general_all')
        outputs_numb = self.evaluation(split="test", cate='numb')
        outputs_manipulation = self.evaluation(split="test", cate='manipulation')
        outputs_ambre = self.evaluation(split="test", cate='ambre')
        outputs_infer = self.evaluation(split="test", cate='infer')
        outputs_ambrr = self.evaluation(split="test", cate='ambrr')
        outputs_detail = self.evaluation(split="test", cate='detail')

        logger.debug(
            (
                f"epoch: {0}, step: {0}, "
                f"data_all: {outputs_all['scalar']}, "
                f"data_all_rsim: {outputs_all_rsim['scalar']}, "
                f"data_manipulation: {outputs_manipulation['scalar']}, "
                f"data_ambre: {outputs_ambre['scalar']}, "
                f"data_infer: {outputs_infer['scalar']}, "
                f"data_ambrr: {outputs_ambrr['scalar']}, "
                f"data_numb: {outputs_numb['scalar']}, "
                f"data_detail: {outputs_detail['scalar']}, "
            )
        )


        # test_outputs = self.evaluation(split="test")
        # outputs["scalar"].update(test_outputs["scalar"])
        # outputs["others"].update(test_outputs["others"])

        # logger.debug(
        #     (
        #         f"epoch: {0}, step: {0}, "
        #         f"data: {outputs['scalar']}"
        #     )
        # )





def eval_propv1_scain_geneva(cfg):
    logger.info(f"script {__name__} start!")

    # SAVE_DIR SETTING
    global SAVE_DIR
    SAVE_DIR = os.path.join(SAVE_ROOT_DIR, cfg.name)
    if os.path.exists(SAVE_DIR):
        logger.warning(f"{SAVE_DIR} already exists. Overwrite by current run?")

        # if "stdin" in cfg:
        #     stdin = cfg.stdin
        # else:
        #     stdin = input("Press [Y/n]: ")

        shutil.rmtree(SAVE_DIR)
        # # if Yes --> delete old files
        # if stdin == "Y":
        #     shutil.rmtree(SAVE_DIR)
        # # if No --> create temporary directory
        # elif stdin == "n":
        #     SAVE_DIR = os.path.join(f"/var/tmp/{uuid.uuid4()}", cfg.name)
        #     logger.warning(f"temporary save at {SAVE_DIR}.")
        # else:
        #     raise ValueError

    os.makedirs(SAVE_DIR)

    trainer = PropV1ScainTrainer(cfg)
    trainer.eval()
