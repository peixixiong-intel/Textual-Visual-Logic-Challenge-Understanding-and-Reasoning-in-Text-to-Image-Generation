import os
from collections import defaultdict

import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast

from data.codraw_retrieval_dataset import CoDrawRetrievalDataset, codraw_retrieval_collate_fn
from data.iclevr_retrieval_dataset import ICLEVRRetrievalDataset, iclevr_retrieval_collate_fn
from modules.retrieval.sentence_encoder import SentenceEncoder, BERTSentenceEncoder

# ToDo
import numpy as np

from logging import getLogger

logger = getLogger(__name__)

def find_idx(ipt_list, ipt_item):
    indices = [i for i, x in enumerate(ipt_list) if x == ipt_item]
    return indices


class SentenceEmbeddingGenerator:
    def __init__(self, cfg):
        self.cfg = cfg

        # model
        self.sentence_encoder_type = cfg.sentence_encoder_type
        if "model_path" not in cfg:
            logger.warning(
                "model_path is not specified. "
                "use initial weight of pretrained models."
            )
            state_dict = None
        else:
            state_dict = torch.load(cfg.model_path)

        #ToDo
        if "negative_samples" not in self.cfg:
            self.cfg.negative_samples = False
            self.cfg.negative_N = 0
            self.cfg.substitute_rate = 0
        if "negative_cp" not in self.cfg:
            self.cfg.negative_cp = False
        if "rel_enhance" not in self.cfg:
            self.cfg.rel_enhance = False


        if cfg.sentence_encoder_type == "rnn":
            self.rnn_prev = nn.DataParallel(
                SentenceEncoder(cfg.text_dim),
                device_ids=[0],
            ).cuda()
            self.rnn_curr = nn.DataParallel(
                SentenceEncoder(cfg.text_dim),
                device_ids=[0],
            ).cuda()
            if state_dict is not None:
                self.rnn_prev.load_state_dict(state_dict["rnn_prev"])
                self.rnn_curr.load_state_dict(state_dict["rnn_curr"])
            self.rnn_prev.eval()
            self.rnn_curr.eval()
        elif cfg.sentence_encoder_type == "bert":
            self.tokenizer = BertTokenizerFast.\
                from_pretrained("bert-base-uncased")
            self.bert = nn.DataParallel(
                BERTSentenceEncoder(),
                device_ids=[0],
            ).cuda()
            if state_dict is not None:
                self.bert.load_state_dict(state_dict["bert"])
            self.bert.eval()
        else:
            raise ValueError

        # dataset
        if cfg.dataset == "codraw":
            # codraw-train
            self.dataset = CoDrawRetrievalDataset(
                cfg.dataset_path,
                cfg.glove_path,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.dataloader.collate_fn = codraw_retrieval_collate_fn
            # codraw-valid
            self.valid_dataset = CoDrawRetrievalDataset(
                cfg.valid_dataset_path,
                cfg.glove_path,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.valid_dataloader.collate_fn = codraw_retrieval_collate_fn
            # codraw-test
            self.test_dataset = CoDrawRetrievalDataset(
                cfg.test_dataset_path,
                cfg.glove_path,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.test_dataloader.collate_fn = codraw_retrieval_collate_fn
        elif "iclevr" in cfg.dataset :
            # iclevr-train
            self.dataset = ICLEVRRetrievalDataset(
                cfg.dataset_path,
                cfg.glove_path,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.dataloader.collate_fn = iclevr_retrieval_collate_fn
            # iclevr-valid
            self.valid_dataset = ICLEVRRetrievalDataset(
                cfg.valid_dataset_path,
                cfg.glove_path,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.valid_dataloader.collate_fn = iclevr_retrieval_collate_fn
            # iclevr-test
            self.test_dataset = ICLEVRRetrievalDataset(
                cfg.test_dataset_path,
                cfg.glove_path,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.test_dataloader.collate_fn = iclevr_retrieval_collate_fn
        else:
            raise ValueError

    def generate(self, save_path, split="train"):
        if self.cfg.negative_cp:
            substitute_dictionary = new_dir_dict
        else:
            substitute_dictionary = get_substitute_dictionary()
        # keys = [(access_id, turn_index), ...]
        if split == "train":
            keys = self.dataset.keys
            dataloader = self.dataloader
        elif split == "valid":
            keys = self.valid_dataset.keys
            dataloader = self.valid_dataloader
        elif split == "test":
            keys = self.test_dataset.keys
            dataloader = self.test_dataloader
        else:
            raise ValueError

        # all_text_features: List of Tensor (D,)
        # all_text_memories: List of Tensor (ml, D)
        # all_text_lengths: List of int [text_length, ...]
        # mb: mini-batch, ml: max dialog length in mini-batch
        all_text_features = []
        all_text_memories = []
        all_text_lengths = []
        all_rel_mask = []

        # ToDo
        all_text_neg_features = []
        all_text_neg_memories = []
        all_text_neg_lengths = []
        # all_utters = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                if self.sentence_encoder_type == "rnn":
                    # extract from batch
                    prev_embs = batch["prev_embs"]
                    embs = batch["embs"]
                    prev_seq_len = batch["prev_seq_len"]
                    seq_len = batch["seq_len"]

                    # forward sentence encoder
                    _, _, context = self.rnn_prev(
                        prev_embs, prev_seq_len)
                    text_memories, text_feature, _ = self.rnn_curr(
                        embs, seq_len, context)

                elif self.sentence_encoder_type == "bert":
                    # ToDo
                    prev_utter = batch["prev_utter"]
                    utter = batch["utter"]

                    inputs = self.tokenizer(
                        text=prev_utter,
                        text_pair=utter,
                        add_special_tokens=True,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )

                    text_memories, text_feature, key_padding_mask = self.bert(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        inputs["token_type_ids"],
                    )


                    if self.cfg.rel_enhance:
                        # text_memories torch.Size([50, 92, 768])
                        # inputs["input_ids"] torch.Size([50, 92])
                        rel_mask = torch.zeros(text_memories.size(0), text_memories.size(1))
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2392, 1)  # front
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2369, 1)  # behind
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2187, 1)  # left
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2157, 1)  # right
                        #ToDo
                        # Add more directions
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2039, 1)  # up
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2091, 1)  # down
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2067, 1)  # back
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 3953, 1)  # bottom
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2690, 1)  # middle
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2104, 1)  # under
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2327, 1)  # top
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2682, 1)  # above
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2369, 1)  # behind
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2917, 1)  # below
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2503, 1)  # inside
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2521, 1)  # far
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2379, 1)  # near
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2077, 1)  # before
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2503, 1)  # inside
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2648, 1)  # outside

                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2312, 1)  # large
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 2235, 1)  # small
                        rel_mask = rel_mask.masked_fill(inputs["input_ids"] == 5396, 1)  # medium

                        rel_mask = rel_mask.unsqueeze(2).repeat(1, 1, 1)


                    # ToDo random subsitute input
                    # Change utter
                    # Randomly mask
                    # Randomly substitute
                    operation_list = []

                    if self.cfg.negative_samples and self.sentence_encoder_type == "bert":
                        text_memories_negs, text_feature_negs, key_padding_mask_negs = [], [], []
                        for neg_n in range(self.cfg.negative_N):
                            text_memories_negs_, text_feature_negs_, key_padding_mask_negs_ = [], [], []
                            for u_id, utter_sentence in enumerate(utter):
                                utter_sentence_list = utter_sentence.split()
                                for utter_word_idx in range(len(utter_sentence_list)):
                                    operation = np.random.choice(['sub', 'keep'], size=(1,),
                                                                 p=[self.cfg.substitute_rate,
                                                                    1 - self.cfg.substitute_rate])[0]
                                    if utter_word_idx == len(utter_sentence_list) - 1 and 'sub' not in operation_list:
                                        # avoid always keep
                                        operation = 'sub'
                                    operation_list.append(operation)
                                    utter_word = utter_sentence_list[utter_word_idx]
                                    if utter_word in substitute_dictionary:
                                        if operation == 'sub':
                                            utter_word_new = \
                                                np.random.choice(substitute_dictionary[utter_word], size=(1,))[0]
                                        else:
                                            utter_word_new = utter_word
                                        utter_sentence_list[utter_word_idx] = utter_word_new
                                neg_utter = [' '.join(utter_sentence_list)]

                                inputs_neg = self.tokenizer(
                                    text=[prev_utter[u_id]],
                                    text_pair=neg_utter,
                                    add_special_tokens=True,
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt",
                                )

                                text_memories_neg, text_feature_neg, key_padding_mask_neg = self.bert(
                                    inputs_neg["input_ids"],
                                    inputs_neg["attention_mask"],
                                    inputs_neg["token_type_ids"],
                                )

                                text_memories_negs_.append(text_memories_neg)
                                text_feature_negs_.append(text_feature_neg)
                                key_padding_mask_negs_.append(key_padding_mask_neg)

                            text_memories_negs.append(np.array(text_memories_negs_))
                            text_feature_negs.append(np.array(text_feature_negs_))
                            key_padding_mask_negs.append(np.array(key_padding_mask_negs_))

                '''
                text_memories_negs [batch_size, negative_N, text_memories_len, dim]
                text_feature_negs [batch_size, negative_N, text_feature_len, dim]
                key_padding_mask_negs [batch_size, negative_N, key_padding_mask_len]
                '''

                # push to list
                for i in range(text_memories.size(0)):
                    if self.sentence_encoder_type == "rnn":
                        # Do not support neg samples
                        all_text_features.append(
                            text_feature[i].cpu().numpy())
                        all_text_memories.append(
                            text_memories[i].cpu().numpy())
                        all_text_lengths.append(
                            seq_len[i].cpu().numpy())
                    elif self.sentence_encoder_type == "bert":
                        all_text_features.append(
                            text_feature[i].cpu().numpy())

                        # output text memories of bert includes embedding of previous instructions.
                        # key_padding_mask is False where [CLS] & [[SENTENCE B], ...] & [SEP]
                        # sum of NOT key_padding_mask == text length
                        bool_indices = ~key_padding_mask[i]
                        _seq_len = np.array(bool_indices.sum().item())
                        _text_memories = text_memories[i][bool_indices]

                        all_text_memories.append(
                            _text_memories.cpu().numpy())
                        all_text_lengths.append(
                            _seq_len)
                        if self.cfg.rel_enhance:
                            _rel_mask = rel_mask[i][bool_indices]
                            all_rel_mask.append(_rel_mask.cpu().numpy())

                if self.cfg.negative_samples and self.sentence_encoder_type == "bert":
                    for i in range(text_memories.size(0)):
                        all_text_neg_memories_ = []
                        for neg_n in range(self.cfg.negative_N):
                            # [batch size x neg_N]
                            all_text_neg_features.append(
                                text_feature_negs[neg_n][i].cpu().numpy())

                            bool_indices_neg = ~key_padding_mask_negs[neg_n][i]
                            _seq_len_neg = np.array(bool_indices_neg.sum().item())
                            _text_memories_neg = text_memories_negs[neg_n][i][bool_indices_neg]

                            all_text_neg_memories_.append(
                                _text_memories_neg.cpu().numpy())  # [bn*neg, 91, 768]
                            all_text_neg_lengths.append(
                                _seq_len_neg)

                        all_text_neg_memories.append(all_text_neg_memories_)

        # mapping dataset_id(did) to tuple of (start_index, end_index)
        # keys = [(access_id, turn_index), ...]
        id2idxtup = defaultdict(list)
        for i, (did, tid) in enumerate(keys):
            # start end
            if did not in id2idxtup:
                id2idxtup[did] = [i, i]
            else:
                id2idxtup[did][1] = i

        # create h5 datasets
        h5 = h5py.File(save_path, "w")
        for did in id2idxtup.keys():
            start, end = id2idxtup[did]
            end += 1

            # turns_text_embedding: shape=(l, D)
            # turns_text_length: shape=(l,)
            # l: dialog length
            turns_text_embedding = np.stack(
                all_text_features[start:end], axis=0)
            turns_text_length = np.array(
                all_text_lengths[start:end])

            # turns_word_embeddings: shape=(l, ms, D)
            # ms: max text length of a dialog
            # it means that turns_word_embeddings is already padded.
            turns_word_embeddings = np.zeros(
                (len(turns_text_length), max(turns_text_length), self.cfg.text_dim))
            for i, j in enumerate(range(start, end)):
                text_length = turns_text_length[i]
                turns_word_embeddings[i, :text_length] = \
                    all_text_memories[j][:text_length]

            if self.cfg.rel_enhance:
                turns_rel_enhancement = np.zeros(
                    (len(turns_text_length), max(turns_text_length), 1))
                for i, j in enumerate(range(start, end)):
                    text_length = turns_text_length[i]
                    turns_rel_enhancement[i, :text_length] = \
                        all_rel_mask[j][:text_length]

            if self.cfg.negative_samples and self.sentence_encoder_type == "bert":
                # turns_text_neg_embedding: shape=(l, Neg, D)
                # turns_text_neg_length: shape=(l, Neg, )
                # l: dialog length
                turns_text_neg_embedding = np.stack(
                    all_text_neg_features[self.cfg.negative_N * start:self.cfg.negative_N * end],
                    axis=0).reshape((turns_text_embedding.shape[0],
                                     self.cfg.negative_N,
                                     turns_text_embedding.shape[1],))
                turns_text_neg_length = np.array(
                    all_text_neg_lengths[self.cfg.negative_N * start:self.cfg.negative_N * end]).reshape(
                    (turns_text_length.shape[0],
                     self.cfg.negative_N))
                # turns_text_neg_length = np.repeat(np.expand_dims(turns_text_length, axis=1), self.cfg.negative_N, axis=1)

                # turns_word_embeddings: shape=(l, Neg, ms, D)
                # ms: max text length of a dialog
                # it means that turns_word_embeddings is already padded.
                turns_word_neg_embeddings = np.zeros(
                    (len(turns_text_neg_length), self.cfg.negative_N, max(max(turns_text_neg_length)),
                     self.cfg.text_dim))
                for neg_n in range(self.cfg.negative_N):
                    for i, j in enumerate(range(start, end)):
                        text_length_neg = turns_text_neg_length[i][neg_n]  # same length for negative samples
                        if turns_word_neg_embeddings.shape[2] < text_length_neg:
                            print(neg_n, i, j, turns_word_neg_embeddings.shape, text_length_neg)
                        turns_word_neg_embeddings[i, neg_n, :text_length_neg] = \
                            all_text_neg_memories[j][neg_n][:text_length_neg]



            # print('turns_text_embedding', turns_text_embedding.shape)
            # print('turns_text_neg_embedding', turns_text_neg_embedding.shape)
            # print('turns_word_embeddings', turns_word_embeddings.shape)
            # print('turns_word_neg_embeddings', turns_word_neg_embeddings.shape)
            # print('turns_text_length', turns_text_length.shape)
            # print('turns_text_neg_length', turns_text_neg_length.shape)
            # print('turns_rel_enhancement', turns_rel_enhancement.shape)
            # print('all_utters', all_utters)
            # exit()

            scene = h5.create_group(did)
            scene.create_dataset(
                "turns_text_embedding", data=turns_text_embedding)
            scene.create_dataset(
                "turns_word_embeddings", data=turns_word_embeddings)
            scene.create_dataset(
                "turns_text_length", data=turns_text_length)
            if self.cfg.rel_enhance:
                scene.create_dataset(
                    "turns_rel_enhancement", data=turns_rel_enhancement)
            if self.cfg.negative_samples and self.sentence_encoder_type == "bert":
                scene.create_dataset(
                    "turns_text_neg_embedding", data=turns_text_neg_embedding)
                scene.create_dataset(
                    "turns_word_neg_embeddings", data=turns_word_neg_embeddings)
                scene.create_dataset(
                    "turns_text_neg_length", data=turns_text_neg_length)


# ToDo
shape_list = ['cube', 'sephere', 'cylinder']
color_list = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
direction_list = ['left', 'right', 'front', 'behind', 'center', ]
new_dir_dict = {"left": ["right"],
                "right": ["left"],
                "front": ["behind"],
                "behind": ["front"]}


def get_substitute_dictionary():
    substitute_dictionary = {}
    for sub_list in [shape_list, color_list, direction_list]:
        for sub_item in sub_list:
            substitute_dictionary[sub_item] = list(set(sub_list) - set([sub_item]))
    return substitute_dictionary


def create_embs_from_model(cfg):
    logger.info(f"script {__name__} start!")

    generator = SentenceEmbeddingGenerator(cfg)

    for split in ["train", "valid", "test"]:
        if cfg.rel_enhance:
            save_path = os.path.join(
                cfg.save_root_dir,
                f"{cfg.dataset}_{split}_embeddings_rel_{cfg.fork}.h5",
            )
        elif cfg.negative_cp:
            save_path = os.path.join(
                cfg.save_root_dir,
                f"{cfg.dataset}_{split}_embeddings_w_cp_neg_{cfg.negative_N}_{cfg.fork}.h5",
            )
        elif cfg.negative_samples and cfg.negative_N != 1:
            save_path = os.path.join(
                cfg.save_root_dir,
                f"{cfg.dataset}_{split}_embeddings_w_neg_{cfg.negative_N}_{cfg.fork}.h5",
            )
        elif cfg.negative_samples and cfg.negative_N == 1:
            save_path = os.path.join(
                cfg.save_root_dir,
                f"{cfg.dataset}_{split}_embeddings_w_neg_{cfg.fork}.h5",
            )
        else:
            save_path = os.path.join(
                cfg.save_root_dir,
                f"{cfg.dataset}_{split}_embeddings_{cfg.fork}.h5",
            )
        logger.info(f"create additional dataset: {save_path}")
        generator.generate(save_path, split)
