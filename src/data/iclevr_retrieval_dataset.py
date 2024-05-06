import h5py
from ast import literal_eval

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import re, regex

class ICLEVRRetrievalDataset(Dataset):
    def __init__(self, dataset_path, glove_path):
        super().__init__()
        self.dataset_path = dataset_path

        self.dataset = None
        self.background = None
        # keys = [(access_id, turn_index), ...]
        self.keys = []

        with h5py.File(dataset_path, "r") as f:
            background = f["background"][...]
            self.background = cv2.resize(background, (128, 128))

            # _keys = [access_id, ...]
            # NOTE: access_id != scene_id, access_id is an access key of h5 to each episodes.
            # NOTE: f.keys() = [{access_id}, ..., background, entities]
            _keys = [key for key in f.keys() if key.isdigit()]
            for key in _keys:
                # dialog_length = f[key]["objects"][...].shape[0]
                # assert dialog_length == 5, \
                #     f"iclevr dialog length must be 5 but {dialog_length} of {key}."
                dialog_length = 1 #f[key]["objects"][...].shape[0]
                assert dialog_length == 1, \
                    f"iclevr dialog length must be 5 but {dialog_length} of {key}."
                self.keys.extend([(key, t) for t in range(dialog_length)])

        # build glove vocabs
        self.glove = _parse_glove(glove_path)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, "r")

        access_id, turn_index = self.keys[idx]
        example = self.dataset[access_id]
        image = example["images"][...][turn_index]  # (H, W, C), BGR
        objects = example["objects"][...][turn_index]  # (24,)
        # both text and scene_id need decode preprocessing
        utters = self._decode_text_from_h5(example["text"][...])  # (L=5,)
        utter = utters[turn_index]
        scene_id = self._decode_scene_id_from_h5(example["scene_id"][...])

        if turn_index > 0:
            # (H, W, C), BGR
            prev_image = example["images"][...][turn_index - 1]
            prev_utters = utters[:turn_index]  # (turn_index-1,)
            prev_objects = example["objects"][...][turn_index - 1]  # (24,)
        else:
            prev_image = self.background  # (H, W, C), BGR
            prev_utters = ["ready"]  # padding sentence
            prev_objects = np.zeros_like(objects)  # (24,)

        # image preprocessing
        image = self._image_preprocessing(image)
        prev_image = self._image_preprocessing(prev_image)

        # text preprocessing
        prev_utter = " ".join([ut for ut in prev_utters])
        embs, seq_len = self._text_preprocessing(utter)
        prev_embs, prev_seq_len = self._text_preprocessing(prev_utter)

        # added objects binary flag
        added_objects = objects - prev_objects
        added_objects = np.clip(added_objects, 0, 1)

        sample = {
            # image
            "prev_image": prev_image,
            "image": image,
            # text
            "prev_embs": prev_embs,
            "embs": embs,
            "prev_seq_len": prev_seq_len,
            "seq_len": seq_len,
            "prev_utter": prev_utter,
            "utter": utter,
            # auxiliary
            "objects": objects,
            "added_objects": added_objects,
            # info
            "scene_id": scene_id,
        }
        return sample

    def _decode_text_from_h5(self, text):
        return literal_eval(literal_eval(str(text)).decode())
        # return literal_eval(str(text))

    def _decode_scene_id_from_h5(self, scene_id):
        return literal_eval(str(scene_id)).decode()
        # return str(scene_id)

    def _image_preprocessing(self, image):
        image = image[..., ::-1]
        image = image / 128. - 1
        image = np.transpose(image, (2, 0, 1))
        return image

    def separate_symbols(self, text):
        # Adjusted pattern to ensure spaces around commas, semicolons, parentheses, and colons
        # This pattern matches any punctuation mark and ensures separation
        pattern = r'(?<!\s)([\p{P}])|([\p{P}])(?!\s)'
        # Replace the matched pattern with the punctuation surrounded by spaces
        separated_text = regex.sub(pattern, r' \1\2 ', text)
        # Fix potential multiple spaces introduced by the substitution
        separated_text = regex.sub(r'\s{2,}', ' ', separated_text)
        return separated_text.strip()

    def replace_multiple_periods(self, text):
        # This pattern matches sequences of periods and spaces that include at least one period
        pattern = r'(\.\s*)+\.?'
        # Replace the matched pattern with a single period
        cleaned_text = re.sub(pattern, '.', text)
        return cleaned_text

    def _text_preprocessing(self, text):
        text = self.separate_symbols(self.replace_multiple_periods(text))
        tokens = text.split()
        seq_len = len(tokens)
        word_embeddings = np.zeros((seq_len, 300))
        for i, w in enumerate(tokens):
            word_embeddings[i] = self.glove[w]
        return word_embeddings, seq_len


def _parse_glove(glove_path):
    """Construct Dict[word, glove_embedding(300)] from .txt

    Parameters
    ----------
    glove_path : str
        path to glove (word, values) text file.
        e.g.)
        green 0.01 -6.44 1.23 ...
        add 1.11 2.30 -0.13 ...

    Returns
    -------
    glove: dict
        key = word token
        value = 300-dim glove word embedding
    """
    glove = {}
    with open(glove_path, "r") as f:
        for line in f:
            splitline = line.split()
            word = splitline[0]
            try:
                embedding = np.array([float(val) for val in splitline[1:]])
                glove[word] = embedding
            except:
                pass
    return glove


def iclevr_retrieval_collate_fn(batch):
    batch_size = len(batch)
    c, h, w = batch[0]["prev_image"].shape

    max_prev_len = max([b["prev_seq_len"] for b in batch])
    max_len = max([b["seq_len"] for b in batch])

    batch_prev_images = np.zeros(
        (batch_size, c, h, w), dtype=np.float32)
    batch_images = np.zeros(
        (batch_size, c, h, w), dtype=np.float32)
    batch_prev_embs = np.zeros(
        (batch_size, max_prev_len, 300), dtype=np.float32)
    batch_embs = np.zeros(
        (batch_size, max_len, 300), dtype=np.float32)
    batch_prev_seq_lens = np.zeros(
        (batch_size,), dtype=np.int64)
    batch_seq_lens = np.zeros(
        (batch_size,), dtype=np.int64)
    batch_objects = np.zeros(
        (batch_size, 24), dtype=np.float32)
    batch_added_objects = np.zeros(
        (batch_size, 24), dtype=np.float32)
    batch_prev_utters = []
    batch_utters = []

    for i, b in enumerate(batch):
        prev_image = b["prev_image"]
        image = b["image"]
        prev_embs = b["prev_embs"]
        prev_seq_len = b["prev_seq_len"]
        embs = b["embs"]
        seq_len = b["seq_len"]
        prev_utter = b["prev_utter"]
        utter = b["utter"]
        objects = b["objects"]
        added_objects = b["added_objects"]

        batch_prev_images[i] = prev_image
        batch_images[i] = image
        batch_prev_embs[i, :prev_seq_len] = prev_embs
        batch_embs[i, :seq_len] = embs
        batch_prev_seq_lens[i] = prev_seq_len
        batch_seq_lens[i] = seq_len
        batch_objects[i] = objects
        batch_added_objects[i] = added_objects
        batch_prev_utters.append(prev_utter)
        batch_utters.append(utter)

    sample = {
        # image
        "prev_image": torch.FloatTensor(batch_prev_images),
        "image": torch.FloatTensor(batch_images),
        # text
        "prev_embs": torch.FloatTensor(batch_prev_embs),
        "embs": torch.FloatTensor(batch_embs),
        "prev_seq_len": torch.LongTensor(batch_prev_seq_lens),
        "seq_len": torch.LongTensor(batch_seq_lens),
        # auxiliary
        "objects": torch.FloatTensor(batch_objects),
        "added_objects": torch.FloatTensor(batch_added_objects),
        # raw text
        "prev_utter": batch_prev_utters,
        "utter": batch_utters,
    }
    return sample
