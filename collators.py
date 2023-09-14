import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
from torch import nn

import numpy as np

from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

@dataclass
class SignLanguageWhisperCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors

        T, H, W, C = features[0]["input_features"].shape
        
        input_features = [{"input_features": feature["input_features"].flatten(1)} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", padding="max_length", max_length=1500, truncation=True)

        batch["input_features"] = torch.reshape(batch["input_features"], (-1, 1500, H, W, C))

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


@dataclass
class SignLanguageT5Collator:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = [feature["labels"] for feature in features]
        # attention_masks = [feature["attention_mask"] for feature in features]
        inputs_embeds = [feature["input_features"] for feature in features]
        
        # Padding
        max_len = max([emb.shape[0] for emb in inputs_embeds])
        max_len = min(256, max_len)
        padded_inputs_embeds = []
        for emb in inputs_embeds:
            #print(f"inputs_embeds shape is {emb.shape}")
            pad_len = max_len - emb.shape[0]  # calculate how much padding to add
            emb_pad = torch.nn.functional.pad(emb, (0, 0, 0, 0, 0, 0, 0, pad_len), value=0)  # pad_token_id for embeddings
            #print(f"inputs_embeds shape is {emb_pad.shape}")
            padded_inputs_embeds.append(emb_pad)

        # Prepare labels
        max_label_length = max(len(l) for l in labels)
        if self.pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        for feature in features:
            remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
            feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)

        labels = torch.tensor([f["labels"] for f in features])

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=labels)
            for feature, decoder_input_id in zip(features, decoder_input_ids):
                feature["decoder_input_ids"] = decoder_input_id

        batch = {}
        
        batch["input_features"] = torch.stack(padded_inputs_embeds, dim=0)
        batch["labels"] = torch.stack([torch.tensor(f["labels"]) for f in features], dim=0)
        batch["decoder_input_ids"] = torch.stack([torch.tensor(f["decoder_input_ids"]) for f in features], dim=0)

        return batch


@dataclass
class SignLanguageWhisperCollatorBicubic:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors

        T, H, W, C = features[0]["input_features"].shape
        
        input_features = []
        for feature in features:
            f = feature["input_features"].permute(3, 1, 2, 0)
            try:
                interp = torch.nn.functional.interpolate(f, scale_factor=(1.0, 4.0), mode="bicubic").permute(3, 1, 2, 0)
            except:
                interp = f.permute(3, 1, 2, 0)
            input_features.append({"input_features": interp.flatten(1)})
        
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", padding="max_length", max_length=1500, truncation=True)

        batch["input_features"] = torch.reshape(batch["input_features"], (-1, 1500, H, W, C))

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

