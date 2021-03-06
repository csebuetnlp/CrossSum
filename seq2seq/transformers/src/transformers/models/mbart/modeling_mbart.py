# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..bart.modeling_bart import BartForConditionalGeneration, BartModel
from .configuration_mbart import MBartConfig


_CONFIG_FOR_DOC = "MBartConfig"
_TOKENIZER_FOR_DOC = "MBartTokenizer"

MBART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mbart-large-cc25",
    "facebook/mbart-large-en-ro",
    # See all multilingual BART models at https://huggingface.co/models?filter=mbart
]


class MBartModel(BartModel):
    r"""
    This class overrides :class:`~transformers.BartModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = MBartConfig
    _keys_to_ignore_on_load_missing = [
        "encoder.embed_positions.weight",
        "decoder.embed_positions.weight",
    ]
    _keys_to_ignore_on_save = [
        "encoder.embed_positions.weight",
        "decoder.embed_positions.weight",
    ]


class MBartForConditionalGeneration(BartForConditionalGeneration):
    r"""
    This class overrides :class:`~transformers.BartForConditionalGeneration`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::
        >>> from transformers import MBartForConditionalGeneration, MBartTokenizer
        >>> model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
        >>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro")
        >>> article = "UN Chief Says There Is No Military Solution in Syria"
        >>> batch = tokenizer.prepare_seq2seq_batch(src_texts=[article], return_tensors="pt")
        >>> translated_tokens = model.generate(**batch)
        >>> translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        >>> assert translation == "??eful ONU declar?? c?? nu exist?? o solu??ie militar?? ??n Siria"
    """
    model_type = "mbart"
    config_class = MBartConfig
    _keys_to_ignore_on_load_missing = [
        "model.encoder.embed_positions.weight",
        "model.decoder.embed_positions.weight",
    ]
    _keys_to_ignore_on_save = [
        "model.encoder.embed_positions.weight",
        "model.decoder.embed_positions.weight",
    ]
