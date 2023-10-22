# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps
from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "MODELS_MAP"
]

MODELS_MAP = {
    'clip-vit-large-patch14': {
        'model_type': 'clip',
        'model_name_or_path': '/data/nlp/pre_models/torch/clip/clip-vit-large-patch14',
        'config_name': '/data/nlp/pre_models/torch/clip/clip-vit-large-patch14/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/clip/clip-vit-large-patch14',
    },
    'chinese-clip-vit-large-patch14': {
        'model_type': 'chinese_clip',
        'model_name_or_path': '/data/nlp/pre_models/torch/clip/chinese-clip-vit-large-patch14',
        'config_name': '/data/nlp/pre_models/torch/clip/chinese-clip-vit-large-patch14/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/clip/chinese-clip-vit-large-patch14',
    },
    'chinese-clip-vit-huge-patch14': {
        'model_type': 'chinese_clip',
        'model_name_or_path': '/data/nlp/pre_models/torch/clip/chinese-clip-vit-huge-patch14',
        'config_name': '/data/nlp/pre_models/torch/clip/chinese-clip-vit-huge-patch14/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/clip/chinese-clip-vit-huge-patch14',
    },

}


# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING




