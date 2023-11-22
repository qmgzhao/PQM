from pydoc import ModuleScanner
from statistics import mode
import sys
import torch
import whisper
from bitsandbytes import functional as F


def print_keys(model):
    for name, module in list(model.named_modules()):
        if "lora_" in name:
            print(name)


def quantize_state_dict(model):
    state_dict_fp32 = model.state_dict()
    state_dict = state_dict_fp32.copy()
    
    # blocksize in [4096, 2048, 1024, 512, 256, 128, 64]
    blocksize = 64
    # quant_type in ['fp4', 'nf4']
    quant_type = 'nf4'

    for name, param in state_dict.items():
        if ("query" in name or "key" in name or "value" in name or "out" in name or "mlp.0" in name or "mlp.2" in name or "conv" in name or "token_embedding" in name) and "weight" in name:
            q_param, state = F.quantize_4bit(param, blocksize=blocksize, quant_type=quant_type)
            d_param = F.dequantize_4bit(q_param, state, quant_type=quant_type)
            state_dict[name] = d_param
            # state_dict[name] = q_param
    
    model.load_state_dict(state_dict)