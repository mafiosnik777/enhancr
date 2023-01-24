import os
import sys
import torch
import argparse

# workaround for relative imports with embedded python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from arch.SRVGGNet import SRVGGNetCompact as RealESRGAN
from arch.RRDBNet import RRDBNet as ESRGAN

parse = argparse.ArgumentParser(description="")
parse.add_argument("--input", metavar="--input", type=str, help="input model")
parse.add_argument("--output", metavar="--output", type=str, help="output model")
args = parse.parse_args()


def load_state_dict(state_dict):
    # Modified from https://github.com/chaiNNer-org/chaiNNer/blob/f05c3b51e5bc22372eacd8322ce291bf38a6f1c8/backend/src/nodes/impl/pytorch/model_loading.py
    state_dict_keys = list(state_dict.keys())

    if "params_ema" in state_dict_keys:
        state_dict = state_dict["params_ema"]
    elif "params-ema" in state_dict_keys:
        state_dict = state_dict["params-ema"]
    elif "params" in state_dict_keys:
        state_dict = state_dict["params"]

    state_dict_keys = list(state_dict.keys())
    
    if "body.0.weight" in state_dict_keys and "body.1.weight" in state_dict_keys:
        model = RealESRGAN(state_dict)
    else:
        try:
            model = ESRGAN(state_dict)
        except:
            print("[Error] Couldn't convert model", file=sys.stderr)
    return model

state_dict = torch.load(args.input, map_location=torch.device('cpu'))

model = load_state_dict(state_dict)

input_names = ["input"]
output_names = ["output"]

f1 = torch.rand((1, 3, 64, 64))
x = f1

torch.onnx.export(
    model,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    args.output,  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=16,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=input_names,  # the model's input names
    output_names=output_names,
    dynamic_axes={'input' : {3 : 'width', 2: 'height'}} )#
del model

print("[Conversion] Successfully converted model to onnx: " + "'" + args.input + "'", file=sys.stderr)
