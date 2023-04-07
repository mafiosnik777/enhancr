import sys
import torch
import os

# workaround for relative imports with embedded python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from GMFSS_Fortuna_Union_arch import Model_inference

class GMFSS_Fortuna_union:
    def __init__(self, partial_fp16=False):
        self.cache = False
        self.amount_input_img = 2

        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('medium')

        self.model = Model_inference(partial_fp16=partial_fp16)
        self.model.eval()

    def execute(self, I0, I1, timestep):
        with torch.inference_mode():
            middle = self.model(I0, I1, timestep).cpu()
        return middle