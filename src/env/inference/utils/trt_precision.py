import tensorrt as trt
from polygraphy.backend.trt import TrtRunner

def check_model_precision_trt(model_path):
    with open(model_path, 'rb') as f:
        engine_data = f.read()
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    runner = TrtRunner(engine)
    with runner:
        input_metadata = runner.get_input_metadata()
        input_precision = input_metadata["input"].dtype
        return input_precision