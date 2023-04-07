import onnx

def check_model_precision(model_path):
    model = onnx.load(model_path)
    input_data_type = model.graph.input[0].type.tensor_type.elem_type
    data_type_str = onnx.TensorProto.DataType.Name(input_data_type)
    del model
    return data_type_str