import onnx
import argparse

def check_model_precision_onnx(model_path):
    model = onnx.load(model_path)
    input_data_type = model.graph.input[0].type.tensor_type.elem_type
    data_type_str = onnx.TensorProto.DataType.Name(input_data_type)
    del model
    return data_type_str

if __name__ == '__main__':
    # manual check by calling script
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", metavar="--input", type=str, help="input model")
    args = parser.parse_args()

    print(check_model_precision_onnx(args.input))