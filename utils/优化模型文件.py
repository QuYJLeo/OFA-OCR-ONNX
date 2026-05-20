import onnx
from onnx import helper, TensorProto
from onnxconverter_common import float16
import numpy as np
from onnxsim import simplify

def print_model_io_types(model_path):
    # 加载 ONNX 模型
    model = onnx.load(model_path)

    # 打印输入类型和形状
    print("Inputs:")
    for input in model.graph.input:
        tensor_type = input.type.tensor_type
        dtype = tensor_type.elem_type
        dtype_str = helper.tensor_dtype_to_string(dtype)
        shape = [dim.dim_value for dim in tensor_type.shape.dim]
        print(f"  Name: {input.name}, Type: {dtype_str}, Shape: {shape}")

    # 打印输出类型和形状
    print("Outputs:")
    for output in model.graph.output:
        tensor_type = output.type.tensor_type
        dtype = tensor_type.elem_type
        dtype_str = helper.tensor_dtype_to_string(dtype)
        shape = [dim.dim_value for dim in tensor_type.shape.dim]
        print(f"  Name: {output.name}, Type: {dtype_str}, Shape: {shape}")

    # 检查输入是否为 float32，并转换为 float16
    for input in model.graph.input:
        if input.type.tensor_type.elem_type == TensorProto.FLOAT:  # float32
            print(f"Converting input '{input.name}' from float32 to float16")
            input.type.tensor_type.elem_type = TensorProto.FLOAT16  # 转换为 float16

    for output in model.graph.output:
        if output.type.tensor_type.elem_type == TensorProto.FLOAT:  # float32
            print(f"Converting output '{output.name}' from float32 to float16")
            output.type.tensor_type.elem_type = TensorProto.FLOAT16  # 转换为 float16

    # 保存修改后的模型
    output_path = model_path.replace(".onnx", "_fp16.onnx")
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, output_path)
    print(f"Model saved to {output_path}")


# # float32转float16示例使用
# model_path = r"F:\OCR\ofa_ocr_infer_onnx\onxw\encoder_qu.onnx"
# print_model_io_types(model_path)



# encoder    decoder
# # 动态量化模型
# from onnxruntime.quantization import quantize_dynamic, QuantType
# quantize_dynamic(
#     model_input=r"F:\OCR\ofa_ocr_infer_onnx\onxw\decoder.onnx",
#     model_output=r"F:\OCR\ofa_ocr_infer_onnx\onxw\decoder_qu.onnx",
#     weight_type=QuantType.QInt8,  # 使用INT8量化
#     per_channel=True,  # 逐通道量化可能提高兼容性
#     # op_types_to_quantize=["MatMul", "Gemm"]  # 排除 Conv，不量化
# )




# model = onnx.load(r"F:\OCR\ofa_ocr_infer_onnx\onxw\encoder_qu.onnx")
# model_simp, check = simplify(model)
# onnx.save(model_simp, r"F:\OCR\ofa_ocr_infer_onnx\onxw\encoder_qu_simplified.onnx")






