from enum import Enum


class ModelPrecision(Enum):
    INT8 = 0
    FP16 = 1
    FP32 = 2

    @property
    def open_vino_model_suffix(self) -> str:
        if self == ModelPrecision.INT8:
            return "fp16-int8"
        elif self == ModelPrecision.FP16:
            return "fp16"
        elif self == ModelPrecision.FP32:
            return "fp32"

        raise Exception(f"Model precision {self} is not supported by openvino.")
