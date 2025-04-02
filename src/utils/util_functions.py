import openvino as ov
import torch


def get_devices():
    devices = set(["iGPU" if device == "GPU" else device.upper() for device in ov.Core().get_available_devices()])
    
    if torch.cuda.is_available():
        devices.add("GPU")

    return devices