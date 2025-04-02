from PIL import Image
from ultralytics import YOLO
import numpy as np
import base64
import cv2
import openvino as ov
from pathlib import Path

DET_MODEL_NAME = "yolo12n"


def decode_image(contents):
    """
    Decodes a base64-encoded image to a numpy array using OpenCV.
    """
    if not contents:
        return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    np_arr = np.frombuffer(decoded, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


def download_model():
    det_model = YOLO(f"{DET_MODEL_NAME}.pt")
    det_model_path = Path("src", "utils", "object_detection", f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml")
    if not det_model_path.exists():
        det_model.export(format="openvino", dynamic=True, half=True)



def object_detection(image, processor):
    core = ov.Core()
    det_model_path = Path(f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml")
    det_ov_model = core.read_model(det_model_path)

    ov_config = {}
    if processor != "CPU":
        det_ov_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in processor or ("AUTO" in processor and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    det_compiled_model = core.compile_model(det_ov_model, processor, ov_config)
    det_model = YOLO(det_model_path.parent, task="detect")

    if det_model.predictor is None:
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
        args = {**det_model.overrides, **custom}
        det_model.predictor = det_model._smart_load("predictor")(overrides=args, _callbacks=det_model.callbacks)
        det_model.predictor.setup_model(model=det_model.model)
        det_model.predictor.model.dynamic = False

    det_model.predictor.model.ov_compiled_model = det_compiled_model

    res = det_model(image)
    return Image.fromarray(res[0].plot()[:, :, ::-1])
