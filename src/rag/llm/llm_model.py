import logging
from pathlib import Path

import onnxruntime as rt
import openvino as ov
import sys
import torch
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.intel import OVModelForCausalLM

from langchain import hub
from langchain_ollama import ChatOllama
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.language_models.chat_models import BaseChatModel

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, PreTrainedModel
from transformers.pipelines import Pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)

# MODEL_PATH = Path("..", "..", "models") # --> PyCharm
MODEL_PATH = Path("models")


def param_to_string(parameters) -> str:
    """Convert a list / tuple of parameters returned from OV to a string."""
    if isinstance(parameters, (list, tuple)):
        return ', '.join([str(x) for x in parameters])
    else:
        return str(parameters)

def _check_device() -> None:

    # --------------------------- Step 1. Initialize OpenVINO Runtime Core --------------------------------------------
    core = ov.Core()

    # --------------------------- Step 2. Get metrics of available devices --------------------------------------------
    logger.info('Available devices:')
    for device in core.available_devices:
        logger.info(f'{device} :')
        logger.info('\tSUPPORTED_PROPERTIES:')
        for property_key in core.get_property(device, 'SUPPORTED_PROPERTIES'):
            if property_key not in ('SUPPORTED_PROPERTIES'):
                try:
                    property_val = core.get_property(device, property_key)
                except TypeError:
                    property_val = 'UNSUPPORTED TYPE'
                logger.info(f'\t\t{property_key}: {param_to_string(property_val)}')
        logger.info('')

def _check_intel():
    core = ov.Core()
    if core.available_devices:
        return 1
    else:
        return 0

def _check_cuda():
    if torch.cuda.is_available():
        return 1
    else:
        return 0

def _check_npu():
    if "NPU" in ov.Core().available_devices:
        return 1
    else:
        return 0

def get_devices():
    logger.debug("Checking available devices")
    available_devices = []
    if _check_intel():
        logger.debug("Checking if intel CPU is available")
        available_devices.append("cpu")
    if _check_cuda():
        logger.debug("Checking if GPU is available")
        available_devices.append("gpu")
    if _check_npu():
        logger.debug("Checking if NPU is available")
        available_devices.append("npu")
    logger.info("Available devices: %s", available_devices)
    return available_devices


def save_model(model_checkpoint: str, save_dir: str | Path,  device: str, backend=None):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Saving model for %s', device)
    match device:
        case "cpu":
            logging.info("Saving %s model for inference on CPU", model_checkpoint.split("/")[-1])
            save_dir = save_dir / "cpu"
            model = _get_cpu_model(model_checkpoint, backend=backend)
            model.save_pretrained(save_dir)
            logging.info("Model saved to %s", save_dir)
        case "gpu":
            logging.info("Saving %s model for inference on GPU", model_checkpoint.split("/")[-1])
            pass
        case "npu":
            logging.info("Saving %s model for inference on NPU", model_checkpoint.split("/")[-1])
            pass


def _get_cpu_model(model_checkpoint: str | Path, backend=None):
    if backend == "openvino":
        export_flag = True
        if isinstance(model_checkpoint, Path):
            export_flag=False
        model = OVModelForCausalLM.from_pretrained(model_checkpoint, export=export_flag)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    return model

# Create GPU inference model

# Create NPU inference model

# Load model
def load_model(model_checkpoint: str | Path, device, backend=None):
    logging.info("Loading model from %s for inference on %s", model_checkpoint, device)

    model_checkpoint = Path(model_checkpoint)
    
    match device:
        case "cpu":
            model = _get_cpu_model(model_checkpoint / "cpu", backend=backend)
        case "gpu":
            model = None
        case "npu":
            model = None
        case _:
            raise NotImplementedError(f"Model for {device} not implemented.")

    return model

def get_model(model_checkpoint: str, device: str = "cpu", backend=None) -> BaseChatModel:

    logging.info("Getting %s model", model_checkpoint)
    if device not in get_devices():
        raise RuntimeError(f"Device '{device}' is not available. Found {get_devices()} devices")

    model_dir = Path(MODEL_PATH, model_checkpoint)
    if backend:
        model_dir = model_dir / backend
    if not Path(model_dir, device).exists():
        save_model(model_checkpoint, model_dir, device=device, backend=backend)
    model = load_model(model_dir, device, backend=backend)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    pipe = make_pipe(model, model_checkpoint, tokenizer=tokenizer, device=device)
    pipe_hf = HuggingFacePipeline(pipeline=pipe)
    llm = ChatHuggingFace(llm=pipe_hf, tokenizer=tokenizer)

    return llm


def make_pipe(model, model_checkpoint: str, tokenizer=None, device: str = "str") -> Pipeline:
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return pipe


def model_inference(model, tokenizer, msg: str):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")
    result = pipe(msg)
    return result
    # llm = ChatOllama(model="llama3.2", temperature=0)

# def get_model_name():
#     return "llama3.2"

# def prompt_llm(prompt):
#     return llm.invoke(prompt).content


if __name__ == "__main__":

    devices = get_devices()

    # Get huggingface model
    model_checkpoint = "meta-llama/Llama-3.2-3B"
    cpu_model = get_model(model_checkpoint, devices[0])

    # save_model(devices, model_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    ans = model_inference(cpu_model, tokenizer, "He never went out without a book under his arm")
    print(ans)
    pipe = pipeline("text-generation", model=cpu_model, tokenizer=tokenizer, device="cpu")
    result = pipe("He never went out without a book under his arm")
    print(result)

