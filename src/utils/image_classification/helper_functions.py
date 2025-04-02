from pathlib import Path
import requests
import cv2
import numpy as np
import openvino as ov
import base64
import subprocess
import os
import sys

def ensure_model_files(model_dir=Path(".")):
    """
    Checks if the OpenVINO model files (.xml and .bin) exist in the specified directory.
    If not, downloads them from the provided Hugging Face links.

    Parameters:
        model_dir (Path): Directory where the model files should be present.
                          Defaults to the current directory.

    Returns:
        tuple: Paths (as Path objects) to the XML and BIN files.
    """
    xml_path = model_dir / "resnet50_fp16.xml"
    bin_path = model_dir / "resnet50_fp16.bin"
    
    if xml_path.exists() and bin_path.exists():
        print("Model files found.")
    else:
        print("Model files not found. Downloading from Hugging Face...")
        xml_url = "https://huggingface.co/katuni4ka/resnet50_fp16/resolve/main/resnet50_fp16.xml"
        bin_url = "https://huggingface.co/katuni4ka/resnet50_fp16/resolve/main/resnet50_fp16.bin"
        
        # Download XML file
        r_xml = requests.get(xml_url)
        r_xml.raise_for_status()
        xml_path.write_bytes(r_xml.content)
        print("Downloaded:", xml_path)
        
        # Download BIN file
        r_bin = requests.get(bin_url)
        r_bin.raise_for_status()
        bin_path.write_bytes(r_bin.content)
        print("Downloaded:", bin_path)
    
    return xml_path, bin_path


def classify_image(image, processor):
    """
    Classifies an image using the ResNet-50 model with the OpenVINO runtime.

    Parameters:
        image imported by decode_image
        processor (str): Target device for inference (e.g., "CPU", "GPU", "NPU", etc.)

    Returns:
        str: Predicted class label from the ImageNet dataset.
    
    Notes:
        - Requires model files 'resnet50_fp16.xml' and 'resnet50_fp16.bin'
          and 'imagenet_classes.txt' (with 1000 ImageNet labels) in the working directory.
        - The image is preprocessed to 224x224, normalized, and converted from BGR to RGB.
    """
    model_directory = Path("src", "utils", "image_classification")
    xml_path, _ = ensure_model_files(model_dir=model_directory)
    # Load OpenVINO model
    core = ov.Core()
    model = core.read_model(str(xml_path))
    compiled_model = core.compile_model(model, processor)

    # Preprocess the image
    if image is None:
        raise ValueError("Unable to load image")
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)   # Add batch dimension

    # Run inference
    result = compiled_model([image])[compiled_model.output(0)]
    top_class = int(np.argmax(result))

    # Load ImageNet labels
    labels_path = Path("src", "utils", "image_classification", "imagenet_classes.txt")
    if not labels_path.exists():
        raise FileNotFoundError("Missing 'imagenet_classes.txt'. Please ensure it is in the working directory.")
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    

    predicted_label = labels[top_class]
    print(f"Predicted Class: {predicted_label}")
    return predicted_label



def decode_image(contents):
    """
    Decodes a base64-encoded image to a numpy array using OpenCV.

    Parameters:
        contents (str): Base64 encoded image string (from Dash Upload component).

    Returns:
        image (numpy.ndarray): Decoded image in BGR format.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    np_arr = np.frombuffer(decoded, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


def run_benchmark_app(device="CPU", hint="latency"):
    """
    Executes the OpenVINO benchmark_app to measure model performance.

    Parameters:
        model_path (str): Path to the OpenVINO IR model file (e.g., 'resnet50_fp16.xml').
        device (str): Target device for benchmarking (e.g., 'CPU', 'GPU', or 'NPU').
                      Defaults to 'CPU'.
        hint (str): Performance hint for the benchmark, such as 'latency' or 'throughput'.
                    Defaults to 'latency'.

    Returns:
        str: The standard output from the benchmark_app command if successful.
             Returns None if an error occurs.

    Notes:
        - Ensure that the 'benchmark_app' executable is in your system PATH.
        - This function uses subprocess to call the terminal command:
          benchmark_app -m <model_path> -d <device> -hint <hint>
    """
    benchmark_path = os.path.join(sys.prefix, "Scripts", "benchmark_app.exe")
    model_path = Path("src", "utils", "image_classification", "resnet50_fp16.xml")
    command = ["benchmark_app", "-m", model_path,"-d", device, "-hint", hint, "-t 5"]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error executing benchmark_app:")
        print(e)
        return None
    
if __name__ == '__main__':
    None