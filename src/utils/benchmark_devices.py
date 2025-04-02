import logging
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
import torch
import openvino as ov
import subprocess

import plotly.express as px

from src.image_segmentation.yolo import _create_ov_model, parse_output

MODEL_NAME = "yolo12x"

logger = logging.getLogger(__name__)
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO)

devices = set(["iGPU" if device == "GPU" else device.upper() for device in ov.Core().get_available_devices()])
if torch.cuda.is_available():
    devices.add("GPU")


def benchmark_gpu():

    export_format = 'torchscript'
    benchmark_results = {}
    model = YOLO(Path("models", MODEL_NAME, 'gpu', f"{MODEL_NAME}.pt"))

    res = model.benchmark(data="data/coco128.yaml", imgsz=640, half=False, 
                            device='cuda', project=Path("data", "benchmarks", device), 
                            name="YOLO", format = export_format)

    benchmark_results["inference_time"] = res['Inference time (ms/im)'].values[0]
    benchmark_results["fps"] = res['FPS'].values[0]
    logging.info(f"Benchmark result for GPU:\n{benchmark_results}")
    return benchmark_results


def benchmark_ov_devices(device):
    device = device.lower().strip()
    benchmark_results = {}
    img_path = Path("datasets", "coco128", 'images', 'train2017')

    # Temporary fix to solve GPU/iGPU issue for benchmarking
    if device == 'gpu':
        device = 'igpu'
            
    model_path = Path("models", MODEL_NAME, device, f"{MODEL_NAME}_openvino_model", f"{MODEL_NAME}.xml")
    if not model_path.exists():
        logging.info("Model not found, creating new model")
        Path("models", MODEL_NAME, device).mkdir(exist_ok=True, parents=True)
        core = ov.Core()
        model = _create_ov_model(model_path, ov_core=core, device=device, model_name=MODEL_NAME)
        # model = compile_ov_model(core, ov_model=model_ov, device=device, model_path=model_path)

    if device == "igpu":
        device = "gpu"

    res = subprocess.run(["benchmark_app", "-m", model_path, '-d', device.upper(), "-i", img_path], 
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)
    
    # parse output
    logging.info(res.stdout)
    parsed_out = parse_output(res.stdout)
    benchmark_results["inference_time"] = parsed_out['latency']
    benchmark_results["fps"] = parsed_out['fps']
    logging.info(f"Benchmarking result for {device}:\n{benchmark_results}")
    return benchmark_results

# benchmark_model()
if __name__ == "__main__":

    devices = set(["iGPU" if device == "GPU" else device.upper() for device in ov.Core().get_available_devices()])
    if torch.cuda.is_available():
        devices.add("GPU")

    benchmarking_results = {}
    for device in devices:
        
        if device == 'GPU':
            logging.info("Starting benchmarking on GPU...")
            res = benchmark_gpu()
        else:
            logging.info(f"Starting benchmarking on {device} with OpenVino...")
            res = benchmark_ov_devices(device)
        benchmarking_results[device] = res

    df_results = pd.DataFrame(benchmarking_results)
    df_results.to_csv(Path('reports', 'benchmarks', f'detection_benchmarks_{MODEL_NAME}.csv'))

    df_results = pd.read_csv(Path('reports', 'benchmarks', f'detection_benchmarks_{MODEL_NAME}.csv'), index_col=0)
    df_results = df_results.T.reset_index().melt(id_vars=['index'])
    # df_results = df_results.set_index(df_results.columns[0])
    fig = px.bar(df_results[df_results['variable'] == 'inference_time'].sort_values(by='value'), x='index', y='value', color='index', title=MODEL_NAME)
    fig.write_html(Path('reports', 'figures', f'detection_benchmarks_{MODEL_NAME}.html'))
    fig.show()
