import os
import sys
from pathlib import Path
import subprocess

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
    command = [benchmark_path, "-m", model_path,"-d", device, "-hint", hint, "-t 5"]
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
    
results = run_benchmark_app(device="GPU", hint="throughput")
print(results)