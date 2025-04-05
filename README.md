# Project: Local AI on mobile workstations using CPU, GPU and NPU capabilities”
==============================
## Video:
Link to video of app [https://www.youtube.com/watch?v=_qHEZ1RhQ9c].
## Setup
- Install uv (https://docs.astral.sh/uv/getting-started/installation/)
    - For windows run in Admin Powershell: 
    `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Install Ollama (https://ollama.com/)
    - Then run `ollama pull <model>` to get required model

## Windows specific:
Giving terminal admin rights:
- Open VSCode as admin
- In terminal run `Set-ExecutionPolicy Unrestricted`

## GitLab/GitHub:
- Add ssh-key to account by creating key with `ssh-keygen` in terminal
- navigate to location of choice for the project folder and git clone (git clone https://github.com/username/project-name.git my-folder)

## UV setup
- In project run `uv sync` to create venv or use `uv run <filename>` to run script

## Usage specifics:
- Add cpu_gpu_usage.csv to data/test
- Download nltk stopwords by tunnin `nltk.download("stopwords")`


## Further steps (VSCode)
- Open the workspace settings.json (ctr + p -> "settings.json" -> select workspace settings)
- Add for windows or Linux: 
```
{
    "python.autoComplete.extraPaths": [
    "${workspaceFolder}"
  ],
    "python.envFile": "${workspaceFolder}/.env",
    "terminal.integrated.env.windows": {"PYTHONPATH": "${workspaceFolder}/"},
    "terminal.integrated.env.linux": {"PYTHONPATH": "${workspaceFolder}/"},
    "terminal.integrated.env.osx": {"PYTHONPATH": "${workspaceFolder}/"},
}
```

- Create .env file in root folder
- Add in .env file `PYTHONPATH=.`

Install following extension (OPTIONAL):
- live server and/or live preview
- html CSS support
- Ruff
- IntelliCode & IntelliCode Completions for suggestions and autocompletion

## Running and using the app

Open app.py page in VSCode. Press play button on the top right. After running through packages, an IP address will appear. Ctrl click the IP address to open the app in a browser window. 

Key Features of the app:
Local Large Language Model (LLM) with RAG:
Runs entirely on-device to preserve privacy, reduce latency, and support offline use. Here it is important to choose the correct model. 
RAM/VRAM and Disk Space Considerations: Tiny models run on consumer GPUs, bigger models start asking for powerful GPUs or, you know, datacenters.

RAG means retrieval augmented generation. The LLM "retrieves" information from PDFs to ensure accuracy. It has been given PDF documents about HP products, so feel free to ask questions about them! You can ask the chatbot powered by Mistral general questions about NPUs and GPUs as well.
Example prompts:
What is an NPU?
What are the specifications for Zbook Power?

Image Classification:
Benchmarks Intel's integrated CPU, GPU, and NPU. 

1. The picture of an Irish Setter can be uploaded from src, utils, image_classification, dog.png. This handsome boy will be classified as an Irish settler when the classify image button is pressed. 

ResNet-50 is a convolutional neural network (CNN) that excels at image classification. We used a Resnet-50 model from huggingface contributed to by Ekaterina Aidova (katuni4ka). The model is in fp16 format converted to OpenVino format (.xml and .bin). 

2. When you press the benchmark butten, it will run the inference around 200 times in approximately 5 s to return average latency and throughput. 

The benchmarks are done using OpenVino's benchmark_app (https://docs.openvino.ai/nightly/get-started/learn-openvino/openvino-samples/benchmark-tool.html) which can benchmark Intel's integrated CPU, GPU, and NPU.

Object Detection:
Evaluates more complex workloads. 
1. From src, utils, object_detection, data upload coco_bike.png and click detect.

YOLO (you only look once!) from Ultralytics is used for real-time object detection and image segmentation. We are using yolo12, version 12 of yolo, which is built on deep learning and computer vision and adapatable to different hardware platforms. 

The benchmarking for Intel's integrated CPU, GPU and NPU is again using OpenVino's benchmark_app. Nvidia's GPU is benchmarked using the Ultralytics package (https://docs.ultralytics.com/modes/benchmark/).

Resource Monitoring:
Displays real-time system usage, visualizing how workloads are distributed

1. Device utilisation of GPU & CPU shows usage in the past 3 minutes.
2. Memory utilisation of GPU & CPU shows usage in the past 3 minutes.

3. Throughput benchmark results in fps. Throughput measures how many tasks can be done per second. It clearly trends towards CPU having the lowest throughput, then the NPU, then the integrated GPU and the dedicated GPU.

4. Inference time comparison by processor plots the inference time for the object detection task on CPU, GPU, and NPU. The amount of time the task takes decrease from CPU to NPU to iGPU to GPU.

About us:

Our LinkedIn and Github profiles are shared on this page! This project was contributed to by Jan Roesnick, Kristina Liang, Marisa Davis, and Natalia Neamtu


