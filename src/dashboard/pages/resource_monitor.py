from pathlib import Path
import warnings

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import platform

import psutil
import GPUtil
import torch
import numpy as np
import dash_daq as daq

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import deque

##For real-time data from mac
from src.dashboard.components.title_section import create_title_section
from src.vis.benchmark_plots import create_figure, get_benchmarkings


import warnings
warnings.simplefilter("ignore", DeprecationWarning)

#-------------------------------------
# app = dash.get_app()

dash.register_page(__name__, title="Resource Monitor", name="Resource Monitor", path='/resource-monitor', order=4)

# app.config.suppress_callback_exceptions = True

#deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])

# TODO: Mask initial zeros
# Dynamic showcase of GPU and CPU usage and memory
# List container with maximum length. First in in First out
cpu_container = deque([0] * 181, maxlen=181)
gpu_container = deque([0] * 181, maxlen=181)

ram_container = deque([0] * 181, maxlen=181)
vram_container = deque([0] * 181,maxlen=181)

x_data = np.arange(-180, 1, 1)

# Ploty configurations
hovertemplate = """
    <b>%{x:%Y}</b><br>
    Usage: %{y}
    <extra></extra>
    """

color_palette = px.colors.qualitative.Set1

# # Template
# hp_template = dict(
#     layout=go.Layout(plot_bgcolor="black", paper_bgcolor="black")
# )


##### Static plot comparing execution times on 3 types:

df_benchmark = get_benchmarkings()
title = "Frames per second" 
suptitle = 'Performance Analysis for Object Detection Task Across Different Devices using YOLO12s'

fig_fps = create_figure(df_benchmark, var="fps", title=title, suptitle=suptitle, 
                        x_label="Device", y_label="Frames per second", theme="plotly_dark")

title = "Inference time" 
suptitle = 'Performance Analysis for Object Detection Task Across Different Devices using YOLO12s'
fig_inference = create_figure(df_benchmark, var='inference_time', title=title, 
                              suptitle=suptitle, x_label="Device", y_label="Time", 
                              ticksuffix='ms', theme="plotly_dark")



fig_benchmark_fps = dcc.Graph(id='benchmark-fps', figure=fig_fps, 
                              config={'staticPlot': True})
fig_benchmark_inference = dcc.Graph(id='benchmark-inference', figure=fig_inference, config={'staticPlot': True})

# More information about benchmark
fade_content = html.Div([
        dcc.Markdown("""
                    ##### More about benchmarking
                    
                    ###### Key metrics
                    - Inference time: Time it takes the model to predict one image. It does not include the preparation of the model (i.e. compiling)
                    or pre-processing. **Less is better**
                    - Frames per second: How many pictures can be processed per second. This includes the whole image processing procedure from making
                     the inference request to receiving the output. Simultaneous to throughput for a one second interval. **The higher the better**

                    ###### Devices shown
                    - CPU: Central processing unit - General-purpose, good for serial task processing.
                    - GPU: Graphical processing unit - Dedicated processor originally designed for gaming but excels at parallel processing. Has high power demands!
                    - iGPU: integrated GPU - Similar to GPU, but integrated into CPU and less powerful.
                    - NPU: Neural processing unit - Optimized for offloading AI and machine learning task, high efficiency and reasonable speed.
                     
                    ###### Models used:
                    YOLO12s:
                    - Latest iteration of the YOLO (**Y**ou **O**nly **L**ook **O**nce) model family
                    - Developed by the University of Buffalo and the University of Chinese Academy of Sciences 
                    - Click [here](https://arxiv.org/abs/2502.12524) for the paper link.
                    - Applied by using the [Ultralytics](https://www.ultralytics.com/) library 
                     
                    ###### Benchmarking:
                     - All benchmarks were done using the COCO128 dataset prepared by Ultralytics, 
                     that contain the first 128 images of the [COCO dataset](https://arxiv.org/abs/1405.0312)
                    - Benchmarks for the CPU, iGPU and NPU are done using the benchmark-tool provided by [OpenVino](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/benchmark-tool.html#)
                    - Benchmarking for the GPU was done via the benchmarking tool of Ultralytics, 
                     that conducts comparable tests on the GPU and provides comparable execution times
                    """,
                    link_target="_blank",)
])

fade_rm = html.Div(
    [
        dbc.Button("More info", id="fade-button-rm", className="mb-3", n_clicks=0,  style={"backgroundColor": "#004ad8", "borderColor": "#004ad8", "color": "white"}),
        dbc.Collapse(
            dbc.Card(
                dbc.CardBody(
                    fade_content
                )
            ),
            id="fade-rm",
            is_open=False,
        ),
    ]
)

##### INFO CARD
cpu = platform.processor()
cpu_cores = psutil.cpu_count(logical=False)

cpu_util = html.Div(
    [
        html.H5("Processor Utilisation"),
        dcc.Graph(id='cpu-usage')
    ]
)

gpu_util = html.Div(
    [
        html.H5("GPU Utilisation"),
        dcc.Graph(id='gpu-usage'),
    ]
)

##### Callbacks 
@callback(
    Output("fade-rm", "is_open"),
    [Input("fade-button-rm", "n_clicks")],
    [State("fade-rm", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(Output('utilization_graph_RM', "figure"),
          Input("global-interval", "n_intervals"),
          )
def update_usage(n_intervals):
    cpu_load = psutil.cpu_percent()

    # cpu_load = [(x/psutil.cpu_count()) * 100 for x in psutil.getloadavg()][1]
    cpu_container.append(cpu_load)
    GPUs = GPUtil.getGPUs() # -->commented because I don't have that. MD
    # GPUs = 'placeholder' # and added this
    if not GPUs:
        return
    gpu = GPUs[0] # -->commented because I don't have that. MD
    gpu_container.append(gpu.load * 100)

    cpu_line = go.Scatter(
        x=x_data,
        y=list(cpu_container),
        name="CPU",
        marker_color=color_palette[0]
    )
    gpu_line = go.Scatter(
        x=x_data,
        y=list(gpu_container),
        name="GPU",
        marker_color=color_palette[1],
    )
    title = "Device utilisation of GPU & CPU"
    suptitle = "Showing the usage percent of GPU & CPU in the past 60 seconds."
    fig = go.Figure(
        data=[cpu_line, gpu_line],
        layout=go.Layout(
            yaxis={"range": [0.1, 100], "title": "Usage", "ticksuffix": "%", 'fixedrange': True},
            xaxis={'range': [-60, 0], "title": "Seconds"},
            title={
            "text": f"{title}<br><sup style='font-weight: normal'>{suptitle}</sup>",
            "font_size": 20,
            "font_weight": "bold"},
            legend={
            "title": "<b>Device</b>"},
            dragmode="pan",
            template="plotly_dark",
            uirevision="a",
            # paper_bgcolor="lightgrey",
        )
    )
    return fig

@callback(Output('memory_graph_RM', "figure"),
          Input("global-interval", "n_intervals")
          )
def update_ram(n_intervals):

    ram = get_memory_usage()
    ram_container.append(ram["cpu"])
    vram_container.append(ram["gpu"])

    cpu_line = go.Scatter(
        x=x_data,
        y=list(ram_container),
        name="CPU",
        marker_color=color_palette[0]
    )
    gpu_line = go.Scatter(
        x=x_data,
        y=list(vram_container),
        name="GPU",
        marker_color=color_palette[1]
    )
    title = "Memory utilisation of GPU & CPU"
    suptitle = "Showing the memory usage for the CPU and GPU in the past 60 seconds."
    fig = go.Figure(
        data=[cpu_line, gpu_line],
        layout=go.Layout(
            yaxis={"range": [0.1, 100], "title": "Usage", "ticksuffix": "%", 'fixedrange': True},
            xaxis={'range': [-60, 0], "title": "Seconds"},
            title={
            "text": f"{title}<br><sup style='font-weight: normal'>{suptitle}</sup>",
            "font_size": 20,
            "font_weight": "bold"},
            legend={
            "title": "<b>Device</b>"},
            dragmode="pan",
            template="plotly_dark",
            uirevision="a",
            # plot_bgcolor="#060606", 
            # paper_bgcolor="#060606",
            colorscale_sequential=px.colors.qualitative.Set1,
            # paper_bgcolor="lightgrey",
        )
    )

    return fig


## Callback for plot 2: Memory
def get_memory_usage():
    # timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    # CPU RAM usage
    cpu_ram = psutil.virtual_memory().percent

    # GPU RAM usage
    gpus = GPUtil.getGPUs()
    gpu_ram = gpus[0].memoryUtil * 100 if gpus else 0

    # NPU RAM usage
    # try:
    #     npu_ram = torch.cuda.memory_reserved(0) / torch.cuda.get_device_properties(0).total_memory * 100
    #     # above --> amount of memory currently reserved by PyTorchon GPU/NPU (device is 0) over total memory to get percentage of current use.
    # except:
    #     npu_ram = 0 

    return {"cpu": cpu_ram, "gpu": gpu_ram}


### LAYOUT

def layout(**kwargs):
    page = html.Div([
        create_title_section('Resource monitor', 'Displaying resource usage in real-time'),
        # html.H1(children='Resource monitor'),
        html.H4("Real-time monitoring of CPU and GPU"),
        html.Br(),
        dbc.Row([
            dbc.Col([dcc.Graph(id = 'utilization_graph_RM'),], width=6),
            dbc.Col([dcc.Graph(id="memory_graph_RM"),], width=6),
        ]),
        ####### DYNAMIC PLOT : Live memory usage
        # dcc.Store(id="memory_store_RM", data=[]),
        # html.H4('Real-time memory usage'),
        # dcc.Interval(
        #     id="interval-component_RM",
        #     interval=1000,  # Change to update every X seconds
        #     n_intervals=0
        # ),
        html.Br(),
        html.H4('Benchmarking results'), # apparently needs a title to work
        dcc.Markdown("""
                     Following are two graphs showcasing the benchmark performance for different devices.
                     We tested 4 different devices, the CPU, the integrated GPU (iGPU), the NPU and the external
                     GPU. Two key performance parameters are shown:
                     - Inference time: How long does it take to process one picture
                     - Frames per Second: How many pictures can be processed per second
                     
                     Both parameters will give an indication, how well the devices perform for the given task.
                     
        """),
        fade_rm,
        dbc.Row([
                dbc.Col(fig_benchmark_fps, width=6),
                dbc.Col(fig_benchmark_inference, width=6)
            ]),
        html.Hr(),
        html.H5('Footnotes'),
        dcc.Markdown("""
        *1. The newness of NPU leads to restricted access due to missing access functionality. 
        Thus, resource usage for the NPU can not be shown. *
                     
        *2. Not all benchmarking tools allow the usage for each device as of 04/2025. Therefore, two different benchmark
        tools needed to be used. However, the utilisation and resulting execution time is considered comparable.*
        """),
        dcc.Interval(
            id="test-interval",
            interval=1000,  # Change to update every X seconds
            n_intervals=0
        ),
        # get_popover_chatbot(),
    ])
    return page