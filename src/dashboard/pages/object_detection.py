from pathlib import Path
import warnings
import time
import logging
import base64
from io import BytesIO
from PIL import Image
import random
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import dash_daq as daq

import plotly.express as px
import plotly.graph_objects as go

from src.image_segmentation.yolo import get_yolo_model, predict_image, plot_image, benchmark_model
from src.utils.image_classification.helper_functions import decode_image
from src.utils.util_functions import get_devices
from src.dashboard.components.title_section import create_title_section

# from src.dashboard.components.upload_form import get_upload_form

warnings.simplefilter("ignore", DeprecationWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO)

dash.register_page(__name__, title="Object Detection", name="Object Detection", path='/object-detection', order=3)

upload_img = html.Div([
    dcc.Upload(id="upload-image", 
               children=[
                   html.Div([
                       "Drag and Drop or ", 
                       html.A("Select an Image File")
                    ]),
                ],
                style={
                    'width': '80%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                    },
                multiple=False,
                )
    ])

img_display = dcc.Loading(
    [html.Div(id="output-image-upload", className="output-image", style={"margin": "10px"})],
    overlay_style={"visibility":"visible", "filter": "blur(2px)"},
            custom_spinner=dbc.Spinner(),
        )

device_selection = dbc.Card(
    [
        dbc.CardHeader("Select device for task."),
        dbc.CardBody(
            dbc.RadioItems([f" {x} " for x in get_devices()], " CPU ", id='device-selection', inline=True),
            ),
    ],
    className="w-50",
    style={
        "background-color": "#1b2631",
    },
    outline=True,
)

tool_section = html.Div([
    dbc.Button("Run", color="primary", class_name="btn btn-primary", id="segment-button", n_clicks=0),
    # html.Div(id="segment-button"),
])

# device_selection = html.Div(className="dropdown-section", c:\Users\Jan\Downloads\benchmarks_fps.png c:\Users\Jan\Downloads\benchmarks_inference.pngchildren=[
#             html.Label("Select Processor:"),
#             dcc.Dropdown(
#                 id="processor-dropdown",
#                 options=[
#                     {"label": "CPU", "value": "CPU"},
#                     {"label": "GPU", "value": "GPU"},
#                     {"label": "NPU", "value": "NPU"}
#                 ],
#                 value="CPU",
#                 clearable=False,
#                 style={"width": "50%", "margin": "5px"}
#             )
#         ]),

result_section = html.Div(id="elapsed-time-segmentation")

benchmark_section = html.Div([
    dcc.Markdown("""
                 To truly test the capability, a processing device can offer, we provide you with a small
                 benchmark test. This benchmark is using a dataset with 128 images and calculates the average
                 time it needs to compute one image, as well as how many frames per second can be
                 processed with the given configuration. 
                 Press the benchmark button and test it out!
                 """),
    dbc.Button("Benchmark", class_name="btn btn-secondary", id="benchmark-button", n_clicks=0),
    html.Div(id="benchmark-results"),
])

# ~~~~~~~~~~~~~ Callbacks ~~~~~~~~~~~~~~
@callback(Output("global-store", "data"),
          Input("device-selection", "value"),
          State("global-store", "data"))
def select_device(value, data):
    logging.info("Selected %s", value)
    data["device"] = value.strip()
    return data

@callback([
    Output("output-image-upload", "children", allow_duplicate=True), 
    Output("elapsed-time-segmentation", "children"),
    Input("segment-button", "n_clicks"), 
    State("upload-image", "contents"),
    State("global-store", "data")
], prevent_initial_call=True)
def segment_image(n_clicks, contents, data):
    logging.info(f"Store values are: {data}")
    device = data["device"].strip()
    if n_clicks == 0 or not contents:
        return "",  ""
    logging.info(f"Selected {device}")
    img = decode_image(contents)
    model = get_yolo_model(device=device)
    start_time = time.time()
    res = predict_image(img, model, device)
    # fig = plot_image(res[0].plot())
    end_time = time.time()
    elapsed_time = end_time - start_time
    res_img = convert_img(res)
    # res_img = dcc.Graph(id="output-image-segment", figure=fig)
    return res_img, html.Div([html.H5("Elapsed time"), html.P(f"{elapsed_time:.2f} seconds"), html.Hr()])

@callback(Output("output-image-upload", "children", allow_duplicate=True),
          Output("benchmark-results", "children"),
          Input("benchmark-button", "n_clicks"),
          State("global-store", "data"),
          prevent_initial_call=True)
def run_benchmark(n_clicks, data):
    logging.info("Start benchmarking...")
    res, img = benchmark_model(device=data["device"])
    # Plot random image

    res_img = convert_img(img)
    return res_img, html.Div([
        html.Br(),
        html.H5("Benchmarking results:"),
        html.Div([
            dcc.Markdown(f"""
                   Running the benchmark on the **{data["device"].upper()}**:
                   - Per image, **{res["inference_time"]} milliseconds** are needed,
                   - **{res["fps"]}** frames per seconds can be processed.
                   """
                )
        ]),
    ])
        

def convert_img(res):
    img = Image.fromarray(res[0].plot()[:, :, ::-1])
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return html.Img(src=f"data:image/png;base64,{img_str}", style={"maxWidth": "100%", "height": "auto"})

def layout(**kwargs):
    content = html.Div(
        [
            create_title_section("Object detection", "Using AI powered tools for semantic object segmentation and classification."),
            dcc.Markdown(children='''
                Paired with the OpenVINO™ toolkit, the NPU accelerates computer vision workloads such as classification, detection, and segmentation — all with seamless integration and minimal setup.
                - Detect multiple objects in real-time with high precision.  
                - Enhance security, automate quality control, and optimize logistics with advanced image analysis.  
                '''),
            dbc.Row([
                dbc.Col(upload_img),
                dbc.Col([device_selection, html.Br(), tool_section,]),
            ]),
            dbc.Row([
                dbc.Col(img_display),
                dbc.Col([
                    html.Hr(),
                    result_section,
                    benchmark_section
                ])
            ]),
        ]
    )
    return content

