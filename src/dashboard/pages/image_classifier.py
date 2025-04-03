import dash 
import logging
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import time
from src.utils.image_classification.helper_functions import (
    classify_image, 
    run_benchmark_app, 
    decode_image, 
    ensure_model_files
)
from src.utils.util_functions import get_devices

# from src.dashboard.components.chatbot_button import get_popover_chatbot
from src.dashboard.components.title_section import create_title_section
from src.vis.benchmark_plots_resnet import create_figure, get_benchmarkings


logger = logging.getLogger(__name__)
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO)


# -----------------------
# Initialize the page
dash.register_page(__name__, title="Image Classification", name="Image Classification", path='/image-classification', order=2)


title_section = create_title_section("Image Classification", "Benchmarking using OpenVINO")

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
    [html.Div(id="output-image-upload", className="output-image-class", style={"margin": "10px"})],
    overlay_style={"visibility":"visible", "filter": "blur(2px)"},
            custom_spinner=dbc.Spinner(),
        )

device_selection = dbc.Card(
    [
        dbc.CardHeader("Select device for task."),
        dbc.CardBody(
            dbc.RadioItems([f" {x} " for x in get_devices()], " CPU ", id='device-selection-class', inline=True),
            ),
    ],
    className="w-75",
    style={
        "background-color": "#1b2631",
    },
    outline=True,
)

tool_section = html.Div([
    dbc.Button("Run", color="primary", id="classify-button", n_clicks=0,
               style={"backgroundColor": "#004ad8", "borderColor": "#004ad8", "color": "white"}),
    html.Div(id="classify-button"),
    # html.Div(id="elapsed-time-classify")
])

result_section = html.Div(id="elapsed-time-classify")

benchmark_section = html.Div([
    dcc.Markdown("""
                 To truly test the capability, a processing device can offer, we provide you with a small
                 benchmark test. This benchmark is using a dataset with 128 images and calculates the average
                 time it needs to compute one image, as well as how many frames per second can be
                 processed with the given configuration. 
                 Press the benchmark button and test it out!
                 """),
    # dbc.Button("Benchmark", id="benchmark-button-classify", n_clicks=0, class_name="btn btn-secondary"),
    # html.Div(id="benchmark-result-classify"),
])

benchmark_button = html.Div([
    dbc.Button("Benchmark", id="benchmark-button-classify", n_clicks=0, class_name="btn btn-secondary"),
    html.Div(id="benchmark-result-classify"),
])

##### Static plot comparing execution times on 3 types:


df_benchmark = get_benchmarkings()
# title = "Benchmark results" 
# suptitle = 'Latency Time for Classification Across Different Devices using ResNet-50'

# fig_latency = create_figure(df_benchmark, 'latency (ms)', title, suptitle, 
#                             "Device", "Time", 'ms', purpose="app")

title = "Inference time" 
suptitle = 'Less is better'
fig_inference = create_figure(df_benchmark, var='inference_time', title=title, 
                              suptitle=suptitle, x_label="Device", y_label="Time", 
                              ticksuffix='ms', purpose="app")

title = "Frames per second" 
suptitle = 'More is better'
fig_throughput = create_figure(df_benchmark, 'fps', title, suptitle, "Device", "Frames per Second", purpose="app")

fig_benchmark_fps = dcc.Graph(id='benchmark-fps', figure=fig_throughput, 
                              config={'staticPlot': True})
fig_benchmark_inference = dcc.Graph(id='benchmark-latency', figure=fig_inference, config={'staticPlot': True})


############## FADE BUTTON

fade_content = html.Div([
        dcc.Markdown("""
                    ##### More about performance metrics
                    
                    ###### Key metrics
                    - Elapsed time: Total time from start to result.
                    - Latency: Time per inference (The process where the AI model makes a prediction, the lower the better).
                    - Throughput: Number of inferences per second (the higher the better).

                    ###### Device considerations
                    - CPU: General-purpose, slower.
                    - GPU: Faster for parallel processing but higher power use.
                    - NPU: Optimized for AI, best efficiency and speed.
                     
                    ###### Other possible use cases
                    - Manufacturing: Quality control with AI-powered defect detection.
                    - Finance & ID Verification: Process identity documents securely on-device.
                    - Smart offices: AI-enhanced document scanning, handwriting recognition, and automation.
                    - Medical imaging: Analyze X-rays or MRIs without sending sensitive data to the cloud.

                    """)
])

fadeIC = html.Div(
    [
        dbc.Button("More info", id="fade-button-ic", className="mb-3", n_clicks=0,  
                   style={"backgroundColor": "#004ad8", "borderColor": "#004ad8", "color": "white"}),
        dbc.Collapse(
            dbc.Card(
                dbc.CardBody(
                    fade_content
                )
            ),
            id="fade-ic",
            is_open=False,
        ),
    ]
)

###### CALBACKS #######
##### Callbacks 
@callback(
    Output("fade-ic", "is_open"),
    [Input("fade-button-ic", "n_clicks")],
    [State("fade-ic", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(Output("global-store", "data", allow_duplicate=True),
          Input("device-selection-class", "value"),
          State("global-store", "data"),
          prevent_initial_call=True)
def select_device(value, data):
    logging.info("Selected %s", value)
    data["device"] = value.strip()
    return data

# Callback to display the uploaded image immediately.
# @callback(
#     Output("output-image-upload", "children"),
#     [Input("upload-image", "contents")]
# )
# def update_uploaded_image(contents):
#     if contents is None:
#         return ""
#     return html.Img(src=contents, className="uploaded-image", style={"maxWidth": "100%", "height": "auto"})

# Callback for image classification


@callback(
    [Output("output-image-upload", "children", allow_duplicate=True),
     Output("elapsed-time-classify", "children"),
     Input("classify-button", "n_clicks"),
     State("upload-image", "contents"),
     State("global-store", "data"),],
     prevent_initial_call=True)
def update_classification(n_clicks, contents, data):
    if n_clicks == 0 or contents is None:
        return "", ""
    
    # core = ov.Core()
    # available_devices = core.available_devices
    # if processor in ["GPU", "NPU"] and processor not in available_devices:
    #     error_message = (f"Error: Selected processor '{processor}' is not available on this system. "
    #                      f"Available devices: {available_devices}")
    #     return error_message, ""
    device = data["device"].strip()
    image = decode_image(contents)
    start_time = time.time()
    try:
        class_label = classify_image(image, device)
    except Exception as e:
        return f"Error during classification: {str(e)}", ""
    end_time = time.time()
    elapsed_time = end_time - start_time
    return html.Img(src=contents), html.Div([html.H5("Predicted Class: "), html.P(f"{class_label}"), html.H5("Elapsed time"), html.P(f"{elapsed_time:.2f} seconds"), html.Hr()])

# Callback for benchmarking
@callback([
    Output("output-image-upload", "children", allow_duplicate=True),
    Output("benchmark-result-classify", "children"),
    Input("benchmark-button-classify", "n_clicks"),
    State("upload-image", "contents"),
    State("global-store", "data"),],
    prevent_initial_call=True
)
def update_benchmark(n_clicks, contents, data):
    if n_clicks == 0 or contents is None:
        return [""]
    
    # core = ov.Core()
    # available_devices = core.available_devices
    # if processor in ["GPU", "NPU"] and processor not in available_devices:
    #     error_message = (f"Error: Selected processor '{processor}' is not available on this system. "
    #                      f"Available devices: {available_devices}")
    #     return error_message, ""
    
    device = data["device"].strip()
    if device == 'iGPU':
        device = 'GPU'
    # Run benchmark for latency
    logging.info(device)
    result_latency = run_benchmark_app(device=device, hint="latency")
    latency_out = "N/A"
    for line in result_latency.split("\n"):
        if "Average" in line:
            parts = line.split()
            latency_out = parts[-2] + " " + parts[-1]
            break
    
    # Run benchmark for throughput
    result_throughput = run_benchmark_app(device=device, hint="throughput")
    throughput_out = "N/A"
    for line in result_throughput.split("\n"):
        if "Throughput" in line:
            parts = line.split()
            throughput_out = parts[-2] + " " + parts[-1]
            break
    
    return html.Img(src=contents), html.Div([html.H5("Latency:"), html.P(latency_out), html.H5("Throughput:"), html.P(throughput_out)])

##### LAYOUT #####

layout = dbc.Container([
    title_section,
    dcc.Markdown(children='''
                This page allows you to classify an image using a locally optimized AI model powered by OpenVINO. 
                Benchmarking helps evaluate performance across different types of processors.  
                '''),
    dbc.Row([
        dbc.Col([
            dbc.Row([upload_img, img_display]),
            dbc.Row(fadeIC)
        ], width=6),
        dbc.Col([
            dbc.Row([device_selection]),
            html.Br(),
            dbc.Row(tool_section),
            html.Hr(),
            dbc.Row([result_section, benchmark_section]),
            dbc.Row(benchmark_button)
        ], width=6),
    ]),
    # dbc.Row([
    #     dbc.Col(upload_img, width=6),
    #     dbc.Col([device_selection, html.Br(), tool_section], width=6)
    # ]),
    # dbc.Row([
    #     dbc.Col(img_display),
    #     dbc.Col([html.Hr(), result_section, benchmark_section]),
    # ]),
    # dbc.Row([
    #     dbc.Col(fadeIC),
    #     dbc.Col(benchmark_button), 
    #     ]),
    html.Hr(),
    html.H4("Benchmarking results across different devices"),
    html.Br(),
    dbc.Row([
                dbc.Col(fig_benchmark_fps, width=6),
                dbc.Col(fig_benchmark_inference, width=6)
            ]),
])


'''
layout = html.Div([
    # Fixed Title Section
    title_section,
    device_selection,
    dbc.Row([
        dbc.Col([upload_section], width=6),
        dbc.Col([result_section], width=6)
    ]),
    fadeIC], 
    # style={
    # "display": "flex", 
    # "height": "100vh",  
    # "overflow": "auto"  
    # }
    )
# Chatbot button (if applicable)
# get_popover_chatbot()
'''

if __name__ == "__main__":
    None
