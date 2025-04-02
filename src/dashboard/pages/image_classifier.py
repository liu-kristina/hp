import dash 
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import time
import openvino as ov
from src.utils.image_classification.helper_functions import (
    classify_image, 
    run_benchmark_app, 
    decode_image, 
    ensure_model_files
)
from src.utils.util_functions import get_devices

# from src.dashboard.components.chatbot_button import get_popover_chatbot
from src.dashboard.components.title_section import create_title_section

# -----------------------
# Initialize the page
dash.register_page(__name__, title="Image Classification", name="Image Classification", path='/image-classification')


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
            dcc.RadioItems([f" {x} " for x in get_devices()], " CPU ", id='device-selection-class'),
            ),
    ],
    className="w-50",
    outline=False,
)

tool_section = html.Div([
    dbc.Button("Run", color="primary", id="classify-button", n_clicks=0),
    html.Div(id="classify-button"),
    # html.Div(id="elapsed-time-classify")
])

result_section = html.Div(id="elapsed-time-classify")

benchmark_section = html.Div([
    dcc.Markdown("""
                TO DO: EXPLAIN BENCHMARKS
                 """),
    dbc.Button("Benchmark", id="benchmark-button-classify", n_clicks=0, class_name="btn btn-secondary"),
    html.Div(id="benchmark-result-classify"),

])


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
        dbc.Button("More info", id="fade-buttonIC", className="mb-3", n_clicks=0,  style={"backgroundColor": "#004ad8", "borderColor": "#004ad8", "color": "white"}),
        dbc.Fade(
            dbc.Card(
                dbc.CardBody(
                    fade_content
                )
            ),
            id="fadeIC-alt",
            is_in=False,
            appear=False,
        ),
    ]
)

###### CALBACKS #######
@callback(
    Output("fadeIC-alt", "is_in"),
    [Input("fade-buttonIC", "n_clicks")],
    [State("fadeIC-alt", "is_in")],
)
def toggle_fade(n, is_in):
    if not n:
        # Button has never been clicked
        return False
    return not is_in

@callback(Output("global-storeIC", "data"),
          Input("device-selection-class", "value"),
          State("global-storeIC", "data"))
def select_device(value, data):
    #logging.info("Selected %s", value)
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
    [Output("elapsed-time-classify", "children"),
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
    
    return [html.Div([html.H5("Predicted Class: "), html.P(f"{class_label}"), html.H5("Elapsed time"), html.P(f"{elapsed_time:.2f} seconds"), html.Hr()])]

# Callback for benchmarking
@callback([
    Output("benchmark-result-classify", "children"),
    Input("benchmark-button-classify", "n_clicks"),
    State("upload-image", "contents"),
    State("global-store", "data"),],
    prevent_initial_call=True
)
def update_benchmark(n_clicks, contents, data):
    if n_clicks == 0 or contents is None:
        return ""
    
    # core = ov.Core()
    # available_devices = core.available_devices
    # if processor in ["GPU", "NPU"] and processor not in available_devices:
    #     error_message = (f"Error: Selected processor '{processor}' is not available on this system. "
    #                      f"Available devices: {available_devices}")
    #     return error_message, ""
    
    device = data["device"].strip()
    # Run benchmark for latency
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
        if "Average" in line:
            parts = line.split()
            throughput_out = parts[-2] + " " + parts[-1]
            break
    
    return [html.Div([html.H5("Latency:"), html.P(latency_out), html.H5("Throughput:"), html.P(throughput_out)])]

##### LAYOUT #####

layout = html.Div([
    title_section,
    dcc.Markdown(children='''
                This page allows you to classify an image using a locally optimized AI model powered by OpenVINO. 
                Benchmarking helps evaluate performance across different types of processors.  
                '''),
    dbc.Row([
        dbc.Col(upload_img, width=6),
        dbc.Col([device_selection, tool_section], width=6)
    ]),
    dbc.Row([
        dbc.Col(img_display),
        dbc.Col([html.Hr(), result_section, benchmark_section]),
       
    ]),
    dbc.Row([ 
        dbc.Col(fadeIC),
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
