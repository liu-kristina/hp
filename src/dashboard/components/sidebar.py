import warnings
from pathlib import Path
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State
import dash
import os
import psutil
import GPUtil
import platform
import plotly.graph_objects as go
from dotenv import load_dotenv
import logging

warnings.simplefilter("ignore", DeprecationWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO)

load_dotenv('.env')
computer_name = os.getenv("COMPUTER_NAME")
logging.info('Working on the %s laptop', computer_name)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#1b2631",
    "color": "#f8f9fa", # text color
    #"border-right": "2px solid #17a2b8" # accent color,
    "z-index": 1000, 
}

if computer_name == "HP ZBook Power G11":
    device_specifications = {
        'cpu': 'Intel(R) Core(TM) Ultra 9 185H' ,
        'npu': 'Intel(R) AI Boost',
        'gpu': 'NVIDIA RTX 3000 Ada (8 GB)',
        'ram': '64 GB DDR5 RAM'
    }
elif computer_name == "HP ZBook Firefly G11":
    device_specifications = {
        'cpu': 'Intel(R) Core(TM) Ultra 7 165H' ,
        'npu': 'Intel(R) AI Boost',
        'gpu': 'NVIDIA RTX A500 (4 GB)',
        'ram': '32 GB DDR5 RAM'
    }
else:
    cpu = platform.processor()
    svmem = psutil.virtual_memory()
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0].name
    else:
        gpu = 'n/a'
    device_specifications = {
        'cpu': cpu,
        'npu': 'n/a',
        'gpu': gpu,
        'ram': f"{svmem.total / (1024**3):.2f} GB"
    }


# the styles for the main content position it to the right of the sidebar and
# add some padding.


# sidebar_header = dbc.Row(
#     [
#         dbc.Col(html.H2("Sidebar", className="display-4")),
#         dbc.Col(
#             [
#                 html.Button(
#                     # use the Bootstrap navbar-toggler classes to style
#                     html.Span(className="navbar-toggler-icon"),
#                     className="navbar-toggler",
#                     # the navbar-toggler classes don't set color
#                     style={
#                         "color": "rgba(0,0,0,.5)",
#                         "border-color": "rgba(0,0,0,.1)",
#                     },
#                     id="navbar-toggle",
#                 ),
#                 html.Button(
#                     # use the Bootstrap navbar-toggler classes to style
#                     html.Span(className="navbar-toggler-icon"),
#                     className="navbar-toggler",
#                     # the navbar-toggler classes don't set color
#                     style={
#                         "color": "rgba(0,0,0,.5)",
#                         "border-color": "rgba(0,0,0,.1)",
#                     },
#                     id="sidebar-toggle",
#                 ),
#             ],
#             # the column containing the toggle will be only as wide as the
#             # toggle, resulting in the toggle being right aligned
#             width="auto",
#             # vertically align the toggle in the center
#             align="center",
#         ),
#     ]
# )

# sidebar = html.Div(
#     [
#         sidebar_header,
#         # we wrap the horizontal rule and short blurb in a div that can be
#         # hidden on a small screen
#         html.Div(
#             [
#                 html.Hr(),
#                 html.P(
#                     "A responsive sidebar layout with collapsible navigation " "links.",
#                     className="lead",
#                 ),
#             ],
#             id="blurb",
#         ),
#         # use the Collapse component to animate hiding / revealing links
#         dbc.Collapse(
#             dbc.Nav(
#                 [
#                     dbc.NavLink("Dashboard", href="/", active="exact"),
#                     dbc.NavLink("Ressource monitor", href="/resource-monitor", active="exact"),
#                     dbc.NavLink("Page 2", href="/page-2", active="exact"),
#                 ],
#                 vertical=True,
#                 pills=True,
#             ),
#             id="collapse",
#         ),
#     ],
#     id="sidebar",
# )



# @callback(
#     Output("sidebar", "className"),
#     [Input("sidebar-toggle", "n_clicks")],
#     [State("sidebar", "className")],
# )
# def toggle_classname(n, classname):
#     if n and classname == "":
#         return "collapsed"
#     return ""


# @callback(
#     Output("collapse", "is_open"),
#     [Input("navbar-toggle", "n_clicks")],
#     [State("collapse", "is_open")],
# )
# def toggle_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open

cpu = platform.processor()
cpu_cores = psutil.cpu_count(logical=False)
info_card = html.Div([
    dbc.Row([html.H5(" Selected device", className="fw-semibold"),]),
    dbc.Row([
        html.P([computer_name]),
        html.P([
            f'CPU: {device_specifications['cpu']}\n', html.Br(),
            f'NPU: {device_specifications['npu']}\n', html.Br(),
            f'GPU: {device_specifications['gpu']}\n', html.Br(),
            f'RAM: {device_specifications['ram']}\n',
        ],
            style={'font-size': '14px'})
    ],
    className="mt-0 text-muted mb-0 pt-0 pb-0"),
])



cpu_util = html.Div(
    [
        dbc.Row([
            html.B('Device usage:')
        ]),
        dbc.Row([
            dbc.Col([
                html.Span([html.B("CPU:"), html.P(id='cpu-usage_side', className="fw-semibold"),]),
            ]),
            dbc.Col([
                html.Span([html.B("GPU:"), html.P(id='gpu-usage_side', className="fw-semibold")]),
            ])
        ]),
    ],
)


@callback(
        Output("cpu-usage_side", "children"),
        Input('global-interval', "n_intervals")
)
def get_cpu_freq(n_intervals):
    # cpu_load = psutil.cpu_percent(interval=None)
    cpu_load = psutil.cpu_percent()
    # cpu_load = [(x/psutil.cpu_count()) * 100 for x in psutil.getloadavg()][1]
    # x_used = psutil.cpu_percent()
    # x_unused = 100 - x_used
    return f"{cpu_load:.2f}%"

@callback(
        Output("gpu-usage_side", "children"),
        Input('global-interval', "n_intervals")
)
def get_gpu_freq(n_intervals):
    GPUs = GPUtil.getGPUs()
    gpu = GPUs[0]
    gpu_load = gpu.load * 100
    return f"{gpu_load:.2f}%"


def create_sidebar(assets):
    sidebar = dbc.Container(
        [   
            html.Img(
            src='assets/0_HP_logo.png', 
            className="hp-logo",
            style={'width': '80px', 
                   'height': 'auto', 
                   'margin-right': '10px'},
            ),
            # html.H2("Sidebar", className="display-4"),
            html.Hr(),
            # html.P("Content", className="lead fw-semibold"),
            dbc.Nav(
                [dbc.NavLink(
                    f"{page['name']}", 
                    href=f"{page['path']}", 
                    active="exact") for page in dash.page_registry.values()
                    ],
                vertical=True,
                pills=True,
                class_name='px-2 mt-0 mb-0'
            ),
            html.Hr(),
            cpu_util,
            html.Hr(),
            info_card
        ],
        style=SIDEBAR_STYLE,
        class_name='container-fluid'
    )
    return sidebar