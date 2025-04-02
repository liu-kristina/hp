import dash
from dash import html

# dash.register_page(__name__, path='/')

layout = html.Div(children=[
    html.H1(children='Creating a tool on a local device'),

    html.Div(children='''
        Comparing specs as performance / battery life / latency etc. between two different devices 
        - On 1st  device RAG is on GPU or NPU (1st Option)
        - On 2nd device RAG runs on CPU

        You can visulize:
        - Real-time on the device
        - Comparing speed 
    '''),
])