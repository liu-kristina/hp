import dash
from dash import html, dcc

# dash.register_page(__name__, path='/')

layout = html.Div(children=[
    html.H1(children='HP Switzerland - AI NPU Use Case'),

    html.Div(children='''
        - the benefit of moving from older generation to new AI PC from HP
        - the benefit of having AI tools on a local device instead of the cloud
    '''),
])



