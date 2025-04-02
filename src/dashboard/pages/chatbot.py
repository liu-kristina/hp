import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from src.dashboard.components.chatbot_button import get_chatbot

# dash.register_page(__name__)

header = html.Div(
    [
        html.H1("Chatbot test"),
        html.P("This is a testpage for a simple chatbot running locally."),
        html.P("For now, the page is a placeholder to show a different presentation of the Chatbot")
        # "an explanation of NPU and a presentation of the Laptop used will be presented. The chatbot is implemented to answer questions regarding the Laptop and its capabilities. Lastly, if the layout allows, this page should include some performance display for NPU/GPU/CPU capabilities")
    ]
)

content_2 = html.Div(
    [
        html.P("This is nonsense text to show how the chatbot will be embedded between two text paragraphs"),
        html.P("Further test test test"),
        html.P("Test test test")
        # "an explanation of NPU and a presentation of the Laptop used will be presented. The chatbot is implemented to answer questions regarding the Laptop and its capabilities. Lastly, if the layout allows, this page should include some performance display for NPU/GPU/CPU capabilities")
    ]
)

chatbot_layout = get_chatbot()


def layout(**kwargs):
    page = dbc.Container(
        fluid=True,
        children=[
            header,
            html.Hr(),
            dcc.Store(id="store-conversation", data=[]),
            chatbot_layout,
            html.Hr(),
            content_2,
        ],
    )
    return page