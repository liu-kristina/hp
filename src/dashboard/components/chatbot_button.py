import logging
import os
from dotenv import load_dotenv

load_dotenv('.env')

from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# from src.llm.llm_model import get_model_name, prompt_llm
from src.rag.rag_interface import init_rag

logger = logging.getLogger(__name__)
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO)

model = os.getenv('LLM_MODEL')
if not model:
    model = "gemma3:12b-it-q4_K_M"
# model = "gemma3:12b-it-q4_K_M"
# model = "llama3.2"
# model = " mistral-small:22b"
# model = "mistral"
device = "cpu"
backend = None
RAG = init_rag(model, device, backend)

def textbox(text, box="other"):
    style = {
        "max-width": "55%",
        "width": "max-content",
        "padding": "0px 0px",
        "border-radius": "20px",
    }

    if box == "self":
        style["margin-left"] = "auto"
        style["margin-right"] = 0

        color = "primary"
        inverse = True

    elif box == "other":
        style["margin-left"] = 0
        style["margin-right"] = "auto"

        color = "light"
        inverse = False

    else:
        raise ValueError("Incorrect option for `box`.")

    return dbc.Card(dcc.Markdown(text), style=style, body=True, color=color, inverse=inverse)

conversation = html.Div(
    style={
        "width": "80%",
        # "max-width": "800px",
        "height": "50vh",
        "margin": "auto",
        "overflow-y": "auto",
    },
    id="display-conversation-popover",
)

controls = dbc.InputGroup(
    style={"width": "90%", "margin": "auto"},
    children=[
        dbc.Input(id="user-input-chatbot", placeholder="Write to the chatbot..."),
        dbc.Button("Submit", id="submit-popover", n_clicks=0)
    ],
)

# controls_pop = dbc.InputGroup(
#     style={"width": "90%", "margin": "auto"},
#     children=[
#         dbc.Input(id="user-input-chatbot", placeholder="Write to the chatbot..."),
#         dbc.Button("Submit", id="submit-popover", n_clicks=0)
#     ],
# )

chatbot_footer = html.Footer(f"Powered by {RAG.llm.model}", className="fw-lighter", style={'textAlign': 'right'})

chatbot_store = dcc.Store(id="store-conversation-chatbot", data=[])

# chatbot_layout = html.Div(
#     [
#         chatbot_store,
#         conversation, 
#         dcc.Loading(
#             [controls],
#             overlay_style={"visibility":"visible", "filter": "blur(2px)"},
#             custom_spinner=dbc.Spinner(),),
#         chatbot_footer, 
#     ]
# )

chatbot_layout = html.Div(
    [
        chatbot_store,
        conversation, 
        dcc.Loading(
            [controls],
            overlay_style={"visibility":"visible", "filter": "blur(2px)"},
            custom_spinner=dbc.Spinner(),),
        chatbot_footer, 
    ]
)

# Popover button
# 40px from right is good value
but = dbc.Button(html.I(className="bi bi-chat-right-text fs-4"), className="me-1",
                 n_clicks=0,
                 id="ask-ai-popover",
                 style={'position': 'fixed', 
                        "bottom": "65px", 
                        "right": "40px",
                        "border-radius": "50%",
                        "width": "65px",
                        "height": "65px"}
        )

pop_over = dbc.Popover(
            dbc.PopoverBody(chatbot_layout),
            target="ask-ai-popover",
            # trigger="click",
            id="popover",
            placement="left",
            is_open=False,
            offset="-250,20",
            style={"width": "40%","max-width": "40%"},
            persistence=True,
            persistence_type="session",
        )

ask_ai_pop = html.Div([but, pop_over])


# Callbacks
@callback(
    Output("popover", "is_open"),
    [Input("ask-ai-popover", "n_clicks")],
    [State("popover", "is_open")],
)
def toggle_popover(n, is_open):
    logging.debug("Popover button toggled")
    if n:
        return not is_open
    return is_open

@callback(
    Output("display-conversation-popover", "children"), [Input("store-conversation-chatbot", "data")]
)
def update_display(chat_history):
    return [
        textbox(x, box="self") if i % 2 == 0 else textbox(x, box="other")
        for i, x in enumerate(chat_history)
    ]


# def call_chatbot(prompt):
#     return RAG.invoke(prompt)


# @callback(Output())
# def delegate_prompt()

@callback(
    [Output("store-conversation-chatbot", "data"), Output("user-input-chatbot", "value")],
    [Input("submit-popover", "n_clicks"), Input("user-input-chatbot", "n_submit")],
    [State("user-input-chatbot", "value"), State("store-conversation-chatbot", "data")],
    prevent_initial_call=True,
)
def run_chatbot(n_clicks, n_submit, user_input, chat_history):

    logging.debug("Run chatbot triggered")
    # logging.info(f"Or Passes submission {n_submit_pop}")
    if not n_clicks and not n_submit:
        logging.debug("No clicks and no submission")
        return chat_history, ""
    if not user_input:
        logging.debug("No prompt was provided")
        return chat_history, ""
    logging.info("Submit prompt triggered")
    # Add the user input to chat history
    chat_history.append(user_input)
    ans = RAG.invoke(user_input)
    logging.debug("Chatbot invoked")
    chat_history.append(ans)  # Simulated chatbot response
    
    # # temporary
    # return chat_history + user_input + "<|endoftext|>" + user_input + "<|endoftext|>", ""

    return chat_history, ""

def get_popover_chatbot():
    return ask_ai_pop

def get_chatbot():
    return chatbot_layout