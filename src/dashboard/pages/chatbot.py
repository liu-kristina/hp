# Import packages
import dash
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from src.dashboard.components.title_section import create_title_section



# For processing all PDFs
from src.utils.PDFprocessor02_Marisa import PDFprocessor
# from src.dashboard.components.chatbot_button import get_chatbot
# from src.rag.generate_vectorstore import get_vectorstore, add_document

# -----------------------
# Initialize the page
dash.register_page(__name__, title="Chatbot", name="Chatbot", path='/chatbot', order=1)


# Initialize the PDFprocessor

description_section = html.Div([
        dcc.Markdown("""
            ##### Getting started
            * Ask questions about HP workstations and get insights on specs, and recommendations based on internal HP documents.

            ##### How to use the chatbot
            * Click the chat icon in the bottom right corner to start a conversation. 
              This AI assistant is designed specifically for HP users, retrieving information from preloaded HP documents to provide precise answers tailored to HP products and services.*
            
            ##### Why run locally?
            * Unlike other AI tools running in the cloud, this chatbot operates locally on your device. 
              It doesn’t require an internet connection to function and ensures complete privacy by keeping all interactions and data processing on your machine.
            
            ##### Use-Cases:
            * Give your AI assistant access to internal an/or sensible documents to increase productivity without compromising privacy
            * Leverage the power of your AI chatbot in the field, on an airplane or without straining your internet traffic
            * Tailor your personal AI on your personal workstation to give the answers you need.

            ##### What can an AI workstation offer?
            * Large language models (LLM) require substantial processing power and memory of their hardware,
            leveraging the power of NPU and GPU devices allows for faster processing time and more functionality
            of AI applications.
            * Using the NPU, task requiring parallel processing can be handled more efficiently by the NPU and 
            reliefs the strain and power demand of either the CPU and/or GPU
            * AI tasks are not limited to LLM models, even during video calling, text, video, or audio processing,
            and many more tasks, AI tasks can be accelerated via increased processing power an AI workstation can offer.
            
            ##### You have more questions?
            * Ask the chatbot and see for yourself!
            """)
    ],
            style={
            'padding-left': '50px',
            'padding-top': '10px' })

# Notes
notes = html.Div([html.P("""
               *Memory availability can lead to slow loading times or no responsiveness of the Chatbot. If the Chatbot-button does not respond,
                try reloading the page and/or wait a few more minutes.
                """, style={'font-size': '11px'})
        ])

# Additional information
more_info = html.Div([
    dcc.Markdown("""
                 
                 ##### How is the chatbot implemented?
                 The chatbot is implemented offline, where a service called [Ollama](https://ollama.com/) 
                 is providing the deployment of the large language model (LLM) on the local machine.
                 The LLM is further refined by the RAG method (see explanation below) to provide
                 answers based on the provided documents. 

                 ##### What is RAG?
                 The chatbot is using the so-called *retrieval-augemnted generation* (RAG) method. 
                 It is leveraging the power of large language models (LLM) by having access
                 to an internal 'store' or 'library' containing your uploaded documents.
                 By cleverly adapting the prompts for the LLM, it can make use of the information
                 in your documents and combine it with the power of the model itself. By having
                 a self-correcting implementation of the RAG, the generated answers are tested for their
                 reliability and rewritten, if they are not grounded in the facts of the document library.
                 More information about this topic can be found [here](https://selfrag.github.io/)

                 ##### Availabilty of LLM on NPU
                 Due to the recent nature of NPU's in laptops, leveraging the full power of NPU is 
                 as of 04/2025 still limited. Accessing LLM for the use in this APP is dependent on
                 the implementation and training by the developers of the various Python packages.
                 Thus, only few functionalities of running a LLM on the NPU are possible. This will most 
                 probably change in the future, as even during the development of this project,
                 more and more functionalities for the NPU became available.
                 """)
])

### Collapse use-cases

use_cases_box = html.Div([
        dcc.Markdown("""
                    ##### Other use cases where a local AI assistant may be of help:
                    ###### Faster support
                    * Employees ask questions about company policies
                    * Users upload manuals and get quick answers on setup, features, or troubleshooting.

                    ###### Legal & Compliance
                    * Contract review: Upload a contract or a compliance document and ask about key clauses and policies.

                    ###### Research
                    * Upload reports and extract insights without reading the entire document.

                    ###### Customer service & Sales enablement
                    * Sales training: Sales reps ask about product specs, comparisons, or use cases without digging through PDFs.
                    * Customer queries: AI quickly answers FAQs about pricing, features, or compatibility.
                    """)
])


fade = html.Div(
    [
        dbc.Button("More information", id="fade-button", className="mb-3", n_clicks=0,  style={"backgroundColor": "#004ad8", "borderColor": "#004ad8", "color": "white"}),
        dbc.Collapse(
            dbc.Card(
                dbc.CardBody(
                    more_info
                )
            ),
            id="fade",
            is_open=False,
        ),
    ]
)

@callback(
    Output("fade", "is_open"),
    [Input("fade-button", "n_clicks")],
    [State("fade", "is_open")],
)
def toggle_fade(n, is_open):
    if n:
        return not is_open
    return is_open



### Chatbot placeholder
# chatbot_placeholder = html.Div([
#         html.H3("Chatbot placeholder"),
#         html.P("This is nonsense text to show how the chatbot will be embedded between two text paragraphs"),
#         html.P("Further test test test"),
#         html.P("Test test test")
#         # "an explanation of NPU and a presentation of the Laptop used will be presented. The chatbot is implemented to answer questions regarding the Laptop and its capabilities. Lastly, if the layout allows, this page should include some performance display for NPU/GPU/CPU capabilities")
#     ], style={'border': '2px dashed gray', 'padding': '20px', 'margin-bottom': '20px'}
# )

# chatbot_layout = get_chatbot()

# chatbot_layout.children

### PDF uploader
pdf_uploader = html.Div([
    html.H3("Upload a PDF File"),
    dcc.Upload(
        id='upload-pdf',
        children=html.Div([
            'Drag and drop or ', html.A('click to upload a PDF file')
        ]),
        style={
            'width': '80%',
            'font-family': 'sans-serif',
            'font-size': '18px',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True  # Allow multiple files to be uploaded
    ),
    html.Div(id='output-data-upload')
])

# Footer image
footer_image = html.Div([
    html.Img(src='/assets/4_footer.png', style={'width': '100%', 'height': '40px','margin-top': '30px'})
])

# App layout
def layout (**kwawrgs):
    page = dbc.Container(
        fluid=True,
        children=[
            create_title_section('Local AI-powered HP workstation assistant', 
                         "Get instant answers on HP Workstations from your local database"),
            description_section,
            notes,
            #collapse,
            fade,
            html.Hr(),
            html.H5('Footnotes'),
            dcc.Markdown("""
                *1. All answers provided by the Chatbot should be takin with care. This chatbot is an automated system designed to provide information and assistance
                         based on provided Documents by HP. It is algorithmically generated and may contain inaccuracies or biases.*

                *2. The RAG mode is leveraging Google's gemma3 12b instruct model. Information about the model can be found [here](https://ai.google.dev/gemma/docs/core)*

                *3. All information provided to the Chatbot is processed offline as well as all answers generated by the chatbot. [Ollama](https://ollama.com/) is providing 
                         the background service.*            
            """,
            link_target="_blank",
            ),
            footer_image

        ]
    )
    return page


# Run the app
if __name__ == '__main__':
    # app.run(port=8888, debug=True)
    None