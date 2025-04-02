# Import packages
import dash
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from src.dashboard.components.title_section import create_title_section


# import pymupdf
# import pymupdf4llm
# import fitz

# # For PDF mining
# import PyPDF2
# import re
# import base64
# import os
# import io

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document


# For processing all PDFs
from src.utils.PDFprocessor02_Marisa import PDFprocessor
# from src.dashboard.components.chatbot_button import get_chatbot
# from src.rag.generate_vectorstore import get_vectorstore, add_document

# -----------------------
# Initialize the page
# app = dash.get_app()
dash.register_page(__name__, title="Chatbot", name="Chatbot", path='/chatbot')


# Initialize the PDFprocessor
processor = PDFprocessor()

'''
title_section = html.Div([
    # HP logo
    html.Img(src='/assets/0_HP_logo.png', style={'width': '80px', 
                                                 'height': 'auto', 
                                                 'margin-right': '20px'}),  
    # Title
    html.Div([
        html.H1('Local AI-powered HP workstation assistant',
                style={'color': '#004ad8', 'margin': '5px', 'font-weight': 'bold', 'font-size': '45px'}),
        html.H3('Get instant answers on HP Workstations from your local database or uploaded PDFs',
                style={'margin-top': '5px', 'font-size': '35px'}) 
    ]),

    # Styling the flex box
    ], style={'display': 'flex', ## keep otherwise will stack logo on text
          'align-items': 'center', 
          'justify-content': 'flex-start', 
          'margin-bottom': '20px',
          'padding-left': '5px',
          'padding-top': '5px' })

'''

description_section = html.Div([
        dcc.Markdown("""
            ##### Getting started
            * Ask questions about HP workstations and get insights on specs, and recommendations.

            ##### How to use the chatbot
            * Click the chat icon in the bottom right corner to start a conversation. 
              This AI assistant is designed specifically for HP users, retrieving information from preloaded HP documents to provide precise answers tailored to HP products and services.
            
            ##### Why run locally?
            * Unlike other AI tools running in the cloud, this chatbot operates locally on your device. 
              It doesn’t require an internet connection to function and ensures complete privacy by keeping all interactions and data processing on your machine.
            
            ##### Why upgrade to an NPU? DO WE KEEP THIS PART_
            Have a look at the resource monitor and see the benefits:
            * Faster AI performance than CPUs for real-time responses.
            * Lower power consumption than GPUs for longer battery life.
            * Seamless multitasking without slowdowns.           
            """)
    ],
            style={
            'padding-left': '50px',
            'padding-top': '10px' })

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

# collapse_state_store = dcc.Store(id="collapse-state", data={"is_open": False})
'''
collapse = html.Div([
        dbc.Button(
            "More use cases",
            id="collapse-button",
            className="mb-3",
            style={"backgroundColor": "#004ad8", "borderColor": "#004ad8", "color": "white"},
            n_clicks=0,
        ),
        dbc.Collapse(
            dbc.Card(dbc.CardBody(use_cases_box)),
            id="collapse",
            is_open=False,
        )],
        style={
            'padding-left': '50px',
            'padding-top': '10px' }
)

@callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open
'''


fade = html.Div(
    [
        dbc.Button("More use cases", id="fade-button", className="mb-3", n_clicks=0,  style={"backgroundColor": "#004ad8", "borderColor": "#004ad8", "color": "white"}),
        dbc.Fade(
            dbc.Card(
                dbc.CardBody(
                    use_cases_box
                )
            ),
            id="fade",
            is_in=False,
            appear=False,
        ),
    ]
)


@callback(
    Output("fade", "is_in"),
    [Input("fade-button", "n_clicks")],
    [State("fade", "is_in")],
)
def toggle_fade(n, is_in):
    if not n:
        # Button has never been clicked
        return False
    return not is_in



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
            #collapse,
            fade,
            # html.Hr(),
            # chatbot_layout,
            # dcc.Store(id='vectorstore'),
            # dcc.Store(id="store-conversation", data=[]),
            #chatbot_layout,
            # html.Hr(),
            # pdf_uploader,
            footer_image

        ]
    )
    return page


# def process_pdf(content, filename):

#     if not filename.lower().endswith(".pdf"):
#         return "Please upload a PDF file."

#     _, content_string = content.split(",")
#     pdf_bytes = base64.b64decode(content_string)
#     pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
#     raw_text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
#     data = Document(page_content=raw_text, metadata=pdf_reader.metadata)
#     return data

# ### ADD CONTENT OF UPLOADED PDF
# def add_new_content(content, filename):
#     # Check if PDF
#     vector_store = get_vectorstore()

#     if not filename.lower().endswith(".pdf"):
#         return html.Div(["Please upload a PDF file."])

#     try:
#         # Extract base64-encoded string and decode it 
#         _, content_string = content.split(",")
#         pdf_bytes = base64.b64decode(content_string)

#         pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
#         raw_text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
#         data = Document(page_content=raw_text, metadata=pdf_reader.metadata)
#         add_document(vector_store, data)
#         vector_store.similarity_search("Prompt engineering")
#         # 1. Extract text

#         ###### Using the PDFprocessor methods:

#         # # 2. Clean the text and get chunks
#         # cleaned_text = processor.clean_text(raw_text)
#         # processor.text = cleaned_text
#         # processor.get_text_chunks()

#         # # 3. Create vector store
#         # processor.get_vectorstore(api_key = api_key)  # To be changed when we decide the embeddings, in the meantime, set api_key in the .env

#         return html.Div([
#             html.H5(filename),
#             html.P("Your file uploaded and processed successfully."),
#             html.Hr()
#             #print(new_content) # Just to check that it's processed
#         ])

#     except Exception as e:
#         return html.Div([f"Error processing the PDF file."])

# # Callbacks-------------------

# @callback(Output('output-data-upload', 'children'),
#               Input('upload-pdf', 'contents'),
#               State('upload-pdf', 'filename')
# )
# def update_output(list_of_contents, list_of_names):
#     if list_of_contents is not None:
#         children = [
#             add_new_content(conts, names) for conts, names in
#             zip(list_of_contents, list_of_names)]
        
#         children = [
#             process_pdf(conts, names) for conts, names in
#             zip(list_of_contents, list_of_names)
#         ]

#         return children



# Save for RAG


# Run the app
if __name__ == '__main__':
    # app.run(port=8888, debug=True)
    None