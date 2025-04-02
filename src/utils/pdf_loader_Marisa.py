# Import packages
from dash import Dash, html, dcc, Input, Output, State

# For PDF mining
import PyPDF2
import re
import base64
import os
import io

# For processing all PDFs
from pdf_mining_Natalia_refactored_Marisa import PDFprocessor

# -----------------------

# Initialize the app
app = Dash()

# Initialize the PDFprocessor
processor = PDFprocessor()

# App layout
app.layout = html.Div([
    html.H1(children = "Marisa's PDF loader"),

    # Upload the file:
    dcc.Upload(
        id='upload-pdf',
        children=html.Div(['Drag and drop or ',
            html.A('click to upload a PDF file')
        ]),
        style={ # --> To be added to a CSS style sheet and have fun with it
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
        multiple=True # Allow multiple files to be uploaded
       ),
    html.Div(id='output-data-upload'),
])

### PDF TO TEXT function --> try changing for the class here

### ADD CONTENT OF UPLOADED PDF

def add_new_content(content, filename):
    # Check if PDF
    if not filename.lower().endswith(".pdf"):
        return html.Div(["Please upload a PDF file."])

    try:
        # Extract base64-encoded string and decode it 
        _, content_string = content.split(",")
        pdf_bytes = base64.b64decode(content_string)

        # 1. Extract text
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        raw_text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

        ###### Using the PDFprocessor methods:

        # 2. Clean the text and get chunks
        cleaned_text = processor.clean_text(raw_text)
        processor.text = cleaned_text
        processor.get_text_chunks()

        # 3. Create vector store
        processor.get_vectorstore(api_key = api_key)  # To be changed when we decide the embeddings, in the meantime, set api_key in the .env

        return html.Div([
            html.H5(filename),
            'uploaded',
            html.Hr(), # horizontal line
            #print(new_content) # Just to check that it's processed
        
        ])

    except Exception as e:
        return html.Div([f"Error processing the PDF file."])

# Callbacks-------------------

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-pdf', 'contents'),
              State('upload-pdf', 'filename')
)
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            add_new_content(conts, names) for conts, names in
            zip(list_of_contents, list_of_names)]
        
        return children


# Save for RAG


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
