import os
from PDFprocessor02_Marisa import PDFprocessor

folder_path = "/Users/marisadavis/Desktop/Constructor_Academy/HP_PROJECT/sample_pdfs"

api_key = "id_ed25519.pub"

embedding_model = 'Ollama'

# Initialize processor
processor = PDFprocessor(folder_path)

# Run processing
processor.pdf_mining(api_key, embedding_model)