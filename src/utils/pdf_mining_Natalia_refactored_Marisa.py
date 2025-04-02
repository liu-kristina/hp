# To read the PDFs
import os
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader

# To work on the PDF content
import re
import unicodedata
from langchain.text_splitter import CharacterTextSplitter

#Embeddings and vectorstore
# from langchain.embeddings import HuggingFaceInstructEmbeddings --> Not used here
# from langchain.embeddings import OpenAIEmbeddings # --> Using this as per Natalia's code, I'm not including the API key as this is not what we will be using I think
from langchain_openai import OpenAIEmbeddings # The previous line was deprecated
from langchain.vectorstores import FAISS # --> has the add_document method so I can keep using this one, need to install: pip install faiss-cpu
import time

# Not sure if this should be part of the class or not. Move it if needed
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class PDFprocessor:
    def __init__(self, pdf_folder=None, faiss_path="faiss_index"):
        self.pdf_folder = pdf_folder
        self.pdf_docs = self.load_pdfs() if pdf_folder else [] # Need an empty list so that can still process the uploaded PDF even if no other PDFs
        self.text = ""
        self.chunks = []
        self.faiss_path = faiss_path
        self.vector_store = None


    # 0. Get all PDFs from 1 folder
    def load_pdfs(self):
        return [os.path.join(self.pdf_folder, f) for f in os.listdir(self.pdf_folder) if f.endswith(".pdf")]

    # 1. Get text from ALL PDFs in a folder
    def get_pdf_text(self):
        text = ""

        for pdf in self.pdf_docs:
            pdf_reader = PdfReader(pdf)
            # print(f"Processing PDF: {pdf}") # REMOVE: just to check if this is done correctly

            for page in pdf_reader.pages:
                text += page.extract_text()
        # print(f"TEST:{text[:200]}") # REMOVE: Checking this part is working
        return text

    # 2. Clean extracted text (moved up from Natalia's code)
    def clean_text(self, text):
        # print(f"TEST:{text[:200]}") # REMOVE: Checking this part is working

        text = unicodedata.normalize("NFKD", text)  # Fix encoding
        text = re.sub(r"[^a-zA-Z0-9.,!?%€$-]", " ", text)  # Remove unwanted chars
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        text = re.sub(r"\.{5,}", " ", text) # Remove sequences of 5 or more dots

        words = text.split()
        clean_text = " ".join(word for word in words if word not in stop_words)

        print(f"TEST clean text:{clean_text[:200]}") # REMOVE: Checking this part is working

        return clean_text

    # 3. Split in chunks
    def get_text_chunks(self):

        print(f"text length: {len(self.text)}")

        text_splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.chunks = text_splitter.split_text(self.text)

        print(f"TEST: Total Chunks = {len(self.chunks)}") 
        return self.chunks

    # 4. Create vector store --> using embeddings to transform words into numbers (similar to word2vec)

    # OPTION 1: We already have a vector store
    def load_vectorstore(self, api_key):
        if os.path.exists(self.faiss_path):
            print(" Retrieving the existing vector store")
            return FAISS.load_local(self.faiss_path,    # --> will be stored in self.vector_store at the end
                                    OpenAIEmbeddings(openai_api_key = api_key),
                                    allow_dangerous_deserialization=True) # --> I got some warning regarding Pickle files that can deliver malicious load. Be careful to only use this locally!!
        else:
            return None 
    
    def get_vectorstore(self, api_key, embedding_model='OpenAI'): #Using OpenAI as in Natalia's code, we can change it for something else when we can make a decision
        start_time = time.time()

        embeddings = OpenAIEmbeddings(openai_api_key = api_key) # Load embedding we will use fir that model we are using

        print(f"TEST: Total Chunks: {len(self.chunks)}")  
        print(f"TEST: First Chunk: {self.chunks[0]}")  

        # OPTION 2: We need to create the vector store from scratch

        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(texts=self.chunks, embedding=embeddings) # puts the text embeddings together
            print("New vector store created")

        # OPTION 3: The vector store exists, we need to add the new docs to it
        else:
            self.vector_store.add_documents(self.chunks, embeddings)
            print("New documents added to existing vector store")

        self.vector_store.save_local(self.faiss_path) #--> Saves it at the root, not in the git repo

        end_time = time.time()
        print(f"Vector store created/updated in {end_time - start_time:.2f} seconds")

        return self.vector_store


    #5. Process PDFs will all steps above
    def pdf_mining(self, api_key):

        # 1. Get text from all PDFs in folder
        raw_text = self.get_pdf_text()

         #2. Clean extracted text
        self.text = self.clean_text(raw_text)

         #3. Split in chunks
        self.get_text_chunks()

        #4. Create vector store
        self.vector_store = self.load_vectorstore(api_key) or self.get_vectorstore(api_key) # Our 3 options


if __name__ == "__main__":
    # folder_path = "/Users/marisadavis/Desktop/Constructor_Academy/HP_PROJECT/sample_pdfs" # CHANGE FOR relative path on laptops
    # pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]

    processor = PDFprocessor(folder_path)
    processor.pdf_mining(api_key)
    print("Something like 'It's all done'")
