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
from langchain_community.vectorstores import FAISS # need to install: pip install faiss-cpu
import time

#Set up environment for api keys:
 # TO DO: os.getenv("MISTRALAI_API_KEY")

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
    def load_vectorstore(self, api_key, embedding_model):
        folder_path = os.path.join(self.faiss_path, embedding_model)

        if os.path.exists(folder_path):
            print(f"Retrieving the existing vector store from {folder_path}")

            ## Getting the correct embeddings for each model:
            if embedding_model == 'Ollama': # Check name is same as in the injection parameter sheet

                        from langchain_ollama import OllamaEmbeddings # make sure the packages are installed before using
                        embeddings = OllamaEmbeddings(model="llama3")

            elif embedding_model == 'Mistral': # Check name is same as in the injection parameter sheet

                        from langchain_mistralai import MistralAIEmbeddings # make sure the packages are installed before using
                        embeddings = MistralAIEmbeddings(model="mistral-embed")

            else: print(f"Error in retrieving the vector store from {folder_path}")


            return FAISS.load_local(folder_path,    # --> will be stored in self.vector_store at the end
                                        embeddings,
                                        allow_dangerous_deserialization=True) # --> I got some warning regarding Pickle files that can deliver malicious load. Be careful to only use this locally!!
        else:
            return None 
    

    def get_vectorstore(self, api_key, embedding_model): #Using OpenAI as in Natalia's code, we can change it for something else when we can make a decision
        start_time = time.time()

        # OPTION 1.a) User selects Ollama model (light)
        if embedding_model == 'Ollama': # Check name is same as in the injection parameter sheet

            from langchain_ollama import OllamaEmbeddings # make sure the packages are installed before using
            embeddings = OllamaEmbeddings(model="llama3")

        #print(f"TEST: Total Chunks: {len(self.chunks)}")  
        #print(f"TEST: First Chunk: {self.chunks[0]}")  

        # OPTION 1.b) User selects Mistral model (heavy)
        elif embedding_model == 'Mistral': # Check name is same as in the injection parameter sheet

            from langchain_mistralai import MistralAIEmbeddings # make sure the packages are installed before using
            embeddings = MistralAIEmbeddings(model="mistral-embed")

        else: print(f"Error in creating {embedding_model} embeddings")

        folder_path = os.path.join(self.faiss_path, embedding_model) # Creating the folder for that model

        # OPTION 2: We need to create the vector store from scratch

        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(texts=self.chunks, embedding=embeddings) # puts the text embeddings together
            print("New vector store created")

        # OPTION 3: The vector store exists, we need to add the new docs to it
        else:
            self.vector_store.add_documents(self.chunks, embeddings)
            print("New documents added to existing vector store")

        # Save vector store
        self.vector_store.save_local(folder_path)

        end_time = time.time()
        print(f"Vector store created/updated in {end_time - start_time:.2f} seconds at {folder_path}")

        return self.vector_store


    #5. Process PDFs will all steps above
    def pdf_mining(self, api_key, embedding_model):

        # 1. Get text from all PDFs in folder
        raw_text = self.get_pdf_text()

         #2. Clean extracted text
        self.text = self.clean_text(raw_text)

         #3. Split in chunks
        self.get_text_chunks()

        #4. Create vector store
        self.vector_store = self.load_vectorstore(api_key, embedding_model) or self.get_vectorstore(api_key, embedding_model) # Our 3 options


if __name__ == "__main__":
    # folder_path = "/Users/marisadavis/Desktop/Constructor_Academy/HP_PROJECT/sample_pdfs" # CHANGE FOR relative path on laptops
    # pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]

    processor = PDFprocessor(folder_path)
    processor.pdf_mining(api_key, embedding_model)
    print("PDF processing has been completed")
