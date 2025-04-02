
from src.rag.rag_graph import SelfRAG
from src.rag.generate_vectorstore import get_vectorstore
from src.rag.llm.llm_model import get_model
from langchain_ollama import ChatOllama
from langchain_core.vectorstores import VectorStore

# TODO: Send message, when recursion limit is reached

def init_rag(model: str, device: str, backend: str = None, vector_store: VectorStore = None):

    if not isinstance(vector_store, VectorStore):
        vectorstore = get_vectorstore()
    # llm = get_model(model_checkpoint=model, device=device, backend=backend)
    llm = ChatOllama(model=model, device=device, temperature=0)
    self_rag = SelfRAG(llm, vectorstore)
    return self_rag


def prompt_rag(rag: SelfRAG, prompt: str):
    return rag.invoke(prompt)

    
# def change_rag(model: str, device: str):
    

if __name__ == "__main__":
    # model = "gemma3:12b-it-q4_K_M"
    model = "mistral"
    # model = "meta-llama/Llama-3.2-3B-Instruct"
    device = "cpu"
    backend = None
    self_rag = init_rag(model, device, backend=backend)
    ans = prompt_rag(self_rag, "What is a workstation?")
    print(ans)
    # self_rag.invoke("What is a workstation?")
