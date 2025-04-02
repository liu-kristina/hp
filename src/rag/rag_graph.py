from pathlib import Path
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
from pprint import pprint


from typing import List
from typing_extensions import TypedDict

from src.rag.rag_components import RAGComponents
# from src.llm.llm_model import get_model
from src.rag.generate_vectorstore import get_vectorstore
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.graph.graph import CompiledGraph


load_dotenv(".env")
# TODO: Implement prompt LLM method for App


# Graph state
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

class RAGNodes:

    def __init__(self, llm, rag_components):
        self.llm = llm
        self.rag: RAGComponents = rag_components

    # Nodes
    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.rag.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.rag.answer_generator().invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.rag.retrieval_grading().invoke(
                {"question": question, "document": d.page_content}
            )
            if "score" in score.keys():
                grade = score["score"]
            else:
                grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}


    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.rag.question_rewriting().invoke({"question": question})
        return {"documents": documents, "question": better_question}


    # Edges
    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"


    def grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.rag.hallucination_grading().invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.rag.answer_grading().invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

def construct_rag(rag_nodes):
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", rag_nodes.retrieve)  # retrieve
    workflow.add_node("grade_documents", rag_nodes.grade_documents)  # grade documents
    workflow.add_node("generate", rag_nodes.generate)  # generatae
    workflow.add_node("transform_query", rag_nodes.transform_query)  # transform_query

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        rag_nodes.decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        rag_nodes.grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    # Compile
    rag = workflow.compile()
    return rag


@dataclass
class SelfRAG:

    llm: BaseChatModel
    vectorstore: VectorStore
    components: RAGComponents = field(init=False)
    nodes: RAGNodes = field(init=False)
    rag: CompiledGraph = field(init=False)

    def __post_init__(self):
        # self.components = RAGComponents(self.llm, self.vectorstore)
        self.components = RAGComponents(self.llm, self.vectorstore)
        self.nodes = RAGNodes(self.llm, self.components)
        self.rag = construct_rag(self.nodes)


    def invoke(self, prompt: str) -> str:
         # Run
        inputs = {"question": prompt}
        for output in self.rag.stream(inputs):
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint("\n---\n")

        # Final generation
        return value["generation"]


if __name__ == "__main__":
    # model_checkpoint = "meta-llama/Llama-3.2-3B-Instruct" 
    
    vectorstore = get_vectorstore()
    # llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOllama(model="gemma3:12b-it-q4_K_M", temperature=0)

    rag = RAGComponents(llm, vectorstore)
    rag_nodes = RAGNodes(llm, rag)
    self_rag = construct_rag(rag_nodes)

    # # Run
    inputs = {"question": "Explain the specifications of a HP workstations"}
    for output in self_rag.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value["generation"])

    # llm = get_model(model_checkpoint, device=device, backend=backend)

    # Testing playground
    retriever = rag.retriever
    retrieval_grader = rag.retrieval_grading()
    question = "Mini Workstation"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    ans = retrieval_grader.invoke({"question": question, "document": doc_txt})
    print(ans)

    # Test hallucination
    answer_gen = rag.answer_generator()
    question = "What is a HP mini Workstation?"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    answer = answer_gen.invoke({"question": question, "context": docs})

    halluci_grader = rag.hallucination_grading()
    ans = halluci_grader.invoke({"documents": docs, "generation": answer})
    print(ans)
    print("finished")
    # Final generation
    # pprint(value["generation"])

    # vectorstore = get_vectorstore()
    # # llm = get_model(model_checkpoint, device=device, backend=backend)

    # llm = ChatOllama(model="llama3.2", device="cpu")
    # rag = RAGComponents(llm, vectorstore)
    # rag_components = rag.get_rag_components()

    # # rag_components = get_rag_components(model_checkpoint=model_checkpoint, device="cpu", backend="openvino")

    # # Testing playground
    # retriever = rag.retriever
    # retrieval_grader = rag.retrieval_grading()
    # question = "Mini Workstation"
    # docs = retriever.invoke(question)
    # doc_txt = docs[1].page_content
    # ans = retrieval_grader.invoke({"question": question, "document": doc_txt})
    # print(ans)
