import logging

# from transformers import PreTrainedModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores.base import VectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser


logger = logging.getLogger(__name__)
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO)

# Prompts
GEN_PROMPT = PromptTemplate(
    template=
    """
    You are a helpful assistant given access to product-related context. 
    Your task is to provide a detailed and informative response to customer 
    inquiries about a specific product, based on the information from the 
    retrieved context.
    Here is the context: {context}.
    Here is the question: {question}
    Your answer must be accurate and fact-based.    
    Please refer to the relevant product context and answer the query in a 
    clear and helpful manner, providing as much relevant detail as possible 
    based on the retrieved context. If the query is about specific 
    features, pricing, usage, or troubleshooting, include that in your response.
    """,
    input_variables = ["context", "question"],
)

RETRIEVAL_PROMPT = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
        )

HALLUCINATION_PROMPT = PromptTemplate(
    template=
    """
    You have access to a set of product-related documents. 
    Here are the documents:
    \n ------- \n
    {documents}
    \n ------- \n
    Here is the answer: {generation} \n
    Your task is to review the generated answer and verify 
    that the information is accurate, fact-based, and supported by the retrieved documents.
    Instructions:
    1. Verification: Review the answer carefully and ensure that all information is directly supported by the retrieved documents. 
    2. Check for Hallucinations: If any information is included in the generated content that is not found in the documents, flag it as a hallucination. 
    3. No Inventions: Do not create new information. If the document does not contain a piece of information needed to answer the query or generate content, explicitly state that the information is not available.
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation
    """,
    input_variables=["documents", "generation"],
)


ANSWER_PROMPT = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)

# JSON addon: Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

REWRITE_PROMPT = PromptTemplate(
    template="""
    You a question re-writer that converts an input question to a better version that is optimized \n 
    for vectorstore retrieval. Look at the initial and formulate an improved question. \n
    Here is the initial question: \n\n {question}. Improved question with no preamble: \n 
    """,
    input_variables=["question"],
)

class RAGComponents:

    def __init__(self, llm: BaseChatModel, vectorstore: VectorStore):
        self.llm = llm
        self.retriever = vectorstore.as_retriever()

    # Grader
    def retrieval_grading(self):

        retrieval_grader = RETRIEVAL_PROMPT | self.llm | JsonOutputParser()
        return retrieval_grader


    def answer_generator(self):

        # Chain
        rag_chain = GEN_PROMPT | self.llm | StrOutputParser()
        return rag_chain


    def hallucination_grading(self):

        hallucination_grader = HALLUCINATION_PROMPT | self.llm | JsonOutputParser()
        return hallucination_grader


    def answer_grading(self):
        
        answer_grader = ANSWER_PROMPT | self.llm | JsonOutputParser()
        return answer_grader

    def question_rewriting(self):

        # Prompt
        question_rewriter = REWRITE_PROMPT | self.llm | StrOutputParser()
        return question_rewriter

    def get_rag_components(self):
        return {
            "retrieve": self.retriever,
            "document_grader": self.retrieval_grading(),
            "generator": self.answer_generator(),
            "rewriter": self.question_rewriting(),
            "hallucination_grader": self.hallucination_grading(),
            "answer_grader": self.answer_grading(),
        }


if __name__ == "__main__":
    # Get from SETTINGS
    model_checkpoint = "meta-llama/Llama-3.2-3B"  # TODO: Get checkpoint/model from interface class
    device = "cpu"

    # rag_components = get_rag_components(model_checkpoint=model_checkpoint, device="cpu", backend="openvino")


    # rag = RAGComponents(llm, vectorstore)
    # rag_comps = rag.get_rag_components()
