from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="Epstein RAG")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Request Model
# -----------------------------
class Query(BaseModel):
    question: str


# -----------------------------
# Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Vector DB
# -----------------------------
db = Chroma(
    collection_name="epstein",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 12,
        "fetch_k": 60
    }
)

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=500
)

# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a retrieval-based assistant. "
        "Answer ONLY using the provided context. "
        "If the answer is not present, say: "
        "'I could not find this information in the documents.' "
        "Limit the answer to 5-6 lines."
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])


# -----------------------------
# Endpoint
# -----------------------------
@app.post("/ask")
async def ask(payload: Query):

    question = payload.question

    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "I could not find this information in the documents.",
            "sources": []
        }

    context = "\n\n".join(d.page_content for d in docs)

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    return {
        "answer": response.content.strip(),
        "sources": [
            {
                "source": d.metadata.get("source"),
                "chunk": d.metadata.get("chunk")
            }
            for d in docs
        ]
    }