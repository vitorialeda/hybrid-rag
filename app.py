import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup
load_dotenv()

model = ChatOpenAI(
    model="llama3.3-70b-instruct",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://inference.do-ai.run/v1",
    temperature=0.7,
    max_tokens=1024,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 2. Indexação — Carregar e dividir documentos
loader = PyPDFDirectoryLoader("./docs/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=120,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
print(f"Split doc into {len(all_splits)} sub-documents.")

# 3. Retriever Vetorial (Dense / Semântico)
vector_store = Chroma(
    collection_name="hybrid_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
vector_store.add_documents(documents=all_splits)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 4. Retriever BM25 (Sparse / Lexical)
bm25_retriever = BM25Retriever.from_documents(all_splits, k=5)

# 5. Hybrid Retriever (Ensemble com RRF)
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],
)

# 6. RAG Chain (Prompt + LLM)
prompt = ChatPromptTemplate.from_template(
    """Você é um assistente útil. Use o contexto abaixo para responder a pergunta.
Se não souber a resposta com base no contexto, diga que não sabe.

Contexto:
{context}

Pergunta: {question}

Resposta:"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 7. Teste
query = "Qual é a universidade do curso ofertado?"

print(f"\nQuery: {query}\n")
for chunk in rag_chain.stream(query):
    print(chunk, end="", flush=True)
print()