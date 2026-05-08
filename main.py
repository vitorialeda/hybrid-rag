import os

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langsmith import traceable

from rag_settings import (
    build_callback_config,
    build_embeddings,
    build_llm,
    build_ragas_llm,
    configure_environment,
    extract_response_text,
    finish_usage_tracker,
    get_chroma_settings,
    get_int_env,
    run_ragas,
    salvar,
    start_usage_tracker,
)

configure_environment("benchmark-hybrid-rag")

DOCS_DIR = os.getenv("DOCS_DIR", "../docs/")
PERSIST_DIR, CHROMA_COLLECTION_NAME = get_chroma_settings(
    "./chroma_hybrid_db_openai",
    "hybrid_collection_openai",
)
RETRIEVER_K = get_int_env("RETRIEVER_K", 3)

test_queries = [
    # FÁCEIS
    "O que significa ‘lógica de programação’ em palavras simples?",
    "De um jeito bem direto: o que é um algoritmo?",
    "Qual é a diferença entre constante e variável?",
    "Pra que serve o comando ‘leia’ em um algoritmo?",

    # MÉDIAS
    "O que é um comando de atribuição e por que o tipo do dado precisa ser compatível com o tipo da variável?",
    "O que são operadores aritméticos (como +, -, * e /) e pra que eles servem?",
    "Pra que servem os operadores relacionais numa expressão?",

    # DIFÍCEIS
    "O que é uma ‘expressão lógica’?",
    "Em uma repetição, o que é um contador e como ele é incrementado?",
    "Como funciona a repetição ‘repita ... até’ e o que ela garante sobre a execução do bloco?"
]


ground_truths = [
    # FÁCEIS
    "Lógica de programação é o uso correto das leis do pensamento, da ‘ordem da razão’ e de processos formais de raciocínio e simbolização na programação de computadores, com o objetivo de produzir soluções logicamente válidas e coerentes para resolver problemas.",
    "Um algoritmo é uma sequência de passos bem definidos que têm por objetivo solucionar um determinado problema.",
    "Um dado é constante quando não sofre variação durante a execução do algoritmo: seu valor permanece constante do início ao fim (e também em execuções diferentes ao longo do tempo). Já um dado é variável quando pode ser alterado em algum instante durante a execução do algoritmo, ou quando seu valor depende da execução em um certo momento ou circunstância.",
    "O comando de entrada de dados ‘leia’ é usado para que o algoritmo receba os dados de que precisa: ele tem a finalidade de atribuir o dado fornecido à variável identificada, seguindo a sintaxe leia(identificador) (por exemplo, leia(X) ou leia(A, XPTO, NOTA)).",

    # MÉDIAS
    "Um comando de atribuição permite fornecer um valor a uma variável. O tipo do dado atribuído deve ser compatível com o tipo da variável: por exemplo, só se pode atribuir um valor lógico a uma variável declarada como do tipo lógico.",
    "Operadores aritméticos são o conjunto de símbolos que representam as operações básicas da matemática (por exemplo: + para adição, - para subtração, * para multiplicação e / para divisão). Para potenciação e radiciação, o livro indica o uso das palavras-chave pot e rad.",
    "Operadores relacionais são usados para realizar comparações entre dois valores de mesmo tipo primitivo. Esses valores podem ser constantes, variáveis ou expressões aritméticas, e esses operadores são comuns na construção de equações.",

    # DIFÍCEIS
    "Uma expressão lógica é aquela cujos operadores são lógicos ou relacionais e cujos operandos são relações, variáveis ou constantes do tipo lógico.",
    "Um contador é um modo de contagem feito com a ajuda de uma variável com um valor inicial, que é incrementada a cada repetição. Incrementar significa somar um valor constante (normalmente 1) a cada repetição.",
    "A estrutura de repetição ‘repita ... até’ permite que um bloco (ou ação primitiva) seja repetido até que uma determinada condição seja verdadeira. Pela sintaxe da estrutura, o bloco é executado pelo menos uma vez, independentemente da validade inicial da condição."
]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_hybrid_retriever():
    embeddings = build_embeddings()

    loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split doc into {len(all_splits)} sub-documents.")

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    if vector_store._collection.count() == 0:
        batch_size = 500
        print(f"Adicionando {len(all_splits)} documentos ao Chroma em batches...")

        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i + batch_size]
            vector_store.add_documents(documents=batch)
            print(f"  {min(i + batch_size, len(all_splits))}/{len(all_splits)} chunks adicionados")

        print("Ingestão concluída!")
    else:
        print(
            "Coleção existente encontrada com "
            f"{vector_store._collection.count()} documentos. Pulando ingestão."
        )

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    bm25_retriever = BM25Retriever.from_documents(all_splits, k=RETRIEVER_K)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )

    return hybrid_retriever, embeddings, vector_store


@traceable(name="hybrid-rag-query", run_type="chain")
def hybrid_rag(query, retriever, llm, callbacks=None):
    context_docs = retriever.invoke(query)
    contexts = [doc.page_content for doc in context_docs]
    context = format_docs(context_docs)

    prompt = f"""Você é um assistente útil. Use o contexto abaixo para responder a pergunta.
Se não souber a resposta com base no contexto, diga que não sabe.

Contexto:
{context}

Pergunta:
{query}

Resposta:"""

    answer = extract_response_text(
        llm.invoke(prompt, config=build_callback_config(callbacks))
    )
    return answer, contexts


def main():
    hybrid_retriever, embeddings, vector_store = build_hybrid_retriever()

    print(f"Vectorstore pronto: {vector_store._collection.count()} chunks indexados.")

    for run in range(5):
        print(f"\n=== RODADA {run + 1}/5 ===")
        answer_llm = build_llm()
        eval_llm = build_ragas_llm()

        print("Coletando respostas para avaliacao RAGAS...")
        ragas_data = []

        for i, query in enumerate(test_queries):
            print(f"  [{i + 1}/{len(test_queries)}] {query}")
            tracker, started_at = start_usage_tracker()
            answer, contexts = hybrid_rag(
                query,
                hybrid_retriever,
                answer_llm,
                callbacks=[tracker],
            )
            ragas_item = {
                "question": query,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truths[i]
            }
            ragas_item.update(finish_usage_tracker(tracker, started_at))
            ragas_data.append(ragas_item)

        df_resultado = run_ragas(ragas_data, eval_llm, embeddings)
        salvar(df_resultado, nome_base=f"hybrid-rag-run-{run + 1}")


if __name__ == "__main__":
    main()

