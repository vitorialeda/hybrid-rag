import os
import time
from itertools import count

from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import RunConfig, evaluate
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

USAGE_COLS = [
    "answer_response_time_seconds",
    "answer_input_tokens",
    "answer_output_tokens",
    "answer_total_tokens",
]

EXPORT_COLS = ["question", *METRIC_COLS, *USAGE_COLS]

DEFAULT_RAGAS_TIMEOUT_SECONDS = 600
DEFAULT_RAGAS_MAX_WORKERS = 4


def get_int_env(name, default, minimum=1):
    value = os.getenv(name)

    if value is None:
        return default

    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} precisa ser um numero inteiro. Valor atual: {value}"
        ) from exc

    if parsed_value < minimum:
        raise RuntimeError(
            f"{name} precisa ser maior ou igual a {minimum}. "
            f"Valor atual: {parsed_value}"
        )

    return parsed_value


def configure_environment(project_name):
    load_dotenv()

    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
    os.environ["LANGSMITH_ENDPOINT"] = os.getenv(
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com",
    )
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", project_name)


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY nao encontrada. Crie um arquivo .env a partir do "
            ".env.example e informe sua chave da OpenAI."
        )

    return api_key


def get_chroma_settings(default_persist_dir, default_collection_name):
    return (
        os.getenv("CHROMA_PERSIST_DIR", default_persist_dir),
        os.getenv("CHROMA_COLLECTION_NAME", default_collection_name),
    )


def build_embeddings():
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        api_key=get_openai_api_key(),
    )


def build_llm():
    return ChatOpenAI(
        api_key=get_openai_api_key(),
        model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
        reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "medium"),
        temperature=None,
        use_responses_api=True,
    )


def build_ragas_llm():
    return LangchainLLMWrapper(
        build_llm(),
        bypass_n=True,
        bypass_temperature=True,
    )


def extract_response_text(response):
    text = getattr(response, "text", None)

    if callable(text):
        text = text()

    if text:
        return text

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                parts.append(block.get("text", ""))

        return "\n".join(part for part in parts if part)

    return str(content)


def _empty_token_usage():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def _normalizar_token_usage(usage):
    tokens = _empty_token_usage()

    if not isinstance(usage, dict):
        return tokens

    tokens["input_tokens"] = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("prompt_token_count")
        or 0
    )
    tokens["output_tokens"] = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("completion_token_count")
        or 0
    )
    tokens["total_tokens"] = (
        usage.get("total_tokens")
        or usage.get("total_token_count")
        or tokens["input_tokens"] + tokens["output_tokens"]
    )

    return tokens


def _somar_token_usage(total, usage):
    total["input_tokens"] += usage.get("input_tokens", 0) or 0
    total["output_tokens"] += usage.get("output_tokens", 0) or 0
    total["total_tokens"] += usage.get("total_tokens", 0) or 0
    return total


def extract_token_usage(response):
    usage = getattr(response, "usage_metadata", None)

    if usage:
        return _normalizar_token_usage(usage)

    response_metadata = getattr(response, "response_metadata", None) or {}

    for key in ("token_usage", "usage"):
        if response_metadata.get(key):
            return _normalizar_token_usage(response_metadata[key])

    return _normalizar_token_usage(response_metadata)


def extract_llm_result_token_usage(result):
    llm_output = getattr(result, "llm_output", None) or {}

    for key in ("token_usage", "usage"):
        if isinstance(llm_output, dict) and llm_output.get(key):
            return _normalizar_token_usage(llm_output[key])

    usage = _normalizar_token_usage(llm_output)
    if usage["total_tokens"]:
        return usage

    total = _empty_token_usage()

    for generations in getattr(result, "generations", []) or []:
        for generation in generations:
            message = getattr(generation, "message", None)

            if message is not None:
                _somar_token_usage(total, extract_token_usage(message))

            generation_info = getattr(generation, "generation_info", None) or {}
            for key in ("token_usage", "usage"):
                if generation_info.get(key):
                    _somar_token_usage(total, _normalizar_token_usage(generation_info[key]))

    return total


class TokenUsageTracker(BaseCallbackHandler):
    def __init__(self):
        self._tokens = _empty_token_usage()

    def on_llm_end(self, response, **kwargs):
        _somar_token_usage(self._tokens, extract_llm_result_token_usage(response))

    @property
    def input_tokens(self):
        return self._tokens["input_tokens"]

    @property
    def output_tokens(self):
        return self._tokens["output_tokens"]

    @property
    def total_tokens(self):
        return self._tokens["total_tokens"]


def build_callback_config(callbacks):
    return {"callbacks": callbacks} if callbacks else None


def start_usage_tracker():
    return TokenUsageTracker(), time.perf_counter()


def finish_usage_tracker(tracker, started_at):
    return {
        "answer_response_time_seconds": round(time.perf_counter() - started_at, 6),
        "answer_input_tokens": tracker.input_tokens,
        "answer_output_tokens": tracker.output_tokens,
        "answer_total_tokens": tracker.total_tokens,
    }


def anexar_metricas_execucao(df, ragas_data):
    usage_by_question = {
        item["question"]: {col: item.get(col, 0) for col in USAGE_COLS}
        for item in ragas_data
    }

    for col in USAGE_COLS:
        df[col] = df["question"].map(
            lambda question: usage_by_question.get(question, {}).get(col, 0)
        )

    return df


def preparar_export_ragas(df):
    df = df.rename(columns={"user_input": "question"})

    missing_cols = [col for col in EXPORT_COLS if col not in df.columns]
    if missing_cols:
        raise RuntimeError(
            "Resultado do RAGAS sem colunas esperadas: "
            f"{', '.join(missing_cols)}"
        )

    null_metrics = df[df[METRIC_COLS].isnull().any(axis=1)]
    if not null_metrics.empty:
        failed_questions = null_metrics["question"].fillna("<pergunta ausente>").tolist()
        print(
            "Aviso: RAGAS retornou metricas nulas para as perguntas: "
            f"{failed_questions}. O CSV sera exportado com esses valores nulos."
        )

    return df[EXPORT_COLS]


def build_ragas_run_config():
    return RunConfig(
        timeout=get_int_env("RAGAS_TIMEOUT_SECONDS", DEFAULT_RAGAS_TIMEOUT_SECONDS),
        max_workers=get_int_env("RAGAS_MAX_WORKERS", DEFAULT_RAGAS_MAX_WORKERS),
    )


def run_ragas(ragas_data, llm, embeddings):
    dataset = Dataset.from_list(ragas_data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        run_config=build_ragas_run_config(),
        raise_exceptions=False,
    )

    print("=== RESULTADOS RAGAS ===")
    print(result)

    df = result.to_pandas().rename(columns={"user_input": "question"})
    df = anexar_metricas_execucao(df, ragas_data)
    df = preparar_export_ragas(df)

    print("Detalhes por query:")
    print(df.to_string(index=False))

    return df


def salvar(df, nome_base):
    if not hasattr(salvar, "_results_dir"):
        base_dir = "results"

        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=False)
            salvar._results_dir = base_dir
        else:
            for n in count(2):
                candidate = f"{base_dir}_{n}"

                if not os.path.exists(candidate):
                    os.makedirs(candidate, exist_ok=False)
                    salvar._results_dir = candidate
                    break

        print(f"Resultados desta execucao serao salvos em: {salvar._results_dir}")

    os.makedirs(salvar._results_dir, exist_ok=True)

    for i in count(1):
        nome = os.path.join(
            salvar._results_dir,
            f"{nome_base}_{i}.csv",
        )

        if not os.path.exists(nome):
            df.to_csv(
                nome,
                index=False,
                encoding="utf-8-sig",
                sep=";",
            )

            print(f"Salvo em: {nome}")
            break
