import json
import os
import platform
import re
import time
import traceback
from pathlib import Path

import requests
from fastapi import FastAPI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="Quant LangChain Agent")


# =====================================================
# 环境变量配置
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct-q4_K_M").strip()
QUANT_API = os.getenv("QUANT_API", "http://quant_api:8081").strip()
KNOWLEDGE_PATHS = os.getenv("KNOWLEDGE_PATHS", "/app/knowledge").strip()


def detect_ollama_host():
    env_url = os.getenv("OLLAMA_BASE_URL", "").strip()
    if env_url:
        return env_url

    system_name = platform.system().lower()
    if "darwin" in system_name or "mac" in system_name:
        return "http://192.168.1.26:11434"
    if "linux" in system_name:
        return "http://127.0.0.1:11434"
    return "http://127.0.0.1:11434"


OLLAMA_BASE_URL = detect_ollama_host()
print(f"🔍 [Config] Using Ollama endpoint: {OLLAMA_BASE_URL}")


def get_llm(temperature=0.3):
    print(f"🔍 [Force Check] Expecting model: {LOCAL_MODEL_NAME}")
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        models = resp.json().get("models", [])
        available_models = [m["name"] for m in models]
        print(f"✅ [Ollama] Available models: {available_models}")
        if LOCAL_MODEL_NAME not in available_models:
            raise ValueError(f"Model '{LOCAL_MODEL_NAME}' not found. Available: {available_models}")
        print(f"🚀 [Ollama] Forcing use of model: {LOCAL_MODEL_NAME}")
        return Ollama(model=LOCAL_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=temperature)
    except Exception as e:
        print(f"❌ [Ollama] Connection or model selection failed: {e}")

    if not OPENAI_API_KEY:
        raise ValueError("No local model or OpenAI API key available")
    return OpenAI(api_key=OPENAI_API_KEY, temperature=temperature)


class QueryRequest(BaseModel):
    question: str


class WorkflowSpecRequest(BaseModel):
    prompt: str
    strategyId: str
    userId: str = "local-user"


class WorkflowTasksRequest(BaseModel):
    strategySpec: dict


DOCS_CACHE = []


def load_knowledge_docs():
    global DOCS_CACHE
    if DOCS_CACHE:
        return DOCS_CACHE

    docs = []
    allowed_suffix = {".md", ".txt", ".json", ".yaml", ".yml"}
    for base in [p.strip() for p in KNOWLEDGE_PATHS.split(",") if p.strip()]:
        p = Path(base)
        if not p.exists():
            continue
        files = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in allowed_suffix]
        for f in files[:500]:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    docs.append({"path": str(f), "text": text[:12000]})
            except Exception:
                pass
    DOCS_CACHE = docs
    print(f"📚 [RAG] Loaded {len(DOCS_CACHE)} docs from {KNOWLEDGE_PATHS}")
    return DOCS_CACHE


def retrieve_context(query: str, top_k: int = 4):
    docs = load_knowledge_docs()
    if not docs:
        return []
    q_tokens = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
    scored = []
    for d in docs:
        d_tokens = set(re.findall(r"[a-zA-Z0-9_]+", d["text"].lower()))
        score = len(q_tokens & d_tokens)
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]


# MCP-like tools (local tools, protocol-free MVP)
MODULE_CATALOG = {
    "fetch_market_data": {
        "type": "data_collection",
        "module": "quant_data.stock_collector.price_collector.collector",
        "required_params": ["symbols", "timeframe", "lookback_days"],
    },
    "build_features": {
        "type": "feature_engineering",
        "module": "quant_langchain.features.momentum",
        "required_params": ["indicators", "window"],
    },
    "generate_signals": {
        "type": "signal_generation",
        "module": "quant_langchain.signals.rule_engine",
        "required_params": ["rule"],
    },
    "risk_control": {
        "type": "risk_management",
        "module": "quant_langchain.risk.position_manager",
        "required_params": ["max_position_size", "stop_loss"],
    },
    "backtest_strategy": {
        "type": "backtesting",
        "module": "quant_langchain.backtest.engine",
        "required_params": ["initial_cash", "fee_bps"],
    },
}


def list_quant_modules():
    return [{"taskId": k, **v} for k, v in MODULE_CATALOG.items()]


def validate_workflow_dependencies(spec: dict):
    tasks = spec.get("tasks", [])
    ids = {str(t.get("taskId", "")) for t in tasks}
    errors = []
    for t in tasks:
        tid = str(t.get("taskId", ""))
        for dep in t.get("dependencies", []):
            if dep not in ids:
                errors.append(f"{tid}: dependency '{dep}' not found")
    return {"valid": len(errors) == 0, "errors": errors}


def extract_json_object(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return ""
    return text[start : end + 1]


def extract_json_array(text: str):
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return ""
    return text[start : end + 1]


def run_llm(question: str, temperature=0.2):
    llm = get_llm(temperature=temperature)
    prompt = PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain.run(question)


@app.get("/health")
def health_check():
    return {"status": "ok", "model": LOCAL_MODEL_NAME}


@app.post("/api/ask")
def ask_agent(request: QueryRequest):
    try:
        print(f"\n🧠 [Request] Received question: {request.question}")
        start = time.time()
        answer = run_llm(
            f"You are a quant research assistant. Answer clearly and precisely.\n\n{request.question}",
            temperature=0.7,
        )
        end = time.time()
        print(f"✅ [Success] Model responded in {end - start:.2f}s")
        return {"answer": answer}
    except Exception as e:
        print("❌ [Error] in /api/ask:", e)
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/api/generate-script")
def generate_script(request: QueryRequest):
    try:
        print(f"\n🧩 [Request] Generate script for: {request.question}")
        start = time.time()
        script = run_llm(
            "You are an expert quant Python developer. "
            "Generate a clean, runnable Python script.\n\n"
            f"{request.question}",
            temperature=0.3,
        )
        end = time.time()
        print(f"✅ [Success] Script generated in {end - start:.2f}s")
        return {"script": script}
    except Exception as e:
        print("❌ [Error] in /api/generate-script:", e)
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/api/workflow/generate-spec")
def generate_workflow_spec(request: WorkflowSpecRequest):
    try:
        rag_docs = retrieve_context(request.prompt, top_k=4)
        rag_text = "\n\n".join([f"[{d['path']}]\n{d['text'][:1200]}" for d in rag_docs]) or "NO_CONTEXT"
        modules = list_quant_modules()
        modules_json = json.dumps(modules, ensure_ascii=False)

        question = f"""
You are a quant workflow generator with strict JSON output rules.
Output ONLY a valid JSON object.

User prompt:
{request.prompt}

RAG context:
{rag_text}

Available modules from MCP tools:
{modules_json}

Requirements:
1. Return fields: strategyId, workflowId, name, description, market, owner, tasks, processes, risk, backtest, createdAt
2. strategyId and workflowId MUST be "{request.strategyId}"
3. owner MUST be "{request.userId}"
4. tasks MUST follow order: data_collection -> feature_engineering -> signal_generation -> risk_management -> backtesting
5. tasks[*] fields: taskId,type,module,dependencies,parameters
6. processes must include taskFileName mapping for each task
"""
        answer = run_llm(question, temperature=0.2)
        raw = extract_json_object(answer)
        spec = json.loads(raw) if raw else {}
        if "tasks" not in spec:
            raise ValueError("model output missing tasks")

        spec["strategyId"] = request.strategyId
        spec["workflowId"] = request.strategyId
        spec["owner"] = request.userId

        check = validate_workflow_dependencies(spec)
        return {
            "strategySpec": spec,
            "source": "rag+mcp+llm",
            "ragUsed": [d["path"] for d in rag_docs],
            "dependencyCheck": check,
            "model": LOCAL_MODEL_NAME,
        }
    except Exception as e:
        return {"error": str(e), "source": "llm_failed"}


@app.post("/api/workflow/generate-tasks")
def generate_workflow_tasks(request: WorkflowTasksRequest):
    try:
        spec = request.strategySpec
        spec_json = json.dumps(spec, ensure_ascii=False)
        rag_docs = retrieve_context(spec_json, top_k=3)
        rag_text = "\n\n".join([f"[{d['path']}]\n{d['text'][:1000]}" for d in rag_docs]) or "NO_CONTEXT"

        question = f"""
You are a quant python task generator.
Output ONLY a valid JSON array.

WorkflowSpec:
{spec_json}

RAG context:
{rag_text}

Generate task code array.
Each item requires: taskId, taskType, module, fileName, code.
"""
        answer = run_llm(question, temperature=0.2)
        raw = extract_json_array(answer)
        tasks = json.loads(raw) if raw else []
        if not isinstance(tasks, list):
            tasks = []
        return {
            "tasks": tasks,
            "source": "rag+llm",
            "ragUsed": [d["path"] for d in rag_docs],
            "model": LOCAL_MODEL_NAME,
        }
    except Exception as e:
        return {"error": str(e), "source": "llm_failed", "tasks": []}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8083))
    print(f"🚀 Starting LangChain Agent on port {port} ...")
    uvicorn.run(app, host="0.0.0.0", port=port)
