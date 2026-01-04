from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
import os
import requests
import traceback
import platform
import json
import time

app = FastAPI(title="Quant LangChain Agent")

# =====================================================
# 环境变量配置
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct-q4_K_M").strip()
QUANT_API = os.getenv("QUANT_API", "http://quant_api:8081").strip()


# =====================================================
# 自动识别运行环境（Mac 本地 → Ubuntu 远程）
# =====================================================
def detect_ollama_host():
    env_url = os.getenv("OLLAMA_BASE_URL", "").strip()
    if env_url:
        return env_url

    system_name = platform.system().lower()
    if "darwin" in system_name or "mac" in system_name:
        return "http://192.168.1.26:11434"
    elif "linux" in system_name:
        return "http://127.0.0.1:11434"
    return "http://127.0.0.1:11434"


OLLAMA_BASE_URL = detect_ollama_host()
print(f"🔍 [Config] Using Ollama endpoint: {OLLAMA_BASE_URL}")


# =====================================================
# 测试 Ollama 是否可用
# =====================================================
def test_ollama_connection():
    try:
        print(f"🔎 [Check] Connecting to Ollama server at {OLLAMA_BASE_URL}/api/tags ...")
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            print(f"✅ [Ollama] Connected successfully! Found {len(models)} models:")
            for m in models:
                print(f"   - {m['name']} ({m['details'].get('parameter_size')}, {m['details'].get('quantization_level')})")
            return True
        else:
            print(f"⚠️ [Ollama] Unexpected response: HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ [Ollama] Connection failed: {e}")
        return False


# =====================================================
# 获取可用 LLM（优先本地 Ollama）
# =====================================================
def get_llm(temperature=0.3):
    print(f"🔍 [Force Check] Expecting model: {LOCAL_MODEL_NAME}")

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        models = resp.json().get("models", [])
        available_models = [m["name"] for m in models]

        print(f"✅ [Ollama] Available models: {available_models}")

        # 如果没找到指定模型，则直接报错
        if LOCAL_MODEL_NAME not in available_models:
            raise ValueError(f"❌ Model '{LOCAL_MODEL_NAME}' not found. Available: {available_models}")

        print(f"🚀 [Ollama] Forcing use of model: {LOCAL_MODEL_NAME}")
        return Ollama(model=LOCAL_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=temperature)

    except Exception as e:
        print(f"❌ [Ollama] Connection or model selection failed: {e}")

    # fallback
    if not OPENAI_API_KEY:
        raise ValueError("❌ No local model or OpenAI API key available!")
    return OpenAI(api_key=OPENAI_API_KEY, temperature=temperature)


# =====================================================
# 请求模型
# =====================================================
class QueryRequest(BaseModel):
    question: str


# =====================================================
# 健康检查接口
# =====================================================
@app.get("/health")
def health_check():
    return {"status": "ok"}


# =====================================================
# 智能问答接口
# =====================================================
@app.post("/api/ask")
def ask_agent(request: QueryRequest):
    try:
        print(f"\n🧠 [Request] Received question: {request.question}")
        llm = get_llm(temperature=0.7)
        prompt = PromptTemplate(
            input_variables=["question"],
            template="You are a quant research assistant. Answer this clearly: {question}",
        )

        start = time.time()
        chain = LLMChain(prompt=prompt, llm=llm)
        answer = chain.run(request.question)
        end = time.time()

        print(f"✅ [Success] Model responded in {end - start:.2f}s")
        print(f"🗣️ [Answer Preview]: {answer[:200]}...\n")
        return {"answer": answer}
    except Exception as e:
        print("❌ [Error] in /api/ask:", e)
        traceback.print_exc()
        return {"error": str(e)}


# =====================================================
# Python 脚本生成接口
# =====================================================
@app.post("/api/generate-script")
def generate_script(request: QueryRequest):
    try:
        print(f"\n🧩 [Request] Generate script for: {request.question}")
        llm = get_llm(temperature=0.3)
        prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "You are an expert quant Python developer. "
                "Generate a clean, runnable Python script for this task:\n\n{question}"
            ),
        )
        start = time.time()
        chain = LLMChain(prompt=prompt, llm=llm)
        script = chain.run(request.question)
        end = time.time()

        print(f"✅ [Success] Script generated in {end - start:.2f}s")
        print(f"📝 [Script Preview]: {script[:200]}...\n")
        return {"script": script}
    except Exception as e:
        print("❌ [Error] in /api/generate-script:", e)
        traceback.print_exc()
        return {"error": str(e)}


# =====================================================
# 启动入口
# =====================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8083))
    print(f"🚀 Starting LangChain Agent on port {port} ...")
    uvicorn.run(app, host="0.0.0.0", port=port)