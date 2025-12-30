from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
import os
import requests
import traceback

app = FastAPI(title="Quant LangChain Agent")

# =====================================================
# 环境变量配置
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").strip()
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen2:1.5b-instruct-q4_K_M").strip()
QUANT_API = os.getenv("QUANT_API", "http://quant_api:8081").strip()


# =====================================================
# 自动选择 LLM（优先使用本地 Ollama 模型）
# =====================================================
def get_llm(temperature=0.3):
    try:
        # 检查 Ollama 是否可用
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            print(f"✅ Using local model via Ollama: {LOCAL_MODEL_NAME}")
            return Ollama(model=LOCAL_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=temperature)
    except Exception as e:
        print(f"⚠️ Ollama not available: {e}")

    # 否则回退到 OpenAI
    if not OPENAI_API_KEY:
        raise ValueError("❌ No local model or OpenAI API key available!")
    print("🌐 Falling back to OpenAI API")
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
# 智能问答接口（根据可用性自动切换模型）
# =====================================================
@app.post("/api/ask")
def ask_agent(request: QueryRequest):
    try:
        llm = get_llm(temperature=0.7)
        prompt = PromptTemplate(
            input_variables=["question"],
            template="You are a quant research assistant. Answer this clearly: {question}",
        )
        chain = LLMChain(prompt=prompt, llm=llm)
        answer = chain.run(request.question)
        return {"answer": answer}
    except Exception as e:
        print("❌ Error in /api/ask:", e)
        traceback.print_exc()
        return {"error": str(e)}


# =====================================================
# Python 脚本生成接口
# =====================================================
@app.post("/api/generate-script")
def generate_script(request: QueryRequest):
    try:
        llm = get_llm(temperature=0.3)
        prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "You are an expert quant Python developer. "
                "Generate a clean, runnable Python script for this task:\n\n{question}"
            ),
        )
        chain = LLMChain(prompt=prompt, llm=llm)
        script = chain.run(request.question)
        return {"script": script}
    except Exception as e:
        print("❌ Error in /api/generate-script:", e)
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