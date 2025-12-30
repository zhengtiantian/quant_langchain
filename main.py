from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
import os
import requests

app = FastAPI(title="Quant LangChain Agent")

# 环境变量配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen2:1.5b-instruct-q4_K_M")
QUANT_API = os.getenv("QUANT_API", "http://quant_api:8081")

# =====================================================
# 自动选择 LLM（优先使用本地模型）
# =====================================================
def get_llm(temperature=0.3):
    try:
        # 检查 Ollama 是否可用
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            print(f"✅ Using local model via Ollama: {LOCAL_MODEL_NAME}")
            return Ollama(model=LOCAL_MODEL_NAME, base_url=OLLAMA_BASE_URL)
    except Exception:
        pass

    # 否则使用 OpenAI
    if not OPENAI_API_KEY:
        raise ValueError("❌ No local model or OpenAI key available!")
    print("🌐 Falling back to OpenAI API")
    return OpenAI(api_key=OPENAI_API_KEY, temperature=temperature)


# =====================================================
# Pydantic 模型
# =====================================================
class QueryRequest(BaseModel):
    question: str


# =====================================================
# 健康检查
# =====================================================
@app.get("/health")
def health_check():
    return {"status": "ok"}


# =====================================================
# 问答接口（智能选择模型）
# =====================================================
@app.post("/api/ask")
def ask_agent(request: QueryRequest):
    llm = get_llm(temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["question"],
        template="You are a quant research assistant. Answer this clearly: {question}",
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    answer = chain.run(request.question)
    return {"answer": answer}


# =====================================================
# Python脚本生成接口
# =====================================================
@app.post("/api/generate-script")
def generate_script(request: QueryRequest):
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


# =====================================================
# 启动
# =====================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8083))
    uvicorn.run(app, host="0.0.0.0", port=port)