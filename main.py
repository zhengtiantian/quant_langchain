from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os
import requests

app = FastAPI(title="Quant LangChain Agent")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QUANT_API = os.getenv("QUANT_API", "http://quant_api:8081")


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/ask")
def ask_agent(request: QueryRequest):
    llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["question"],
        template="You are a quant research assistant. Answer this: {question}",
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    answer = chain.run(request.question)
    return {"answer": answer}


@app.post("/api/generate-script")
def generate_script(request: QueryRequest):
    """生成 Python 脚本，用于量化分析或数据抓取"""
    llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.3)
    prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are an expert quant Python developer. "
            "Generate a complete Python script (no explanation) for this task:\n\n{question}"
        ),
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    script = chain.run(request.question)
    return {"script": script}