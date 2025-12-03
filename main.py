import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI(
    title="API RAG Project Solaris/TechShield",
    description="API para consulta de documentos usando Google Gemini e LangChain",
    version="1.0.0"
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("AVISO: A variável de ambiente GOOGLE_API_KEY não foi encontrada.")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    source_documents: List[str] = [] 

rag_chain = None
retriever_global = None

@app.on_event("startup")
async def startup_event():
    global rag_chain, retriever_global
    
    print("Inicializando a base de conhecimento...")

    conteudo_do_txt = """
RELATÓRIO DE SEGURANÇA CIBERNÉTICA 2025
Empresa: TechShield Solutions
Autor: João Silva

1. INTRODUÇÃO
Este documento detalha as novas políticas de senha da TechShield.
A partir de Março de 2025, todas as senhas devem ter 16 caracteres.
O uso de autenticação de dois fatores (2FA) é obrigatório para todos os diretores.

2. INCIDENTES RECENTES
No mês passado, bloqueamos 450 tentativas de phishing vindas de IPs externos.
O servidor 'Alpha-1' foi atualizado para evitar a vulnerabilidade 'Log4Shell'.

3. ORÇAMENTO DE T.I.
Foi aprovada a compra de 50 novos Macbooks para a equipe de desenvolvimento.
O custo total foi de R$ 750.000,00.
"""
    filename = "meu_relatorio.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(conteudo_do_txt)

    loader = TextLoader(filename, encoding="utf-8")
    documento_bruto = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs_processados = splitter.split_documents(documento_bruto)


    if GOOGLE_API_KEY:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=GOOGLE_API_KEY
        )
        
        vector = Chroma.from_documents(
            documents=docs_processados,
            embedding=embeddings,
            collection_name="database1",
            collection_metadata={"teste": "testado", "cor":"azul"},
        )

        retriever_global = vector.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY
        )

        template = """
        Você é um assistente especialista no projeto Solaris e TechShield.
        Use APENAS o contexto fornecido abaixo para responder à pergunta.
        Se a resposta não estiver no contexto, diga que não sabe. Não invente informações.

        Contexto Recuperado:
        {context}

        Pergunta do Usuário:
        {question}

        Resposta Otimizada:
        """
        prompt = PromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever_global | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("Sistema RAG inicializado com sucesso!")
    else:
        print("ERRO: API Key não configurada. O sistema não funcionará corretamente.")


@app.get("/")
def read_root():
    return {"status": "online", "message": "Bem-vindo à API RAG Solaris/TechShield"}

@app.post("/perguntar", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="O sistema RAG não foi inicializado (verifique a API Key).")
    
    try:
        resposta = rag_chain.invoke(request.question)
        
        return AnswerResponse(
            question=request.question,
            answer=resposta
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a pergunta: {str(e)}")