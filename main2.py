import os
import requests
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
import math

app = FastAPI(title="API Agente (Cliente)", version="3.1.0")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
URL_MEMORIA = "http://127.0.0.1:8000/buscar_contexto"

LISTA_DE_FAQ = [

    "Como mudo minha senha?",
    "Esqueci minha senha",
    "Trocar senha",
    "Qual o horario de atendimento?",
    "Suporte funciona ate que horas?",
    "Como resetar o 2FA?",
    "Perdi meu autenticador de dois fatores"

]

faq_vectors = []

embeddings_model = None


def calcular_similaridade(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm_a * norm_b)

class QuestionRequest(BaseModel):
    question: str
    theme_id: str 

class AnswerResponse(BaseModel):
    question: str
    answer: str
    contextos_usados: int

@app.on_event("startup")
async def startup_event():
    global embeddings_model, faq_vectors
    print("INICIANDO AGENTE...")
    
    if GOOGLE_API_KEY:
        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=GOOGLE_API_KEY
            )
            print("Gerando vetores do FAQ...")
            faq_vectors = embeddings_model.embed_documents(LISTA_DE_FAQ)
            print(f"FAQ Carregado com {len(faq_vectors)} perguntas gatilho.")
        except Exception as e:
            print(f"Erro ao carregar FAQ: {e}")

@app.post("/perguntar", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):

    tema_final = request.theme_id
    is_faq = False

    if embeddings_model and faq_vectors:
        try:
            
            user_vector = embeddings_model.embed_query(request.question)
            
            maior_similaridade = 0
            for i, faq_vec in enumerate(faq_vectors):
                score = calcular_similaridade(user_vector, faq_vec)
                if score > maior_similaridade:
                    maior_similaridade = score
            
            print(f"Nivel de certeza FAQ: {maior_similaridade:.2f}")

       
            if maior_similaridade > 0.50:
                print(">>> ROTEAMENTO ATIVADO: Mudando tema para FAQ")
                tema_final = "FAQ"
                is_faq = True
                
        except Exception as e:
            print(f"Erro no roteamento FAQ: {e}")
    
    payload_para_banco = {
        "query": request.question,
        "theme_id": tema_final,
        "k": 3
    }

    print(f"\nEnviando para o Banco ({URL_MEMORIA}):")
    print(f"Payload: {payload_para_banco}")

    contextos = []

    try:
        response = requests.post(URL_MEMORIA, json=payload_para_banco)
        
        print(f"Status Code do Banco: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Erro retornado pelo Banco: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Erro no Banco: {response.text}")
            
        dados = response.json()
        contextos = dados.get("contextos", [])
        print(f"Contextos recebidos: {len(contextos)}")

    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=502, detail="Nao consegui conectar na API do Banco (Porta 8000). Ela esta ligada?")
    except Exception as e:
        print(f"Erro generico: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not contextos:
        return AnswerResponse(
            question=request.question,
            answer=f"Nao encontrei nada sobre '{request.theme_id}' na memoria. Verifique se o ID esta correto ou faca a ingestao.",
            contextos_usados=0
        )

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY
        )

        template = """
        Voce e um assistente especialista.
        Use as informacoes abaixo para responder a pergunta.
        
        INFORMACOES:
        {context_str}
        
        PERGUNTA: {question}
        """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        contexto_unificado = "\n---\n".join(contextos)

        resposta = chain.invoke({
            "context_str": contexto_unificado,
            "question": request.question
        })
        
        return AnswerResponse(
            question=request.question,
            answer=resposta,
            contextos_usados=len(contextos)
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erro na geracao da IA")