import os
import requests
import traceback
import math
from typing import List

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


DATABASE_URL = ""

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(title="API Agente (SQL + FAQ Dinâmico)", version="2.0.0")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
URL_MEMORIA = "http://127.0.0.1:8000/buscar_contexto"

embeddings_model = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def calcular_similaridade(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

class QuestionRequest(BaseModel):
    bot_id: int
    question: str

@app.on_event("startup")
async def startup_event():
    global embeddings_model
    print("--- INICIANDO AGENTE ---")
    if GOOGLE_API_KEY:
        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=GOOGLE_API_KEY
            )
            print("Modelo de Embeddings pronto.")
        except Exception as e:
            print(f"Erro ao carregar modelo de embeddings: {e}")

@app.post("/perguntar")
async def ask_question(request: QuestionRequest, db=Depends(get_db)):
    
    print(f"\n--- Nova Pergunta (Bot ID {request.bot_id}): {request.question} ---")
    
    query_bot = text("""
        SELECT c.name, k.kb_id
        FROM chatbots c
        LEFT JOIN chatbot_kb k ON c.id = k.bot_id
        WHERE c.id = :bid
    """)
    
    query_faq = text("SELECT question FROM chatbot_faqs WHERE bot_id = :bid")

    try:
        result_bot = db.execute(query_bot, {"bid": request.bot_id}).fetchone()
        if not result_bot:
            raise HTTPException(status_code=404, detail="Bot não encontrado")
        
        bot_name = result_bot[0]
        kb_collection_name = result_bot[1]
        
        result_faqs = db.execute(query_faq, {"bid": request.bot_id}).fetchall()
        lista_faq_sql = [row[0] for row in result_faqs] 
        
        print(f"Bot: {bot_name} | Memória: {kb_collection_name} | FAQs Carregados: {len(lista_faq_sql)}")

    except Exception as e:
        print(f"Erro SQL: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no Banco de Dados")

    is_faq_match = False
    
   
    if embeddings_model and lista_faq_sql:
        try:
            user_vector = embeddings_model.embed_query(request.question)
            
           
            faq_vectors = embeddings_model.embed_documents(lista_faq_sql)
            
            
            maior_similaridade = 0
            pergunta_mais_parecida = ""
            
            for i, faq_vec in enumerate(faq_vectors):
                score = calcular_similaridade(user_vector, faq_vec)
                if score > maior_similaridade:
                    maior_similaridade = score
                    pergunta_mais_parecida = lista_faq_sql[i]
            
            print(f"Maior similaridade: {maior_similaridade:.2f} com '{pergunta_mais_parecida}'")

            if maior_similaridade > 0.88:
                print(">>> FAQ DETECTADO! <<<")
                is_faq_match = True
                
        except Exception as e:
            print(f"Erro na verificação de FAQ: {e}")

 
    contextos_texto = ""
    if kb_collection_name:
        payload_memoria = {
            "query": request.question, 
            "collection_name": kb_collection_name,
            "k": 3
        }
        try:
            resp = requests.post(URL_MEMORIA, json=payload_memoria)
            if resp.status_code == 200:
                lista = resp.json().get("contextos", [])
                contextos_texto = "\n---\n".join(lista)
        except Exception as e:
            print(f"Erro conexão memória: {e}")

    try:
       
        temp = 0.1 if is_faq_match else 0.4 
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temp,
            google_api_key=GOOGLE_API_KEY
        )

    
        instrucao_faq = ""
        if is_faq_match:
            instrucao_faq = "ATENÇÃO: O usuário fez uma pergunta identificada como frequente (FAQ). Use o contexto abaixo para responder de forma direta e precisa."

        prompt_final = f"""
        Você é o assistente virtual {bot_name}.
        {instrucao_faq}
        
        CONTEXTO DE CONHECIMENTO:
        {contextos_texto}
        
        PERGUNTA DO USUÁRIO: {request.question}
        """
        
        response = llm.invoke(prompt_final)
        
        return {
            "bot_name": bot_name,
            "answer": response.content,
            "is_faq": is_faq_match,
            "similaridade": f"{maior_similaridade:.2f}" if 'maior_similaridade' in locals() else "0"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erro na geração da IA")