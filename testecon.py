import urllib.parse
from sqlalchemy import create_engine, text

print("--- MONTADOR DE URL SQLALCHEMY ---")

usuario = ""
senha = ""
host = ""
porta = ""
banco = ""

senha_tratada = urllib.parse.quote_plus(senha)

if senha:
    url_final = f"mysql+pymysql://{usuario}:{senha_tratada}@{host}:{porta}/{banco}"
else:

    url_final = f"mysql+pymysql://{usuario}@{host}:{porta}/{banco}"

print("\n" + "="*40)
print(" URL ")
print(url_final)
print("="*40 + "\n")

print("Testando conexao")
try:
    engine = create_engine(url_final)
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print("Conexão funcionou")
except Exception as e:
    print("Não conectou")
    print(e)
    