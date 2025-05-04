from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from init import load_environment, LLM_MODEL
from knowledge import get_or_create_db

def configure_qa_chain(db):
    """Configura a cadeia de QA com o modelo LLM usando a nova API da LangChain."""
    # Inicializa o modelo LLM
    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0.2)
    
    # Configura o retriever
    retriever = db.as_retriever(
        search_type="mmr",
        # Recupera os 6 chunks mais relevantes para depois filtrar
        # Recupera 10 documento
        # Faz um balanceamento entre relevância e diversidade
        search_kwargs={"k": 6, "fetch_k": 10, "lambda_mult": 0.7}
    )
    
    # Template do prompt usando o formato de chat
    template = """
    Assistente jurídico especialista em Direito do Consumidor.
    
    Contexto:
    {context}
    
    Pergunta: {question}
    
    
Instruções para a sua resposta:
1. Identifique os artigos específicos do CDC relevantes para a questão
2. Cite textualmente os trechos mais importantes
3. Explique a interpretação jurídica em linguagem acessível
4. Se necessário, mencione jurisprudência relevante
5. Se a pergunta estiver fora do escopo do CDC, explique educadamente.

Responda de forma estruturada com:
- Fundamentação Legal (artigos aplicáveis)
- Explicação (interpretação dos artigos)
"""

    # Cria o prompt formatado
    prompt = ChatPromptTemplate.from_template(template)
    
    # Define a cadeia usando o novo formato de sequência LCEL (LangChain Expression Language)
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain

def query_rag(qa_chain, user_question):
    """Consulta o sistema RAG com uma pergunta usando a nova API."""
    try:
        # Invoca a cadeia com a pergunta do usuário
        result = qa_chain.invoke(user_question)
        return {"result": result}
    except Exception as e:
        # Em caso de erro, retorna uma mensagem de erro como resultado
        return {"result": f"Erro ao processar a pergunta: {str(e)}"}

def run_bot():
    """Executa o bot interativo para responder perguntas sobre fluxo de caixa."""
    # Carrega o ambiente
    load_environment()
    
    # Obtém ou cria o banco de dados vetorial
    db = get_or_create_db()
    
    # Configura a cadeia QA
    qa_chain = configure_qa_chain(db)
    
    print("\n=== Assistente Virtual sobre Direito do Consumidor ===")
    print("Digite suas perguntas sobre Direito do Consumidor ou 'finalizar' para encerrar.\n")
    
    while True:
        question = input("Pergunta: ")
        if question.lower() in ["sair", "exit", "quit", "finalizar"]:
            break
        
        print("\nProcessando pergunta, por favor aguarde...")
        response = query_rag(qa_chain, question)
        
        print("\nResposta:")
        print(response["result"])
        print("\n" + "-"*80 + "\n")

# Ponto de entrada se o script for executado diretamente
if __name__ == "__main__":
    run_bot()