# streamlit_app.py
import streamlit as st
import os
import time
import re
import json
from datetime import datetime
from src.init import load_environment
from src.knowledge import get_or_create_db
from src.bot import configure_qa_chain

# Configuração da página
st.set_page_config(
    page_title="Assistente de Direito do Consumidor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado com adições para citações e feedback
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .stTextInput>div>div>input {
        font-size: 1rem;
    }
    
    /* Estilos para citações legais */
    .legal-citation {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        font-size: 0.95rem;
    }
    .citation-title {
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0.3rem;
    }
    .citation-content {
        color: #4B5563;
    }
    
    /* Estilos para feedback */
    .feedback-container {
        display: flex;
        flex-direction: column;
        padding: 0.75rem;
        margin-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    .feedback-question {
        font-size: 0.9rem;
        color: #4B5563;
        margin-bottom: 0.5rem;
    }
    .feedback-buttons {
        display: flex;
        gap: 0.5rem;
    }
    .feedback-button {
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .feedback-button-useful {
        background-color: #D1FAE5;
        color: #047857;
        border: 1px solid #047857;
    }
    .feedback-button-useful:hover {
        background-color: #A7F3D0;
    }
    .feedback-button-notuseful {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #B91C1C;
    }
    .feedback-button-notuseful:hover {
        background-color: #FECACA;
    }
    .feedback-submitted {
        font-size: 0.9rem;
        color: #4B5563;
        margin-top: 0.5rem;
        font-style: italic;
    }
    .feedback-comment {
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Título e descrição
st.markdown('<h1 class="main-header">⚖️ Assistente Virtual de Direito do Consumidor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Tire suas dúvidas sobre direitos do consumidor com base na legislação brasileira atual.</p>', unsafe_allow_html=True)

# Função para processar citações legais em texto
def process_citations(text):
    """
    Identifica e formata citações legais no texto.
    Procura padrões como "Art. X", "Lei n. X", etc.
    """
    # Padrão para identificar referências a artigos do CDC
    article_pattern = r'(Art\.?\s*(\d+[º°]?[A-Z]?)[.\s-]+([^.]+))'
    
    # Padrão para referências a leis
    law_pattern = r'(Lei\s+n[º°.]?\s*([0-9\.]+\/[0-9]+)[.\s-]+([^.]+))'
    
    # Padrão para decretos
    decree_pattern = r'(Decreto\s+n[º°.]?\s*([0-9\.]+\/[0-9]+)[.\s-]+([^.]+))'
    
    # Substitui padrões por HTML formatado
    def replace_with_citation(match):
        full_text = match.group(1)
        reference = match.group(2)
        content = match.group(3)
        
        # Formata como uma citação legal
        return f'<div class="legal-citation"><div class="citation-title">{reference}</div><div class="citation-content">{content}</div></div>'
    
    # Aplica as substituições
    text = re.sub(article_pattern, replace_with_citation, text)
    text = re.sub(law_pattern, replace_with_citation, text)
    text = re.sub(decree_pattern, replace_with_citation, text)
    
    return text

# Função para salvar feedback
def save_feedback(question, answer, feedback, comment=""):
    """Salva o feedback do usuário em um arquivo JSON."""
    feedback_dir = "feedback"
    feedback_file = os.path.join(feedback_dir, "feedback.json")
    
    # Cria diretório de feedback se não existir
    if not os.path.exists(feedback_dir):
        os.makedirs(feedback_dir)
    
    # Prepara dados de feedback
    feedback_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "feedback": feedback,
        "comment": comment
    }
    
    # Carrega feedback existente ou cria um novo arquivo
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r', encoding='utf-8') as f:
            try:
                all_feedback = json.load(f)
            except json.JSONDecodeError:
                all_feedback = []
    else:
        all_feedback = []
    
    # Adiciona novo feedback
    all_feedback.append(feedback_data)
    
    # Salva no arquivo
    with open(feedback_file, 'w', encoding='utf-8') as f:
        json.dump(all_feedback, f, ensure_ascii=False, indent=4)

# Inicialização do sistema
@st.cache_resource
def load_system():
    """Carrega o sistema RAG apenas uma vez."""
    load_environment()
    db = get_or_create_db()
    qa_chain = configure_qa_chain(db)
    return db, qa_chain

# Inicialização na primeira execução
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Sou seu assistente virtual especializado em Direito do Consumidor. Como posso ajudar?", "id": "intro"}
    ]
    
    # Inicializa dicionário para controle de feedback
    st.session_state.feedback = {}
    
    # Mostrar mensagem de carregamento
    with st.spinner("Carregando base de conhecimento de legislação..."):
        # Carregar sistema
        db, qa_chain = load_system()
        st.session_state.db = db
        st.session_state.qa_chain = qa_chain

# Exibe o histórico de mensagens com citações formatadas e opções de feedback
for idx, message in enumerate(st.session_state.messages):
    message_id = message.get("id", f"msg_{idx}")
    
    with st.chat_message(message["role"]):
        # Para mensagens do assistente, processa citações
        if message["role"] == "assistant" and message_id != "intro":
            # Processa citações legais
            formatted_content = process_citations(message["content"])
            st.markdown(formatted_content, unsafe_allow_html=True)
            
            # Adiciona opções de feedback se ainda não foi dado
            if message_id not in st.session_state.feedback:
                with st.container():
                    st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                    st.markdown('<div class="feedback-question">Esta resposta foi útil?</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 1, 4])
                    
                    # Botão de feedback positivo
                    if col1.button("👍 Sim", key=f"useful_{message_id}"):
                        st.session_state.feedback[message_id] = "útil"
                        
                        # Encontra a pergunta correspondente (mensagem anterior)
                        question_idx = idx - 1 if idx > 0 else 0
                        question = st.session_state.messages[question_idx]["content"] if st.session_state.messages[question_idx]["role"] == "user" else ""
                        
                        # Salva feedback
                        save_feedback(question, message["content"], "útil")
                        st.rerun()
                    
                    # Botão de feedback negativo
                    if col2.button("👎 Não", key=f"notuseful_{message_id}"):
                        st.session_state.feedback[message_id] = "não útil"
                        st.session_state[f"show_comment_{message_id}"] = True
                        st.rerun()
                    
                    # Área para comentário (aparece após feedback negativo)
                    if st.session_state.get(f"show_comment_{message_id}", False):
                        with st.container():
                            comment = st.text_area("Por que esta resposta não foi útil?", key=f"comment_{message_id}")
                            if st.button("Enviar comentário", key=f"submit_comment_{message_id}"):
                                # Encontra a pergunta correspondente
                                question_idx = idx - 1 if idx > 0 else 0
                                question = st.session_state.messages[question_idx]["content"] if st.session_state.messages[question_idx]["role"] == "user" else ""
                                
                                # Salva feedback com comentário
                                save_feedback(question, message["content"], "não útil", comment)
                                st.session_state[f"show_comment_{message_id}"] = False
                                st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Mostra confirmação de feedback
                st.markdown(
                    f'<div class="feedback-container"><div class="feedback-submitted">✓ Você avaliou esta resposta como {st.session_state.feedback[message_id]}. Obrigado pelo feedback!</div></div>',
                    unsafe_allow_html=True
                )
        else:
            # Para outras mensagens, exibe normalmente
            st.markdown(message["content"])

# Campo de entrada para nova pergunta
if prompt := st.chat_input("Digite sua dúvida sobre direitos do consumidor..."):
    # Adiciona pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Adiciona um indicador de carregamento enquanto processa
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Consultando a legislação..."):
            # Gera resposta
            response = st.session_state.qa_chain.invoke(prompt)
            
            # Simular efeito de digitação
            full_response = ""
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.01)
                # Processa citações em tempo real para o efeito de digitação
                formatted_response = process_citations(full_response + "▌")
                message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
            
            # Exibir resposta completa com citações formatadas
            formatted_response = process_citations(response)
            message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
    
    # Gera ID único para a mensagem para controle de feedback
    message_id = f"msg_{int(time.time())}"
    
    # Adiciona resposta ao histórico
    st.session_state.messages.append({"role": "assistant", "content": response, "id": message_id})
    
    # Roda o aplicativo novamente para mostrar os controles de feedback
    st.rerun()

# Sidebar
with st.sidebar:
    st.image("https://www.gov.br/pt-br/imagens-de-servicos/defesa-do-consumidor.png/@@images/image", width=100)
    st.title("Direito do Consumidor")
    
    # Adiciona informações sobre as leis disponíveis
    st.subheader("Legislação Disponível")
    
    with st.expander("📚 Leis", expanded=True):
        st.markdown("- **CDC** - Lei 8.078/90 (Código de Defesa do Consumidor)")
        st.markdown("- **Lei n° 9.870/1999** (Mensalidades Escolares)")
        st.markdown("- **Lei n° 10.962/2004** (Informação e Precificação)")
        st.markdown("- **Lei n° 12.741/2012** (Carga Tributária)")
        st.markdown("- **Lei n° 12.291/2010** (Exemplar do CDC)")
    
    with st.expander("📃 Decretos e Resoluções"):
        st.markdown("- **Decreto n° 5.903/2006** (Precificação)")
        st.markdown("- **Decreto n° 7.962/2013** (Comércio Eletrônico)")
        st.markdown("- **Resolução CNSP n° 296**")
        st.markdown("- **Resolução CNSP n° 434**")
    
    # Estatísticas de feedback
    if os.path.exists("feedback/feedback.json"):
        try:
            with open("feedback/feedback.json", 'r', encoding='utf-8') as f:
                all_feedback = json.load(f)
            
            # Calcula estatísticas
            total_feedback = len(all_feedback)
            positive_feedback = sum(1 for item in all_feedback if item["feedback"] == "útil")
            
            # Mostra estatísticas
            st.subheader("Estatísticas de Feedback")
            st.metric("Total de avaliações", total_feedback)
            st.metric("Respostas úteis", f"{positive_feedback} ({int(positive_feedback/total_feedback*100)}%)" if total_feedback > 0 else "0 (0%)")
        except:
            pass
    
    # Botão para limpar o histórico
    if st.button("🗑️ Limpar Conversa", type="primary"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Olá! Sou seu assistente virtual especializado em Direito do Consumidor. Como posso ajudar?", "id": "intro"}
        ]
        st.session_state.feedback = {}
        st.rerun()
    
    # Exemplos de perguntas
    st.subheader("Exemplos de perguntas")
    
    exemplo1 = "Qual o prazo para devolução de produtos com defeito?"
    exemplo2 = "Quais são meus direitos em caso de atraso na entrega?"
    exemplo3 = "Posso cancelar uma compra feita pela internet?"
    
    col1, col2, col3 = st.columns(3)
    
    if col1.button("Exemplo 1"):
        st.session_state.messages.append({"role": "user", "content": exemplo1})
        with st.spinner("Processando..."):
            response = st.session_state.qa_chain.invoke(exemplo1)
            message_id = f"msg_{int(time.time())}"
            st.session_state.messages.append({"role": "assistant", "content": response, "id": message_id})
        st.rerun()
        
    if col2.button("Exemplo 2"):
        st.session_state.messages.append({"role": "user", "content": exemplo2})
        with st.spinner("Processando..."):
            response = st.session_state.qa_chain.invoke(exemplo2)
            message_id = f"msg_{int(time.time())}"
            st.session_state.messages.append({"role": "assistant", "content": response, "id": message_id})
        st.rerun()
        
    if col3.button("Exemplo 3"):
        st.session_state.messages.append({"role": "user", "content": exemplo3})
        with st.spinner("Processando..."):
            response = st.session_state.qa_chain.invoke(exemplo3)
            message_id = f"msg_{int(time.time())}"
            st.session_state.messages.append({"role": "assistant", "content": response, "id": message_id})
        st.rerun()
    
    # Visualização de feedback (apenas para admins ou desenvolvedores)
    with st.expander("👁️ Acessar Feedback Detalhado", expanded=False):
        if st.checkbox("Mostrar feedback detalhado"):
            if os.path.exists("feedback/feedback.json"):
                try:
                    with open("feedback/feedback.json", 'r', encoding='utf-8') as f:
                        feedback_data = json.load(f)
                    
                    if feedback_data:
                        # Mostra dados mais recentes primeiro
                        for idx, item in enumerate(reversed(feedback_data)):
                            if idx > 9:  # Limita a 10 itens para não sobrecarregar
                                break
                                
                            st.markdown(f"**Data:** {item['timestamp']}")
                            st.markdown(f"**Pergunta:** {item['question']}")
                            st.markdown(f"**Resposta:** {item['answer'][:100]}...")
                            st.markdown(f"**Avaliação:** {item['feedback']}")
                            if item.get('comment'):
                                st.markdown(f"**Comentário:** {item['comment']}")
                            st.markdown("---")
                    else:
                        st.info("Nenhum feedback registrado ainda.")
                except:
                    st.error("Erro ao ler dados de feedback.")
            else:
                st.info("Nenhum feedback registrado ainda.")
    
    # Rodapé com informações
    st.markdown("---")
    st.caption("© 2025 RAG Sirius - Sistema de Assistência Jurídica")
    st.caption("Versão 1.0")