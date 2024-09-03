import streamlit as st
# Configuração do Streamlit (deve ser a primeira chamada Streamlit)
st.set_page_config(page_title="Chatbot do Projeto Santander", page_icon=":bank:", layout="wide")

import pandas as pd
import numpy as np
from groq import Groq
import os
import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, FakeEmbeddings
from langchain.schema import Document
from typing import Any, List, Mapping, Optional
import yaml
import docx2txt
import ast
import subprocess
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PythonLoader, Docx2txtLoader

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Diretório a ser analisado
directory_path = r'C:\Users\servidor\Downloads\dm2\document-streaming-main\documentacoes'

# Funções auxiliares
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read()

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def yaml_to_formatted_string(yaml_content):
    return yaml.dump(yaml_content, default_flow_style=False)

def get_loader_for_file(file_path: str):
    if file_path.endswith('.txt'):
        return TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith('.docx'):
        return Docx2txtLoader(file_path)
    else:
        return None

def safe_load_document(loader, file_path: str) -> List[Document]:
    try:
        return loader.load()
    except Exception as e:
        logging.error(f"Erro ao carregar {file_path}: {str(e)}")
        return [Document(page_content=f"Erro ao carregar {file_path}", metadata={"source": file_path})]

def load_documents_from_directory(directory: str) -> List[Document]:
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.txt', '.docx')):
                file_path = os.path.join(root, file)
                loader = get_loader_for_file(file_path)
                if loader:
                    documents.extend(safe_load_document(loader, file_path))
                    logging.info(f"Documento carregado: {file_path}")
                else:
                    logging.warning(f"Tipo de arquivo não suportado: {file_path}")
    return documents

# Carregar documentos
@st.cache_resource
def load_and_process_documents():
    documents = load_documents_from_directory(directory_path)
    logging.info(f"Total de documentos carregados: {len(documents)}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    return texts

texts = load_and_process_documents()

# Função para criar embeddings
@st.cache_resource
def create_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings()
        logging.info("Usando HuggingFaceEmbeddings")
        return embeddings
    except Exception as e:
        logging.warning(f"Erro ao carregar HuggingFaceEmbeddings: {e}")
        logging.warning("Usando FakeEmbeddings como fallback")
        return FakeEmbeddings(size=768)

embeddings = create_embeddings()

# Criar o FAISS VectorStore
@st.cache_resource
def create_vectorstore():
    return FAISS.from_documents(texts, embeddings)

vectorstore = create_vectorstore()
retriever = vectorstore.as_retriever()

# Classe wrapper para o Groq
class GroqLLM(LLM):
    client: Any
    model: str = "llama-3.1-70b-versatile"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.client = Groq(api_key=api_key)
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.3,
            top_p=0.9,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            stream=False
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

# Configurar o modelo de linguagem (usando Groq)
@st.cache_resource
def setup_llm():
    GROQ_API_KEY = "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"
    return GroqLLM(api_key=GROQ_API_KEY)

llm = setup_llm()

# Configurar a memória e o prompt
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt_template = """Você é um assistente AI especializado no projeto do Santander e em análise de código. 
Sua tarefa é responder perguntas sobre o código, documentação e funcionalidades do projeto.

Contexto do projeto:
{context}

Histórico da conversa:
{chat_history}

Pergunta do usuário: {question}

Instruções:
1. Analise cuidadosamente o contexto e a pergunta.
2. Se a pergunta for sobre código, forneça explicações detalhadas e, se possível, exemplos de código.
3. Se for sobre documentação ou funcionalidades, dê uma resposta abrangente baseada nas informações disponíveis.
4. Se não tiver certeza sobre algo, indique claramente e sugira onde o usuário pode encontrar mais informações.

Resposta:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "chat_history", "question"]
)

# Criar a cadeia de perguntas e respostas
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT}
)

# Funções auxiliares
def analyze_python_code(code_string):
    try:
        tree = ast.parse(code_string)
        analysis = {
            "imports": [],
            "functions": [],
            "classes": []
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                analysis["imports"].extend(n.name for n in node.names)
            elif isinstance(node, ast.FunctionDef):
                analysis["functions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                analysis["classes"].append(node.name)
        return analysis
    except SyntaxError:
        return "Erro de sintaxe no código fornecido."

def run_pylint(file_path):
    result = subprocess.run(['pylint', file_path], capture_output=True, text=True)
    return result.stdout

def run_pytest(file_path):
    result = subprocess.run(['pytest', file_path], capture_output=True, text=True)
    return result.stdout

def display_code_with_highlighting(code):
    lexer = PythonLexer()
    formatter = HtmlFormatter(style="friendly")
    highlighted_code = highlight(code, lexer, formatter)
    st.markdown(f'<style>{formatter.get_style_defs(".highlight")}</style>', unsafe_allow_html=True)
    st.markdown(highlighted_code, unsafe_allow_html=True)

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment

def get_precise_answer(query):
    query_lower = query.lower()
    
    if "analisar código" in query_lower:
        code_start = query.find("analisar código:") + len("analisar código:")
        code_to_analyze = query[code_start:].strip()
        
        if not code_to_analyze:
            return "Por favor, forneça o código para análise após 'analisar código:'"
        
        analysis_result = analyze_python_code(code_to_analyze)
        return f"Análise do código:\n{json.dumps(analysis_result, indent=2)}"
    
    elif "configuração do airflow" in query_lower:
        airflow_config = yaml_content.get('services', {}).get('airflow', {})
        return f"A configuração detalhada do Airflow é: {json.dumps(airflow_config, indent=2)}"
    
    elif "spark streaming" in query_lower:
        return f"O Spark Streaming é utilizado no projeto para processamento de dados em tempo real. Detalhes da implementação: {spark_streaming_content[:500]}..."
    
    elif "análise de ia" in query_lower or "análise de inteligência artificial" in query_lower:
        return f"A análise de IA no projeto é realizada usando Spark. Detalhes da implementação: {spark_ai_analysis_content[:500]}..."
    
    elif "objetivos do projeto" in query_lower:
        return "Os principais objetivos do projeto do Santander são: [insira aqui os objetivos extraídos da documentação]"
    
    elif "equipe do projeto" in query_lower:
        return "A equipe do projeto do Santander é composta por: [insira aqui os membros da equipe extraídos da documentação]"
    
    elif "cronograma do projeto" in query_lower:
        return "O cronograma do projeto do Santander é o seguinte: [insira aqui o cronograma extraído da documentação]"
    
    elif "benefícios do projeto" in query_lower:
        return "Os principais benefícios do projeto do Santander incluem: [insira aqui os benefícios extraídos da documentação]"
    
    elif "desafios do projeto" in query_lower:
        return "Alguns dos desafios enfrentados pelo projeto do Santander são: [insira aqui os desafios extraídos da documentação]"
    
    else:
        return None

# Adicionar logotipo e cores do Santander
st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 30px;
    }
    .logo-container img {
        max-width: 200px;
    }
    .stButton button {
        background-color: #EC0000;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="logo-container">
        <img src="images/santander-logo.png" alt="Santander Logo" style="width:150px;">
    </div>
    """,
    unsafe_allow_html=True
)

# Interface do chatbot
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Digite sua pergunta sobre o projeto do Santander:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    precise_answer = get_precise_answer(user_input)
    
    if precise_answer:
        full_response = precise_answer
    else:
        result = qa_chain({"question": user_input})
        full_response = result['answer']
    
    with st.chat_message("assistant"):
        st.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    logging.info(f"User Input: {user_input}")
    logging.info(f"Bot Response: {full_response}")

if st.button("Limpar Histórico"):
    st.session_state.messages = []
    memory.clear()
    st.rerun()

if st.button("Exportar Conversa"):
    conversation = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
    st.download_button(
        label="Download Conversa",
        data=conversation,
        file_name="conversa_chatbot.txt",
        mime="text/plain"
    )

# Sidebar
st.sidebar.title("Configurações e Ferramentas")

# Análise de Sentimento
st.sidebar.header("Análise de Sentimento")
if st.sidebar.button("Analisar Sentimento da Conversa"):
    sentiments = [analyze_sentiment(m["content"])['compound'] for m in st.session_state.messages if m["role"] == "user"]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    
    st.sidebar.write(f"Sentimento médio da conversa: {avg_sentiment:.2f}")
    if avg_sentiment > 0.05:
        st.sidebar.write("😃 Conversa positiva!")
    elif avg_sentiment < -0.05:
        st.sidebar.write("😞 Conversa negativa.")
    else:
        st.sidebar.write("😐 Conversa neutra.")

# Feedback do Usuário
st.sidebar.header("Feedback")
feedback_options = ["Muito Útil", "Útil", "Neutro", "Pouco Útil", "Nada Útil"]
user_feedback = st.sidebar.selectbox("Como você avalia a resposta do chatbot?", feedback_options)
feedback_comment = st.sidebar.text_area("Deixe seu comentário (opcional):")

if st.sidebar.button("Enviar Feedback"):
    # Aqui você pode implementar a lógica para salvar o feedback
    st.sidebar.success("Obrigado pelo seu feedback!")

# Métricas de Uso
st.sidebar.header("Métricas de Uso")
if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0
st.session_state.total_messages += 1
st.sidebar.write(f"Total de mensagens: {st.session_state.total_messages}")

# Salvar e Carregar Contexto
def save_context():
    context = {
        "messages": st.session_state.messages,
        "total_messages": st.session_state.total_messages
    }
    with open("conversation_context.json", "w") as f:
        json.dump(context, f)

def load_context():
    try:
        with open("conversation_context.json", "r") as f:
            context = json.load(f)
        st.session_state.messages = context["messages"]
        st.session_state.total_messages = context["total_messages"]
        st.rerun()
    except FileNotFoundError:
        st.sidebar.error("Nenhum contexto salvo encontrado.")

if st.sidebar.button("Salvar Contexto"):
    save_context()
    st.sidebar.success("Contexto salvo com sucesso!")

if st.sidebar.button("Carregar Contexto"):
    load_context()

# Configurações Avançadas
st.sidebar.header("Configurações Avançadas")
temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.3)
max_tokens = st.sidebar.slider("Máximo de Tokens", 100, 8000, 8000)

model_config = {
    "temperature": temperature,
    "max_tokens": max_tokens,
    "top_p": 0.9,
    "frequency_penalty": 0.2,
    "presence_penalty": 0.1
}

# Logs
st.sidebar.header("Logs")
if st.sidebar.checkbox("Mostrar Logs"):
    with open('chatbot.log', 'r') as log_file:
        st.sidebar.text_area("Logs", log_file.read(), height=200)

# Visualização de Embeddings
st.sidebar.header("Visualização de Embeddings")
if st.sidebar.button("Visualizar Embeddings"):
    from sklearn.decomposition import PCA
    import plotly.express as px

    # Realizar PCA nos embeddings
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(vectorizer.transform([doc.page_content for doc in texts]).toarray())

    # Criar um DataFrame com os embeddings 2D
    df_embeddings = pd.DataFrame(embeddings_2d, columns=['PC1', 'PC2'])
    df_embeddings['Documento'] = [f"Doc {i}" for i in range(len(df_embeddings))]

    # Criar o gráfico de dispersão
    fig = px.scatter(df_embeddings, x='PC1', y='PC2', hover_data=['Documento'])
    st.plotly_chart(fig)

# Busca Semântica
st.sidebar.header("Busca Semântica")
semantic_query = st.sidebar.text_input("Digite uma consulta para busca semântica:")
if st.sidebar.button("Realizar Busca Semântica"):
    relevant_docs = retriever.get_relevant_documents(semantic_query)
    st.sidebar.write("Documentos mais relevantes:")
    for i, doc in enumerate(relevant_docs, 1):
        st.sidebar.write(f"{i}. {doc.page_content[:100]}...")

# Análise de Código
st.sidebar.header("Análise de Código")
code_to_analyze = st.sidebar.text_area("Cole o código Python para análise:")
if st.sidebar.button("Analisar Código"):
    if code_to_analyze:
        analysis_result = analyze_python_code(code_to_analyze)
        st.sidebar.json(analysis_result)
    else:
        st.sidebar.warning("Por favor, insira algum código para análise.")

# Ferramentas de Desenvolvimento
st.sidebar.header("Ferramentas de Desenvolvimento")
if st.sidebar.button("Executar Pylint"):
    file_to_analyze = st.sidebar.text_input("Caminho do arquivo para análise Pylint:")
    if file_to_analyze:
        pylint_result = run_pylint(file_to_analyze)
        st.sidebar.text_area("Resultado do Pylint", pylint_result)

if st.sidebar.button("Executar Testes"):
    test_file = st.sidebar.text_input("Caminho do arquivo de teste:")
    if test_file:
        pytest_result = run_pytest(test_file)
        st.sidebar.text_area("Resultado dos Testes", pytest_result)

# Visualização de Código
st.sidebar.header("Visualização de Código")
if st.sidebar.checkbox("Visualizar Código"):
    file_list = [f for f in os.listdir(directory_path) if f.endswith('.py')]
    selected_file = st.sidebar.selectbox("Selecione um arquivo para visualizar:", file_list)
    if selected_file:
        file_path = os.path.join(directory_path, selected_file)
        with open(file_path, 'r') as file:
            code_content = file.read()
        display_code_with_highlighting(code_content)

# Função principal
def main():
    st.title("💬 Chatbot do Projeto Santander")

    # Adicionar logotipo e cores do Santander
    st.markdown(
        """
        <style>
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 30px;
        }
        .logo-container img {
            max-width: 200px;
        }
        .stButton button {
            background-color: #EC0000;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="logo-container">
            <img src="images/santander-logo.png" alt="Santander Logo" style="width:150px;">
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
