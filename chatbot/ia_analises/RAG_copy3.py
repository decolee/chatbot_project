import pandas as pd
import numpy as np
import streamlit as st
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
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from typing import Any, List, Mapping, Optional
import yaml
import docx2txt
import re
from PIL import Image

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Função para ler o conteúdo de um arquivo de texto
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read()

# Função para carregar o arquivo YAML
def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Função para converter YAML em string formatada
def yaml_to_formatted_string(yaml_content):
    return yaml.dump(yaml_content, default_flow_style=False)

# Função de pré-processamento de texto
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove espaços em branco extras
    text = re.sub(r'[^\w\s]', '', text)  # Remove caracteres especiais
    return text.strip().lower()  # Converte para minúsculas e remove espaços no início/fim

# Criar um corpus de documentos
documents = []

# Diretório a ser analisado
directory_path = r'C:\Users\servidor\Downloads\dm2\chatbot\documentacoes'

# Iterar sobre todos os arquivos no diretório
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    if os.path.isfile(file_path):
        try:
            if filename.endswith('.txt'):
                content = read_text_file(file_path)
                documents.append(Document(page_content=preprocess_text(content), metadata={"source": filename}))
            
            elif filename.endswith('.docx'):
                content = docx2txt.process(file_path)
                documents.append(Document(page_content=preprocess_text(content), metadata={"source": filename}))
            
            elif filename.endswith('.py'):
                content = read_text_file(file_path)
                documents.append(Document(page_content=preprocess_text(content), metadata={"source": filename}))
            
            elif filename.endswith('.yaml') or filename.endswith('.yml'):
                yaml_content = load_yaml_file(file_path)
                formatted_yaml = yaml_to_formatted_string(yaml_content)
                documents.append(Document(page_content=preprocess_text(formatted_yaml), metadata={"source": filename}))
            
            logger.info(f"Arquivo processado: {filename}")
        except Exception as e:
            logger.error(f"Erro ao processar o arquivo {filename}: {str(e)}")

logger.info(f"Total de documentos carregados: {len(documents)}")

# Preparar documentos para o LangChain
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
texts = text_splitter.split_documents(documents)
logger.info(f"Número de chunks após a divisão: {len(texts)}")

# Criar TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit([doc.page_content for doc in texts])

class CustomTfidfEmbeddings(Embeddings):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts):
        return self.vectorizer.transform(texts).toarray()

    def embed_query(self, text):
        return self.vectorizer.transform([text]).toarray()[0]

# Criar embeddings
embeddings = CustomTfidfEmbeddings(vectorizer)

# Criar o FAISS VectorStore
vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)

# Criar o retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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
GROQ_API_KEY = "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"
llm = GroqLLM(api_key=GROQ_API_KEY)

# Configurar a memória
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Configurar o prompt
prompt_template = """Você é um assistente AI especializado em informações sobre o projeto do Santander. 
Use os dados fornecidos para responder às perguntas com precisão, detalhe e contexto. 
Sempre forneça respostas completas e abrangentes.
Analise todos os documentos e procure a resposta neles, pode conter a mesma resposta em dois documentos diferentes.

Contexto: {context}

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

def get_precise_answer(query):
    query_lower = query.lower()
    
    if "configuração do airflow" in query_lower:
        airflow_config = yaml_content.get('services', {}).get('airflow', {})
        return f"A configuração detalhada do Airflow é: {json.dumps(airflow_config, indent=2)}"
    
    elif "spark streaming" in query_lower:
        return f"O Spark Streaming é utilizado no projeto para processamento de dados em tempo real. Detalhes da implementação: {spark_streaming_content[:500]}..."
    
    elif "análise de ia" in query_lower or "análise de inteligência artificial" in query_lower:
        return f"A análise de IA no projeto é realizada usando Spark. Detalhes da implementação: {spark_ai_analysis_content[:500]}..."
    
    elif "objetivos do projeto" in query_lower:
        objectives = "Os principais objetivos do projeto do Santander são: [insira aqui os objetivos extraídos da documentação]"
        return objectives
    
    elif "equipe do projeto" in query_lower:
        team = "A equipe do projeto do Santander é composta por: [insira aqui os membros da equipe extraídos da documentação]"
        return team
    
    elif "cronograma do projeto" in query_lower:
        timeline = "O cronograma do projeto do Santander é o seguinte: [insira aqui o cronograma extraído da documentação]"
        return timeline
    
    elif "benefícios do projeto" in query_lower:
        benefits = "Os principais benefícios do projeto do Santander incluem: [insira aqui os benefícios extraídos da documentação]"
        return benefits
    
    elif "desafios do projeto" in query_lower:
        challenges = "Alguns dos desafios enfrentados pelo projeto do Santander são: [insira aqui os desafios extraídos da documentação]"
        return challenges
    
    else:
        return None

# Configuração do Streamlit
st.set_page_config(page_title="ChamAI Chat Santander", page_icon=":bank:", layout="wide")

# Adicionar estilo personalizado
st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 30px;
    }
    .stButton button {
        background-color: #EC0000;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Caminho para o logotipo
logo_path = r"C:\Users\servidor\Downloads\dm2\document-streaming-main\Streamlit\map\images\santander-logo.png"

# Verificar se o arquivo existe e carregá-lo
if os.path.exists(logo_path):
    try:
        logo = Image.open(logo_path)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(logo, width=350)
    except Exception as e:
        st.error(f"Erro ao carregar o logotipo: {str(e)}")
else:
    st.warning("Arquivo de logotipo não encontrado. Verifique o caminho do arquivo.")

st.title("💬 Chatbot ChamAI Santander")

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
    logger.info(f"User Input: {user_input}")
    logger.info(f"Bot Response: {full_response}")

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

# Adicionar uma seção para visualização de dados
st.sidebar.header("Visualização de Dados")

# Adicionar uma seção para análise de sentimento
st.sidebar.header("Análise de Sentimento")

if st.sidebar.button("Analisar Sentimento da Conversa"):
    from textblob import TextBlob

    sentiments = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            blob = TextBlob(message["content"])
            sentiment = blob.sentiment.polarity
            sentiments.append(sentiment)
    
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    
    st.sidebar.write(f"Sentimento médio da conversa: {avg_sentiment:.2f}")
    if avg_sentiment > 0.2:
        st.sidebar.write("😃 Conversa positiva!")
    elif avg_sentiment < -0.2:
        st.sidebar.write("😞 Conversa negativa.")
    else:
        st.sidebar.write("😐 Conversa neutra.")

# Adicionar uma seção para feedback do usuário
st.sidebar.header("Feedback")

user_rating = st.sidebar.slider("Como você avalia a resposta do chatbot?", 1, 5, 3)
user_feedback = st.sidebar.text_area("Deixe seu comentário (opcional):")

if st.sidebar.button("Enviar Feedback"):
    # Aqui você pode implementar a lógica para salvar o feedback
    st.sidebar.success("Obrigado pelo seu feedback!")

# Adicionar uma seção para métricas de uso
st.sidebar.header("Métricas de Uso")

if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0

st.session_state.total_messages += 1

st.sidebar.write(f"Total de mensagens: {st.session_state.total_messages}")

# Adicionar uma função para salvar o contexto da conversa
def save_context():
    context = {
        "messages": st.session_state.messages,
        "total_messages": st.session_state.total_messages
    }
    with open("conversation_context.json", "w") as f:
        json.dump(context, f)

# Botão para salvar o contexto
if st.sidebar.button("Salvar Contexto"):
    save_context()
    st.sidebar.success("Contexto salvo com sucesso!")

# Função para carregar o contexto
def load_context():
    try:
        with open("conversation_context.json", "r") as f:
            context = json.load(f)
        st.session_state.messages = context["messages"]
        st.session_state.total_messages = context["total_messages"]
        st.rerun()
    except FileNotFoundError:
        st.sidebar.error("Nenhum contexto salvo encontrado.")

# Botão para carregar o contexto
if st.sidebar.button("Carregar Contexto"):
    load_context()

# Adicionar uma seção para configurações avançadas
st.sidebar.header("Configurações Avançadas")

temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.3)
max_tokens = st.sidebar.slider("Máximo de Tokens", 100, 8000, 8000)

# Atualizar as configurações do modelo
model_config = {
    "temperature": temperature,
    "max_tokens": max_tokens,
    "top_p": 0.9,
    "frequency_penalty": 0.2,
    "presence_penalty": 0.1
}

# Adicionar uma seção para logging
st.sidebar.header("Logs")

# Exibir os últimos logs
if st.sidebar.checkbox("Mostrar Logs"):
    with open('chatbot.log', 'r') as log_file:
        st.sidebar.text_area("Logs", log_file.read(), height=200)

# Adicionar uma seção para visualização de embeddings
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

# Adicionar uma seção para busca semântica
st.sidebar.header("Busca Semântica")

semantic_query = st.sidebar.text_input("Digite uma consulta para busca semântica:")

if st.sidebar.button("Realizar Busca Semântica"):
    relevant_docs = retriever.get_relevant_documents(semantic_query)
    st.sidebar.write("Documentos mais relevantes:")
    for i, doc in enumerate(relevant_docs, 1):
        st.sidebar.write(f"{i}. {doc.page_content[:100]}...")

# Ler o conteúdo do arquivo "documentacao.docx"
documentation_path = "C:/Users/servidor/Downloads/dm2/document-streaming-main/documentacoes/documentacao.docx"
documentation_content = docx2txt.process(documentation_path)

# Pré-processamento do texto da documentação
preprocessed_documentation = preprocess_text(documentation_content)

# Adicionar o conteúdo da documentação ao corpus
documents.append(Document(page_content=preprocessed_documentation, metadata={"source": "documentacao.docx"}))

# Atualizar os textos e o vectorstore com o novo documento
texts = text_splitter.split_documents([Document(page_content=preprocessed_documentation, metadata={"source": "documentacao.docx"})])
vectorstore.add_documents(texts)

# Atualizar a função get_precise_answer para incluir informações da documentação
def get_precise_answer(query):
    query_lower = query.lower()
    
    if "objetivos do projeto" in query_lower:
        relevant_docs = retriever.get_relevant_documents("objetivos do projeto")
        objectives = "\n".join([doc.page_content for doc in relevant_docs if "objetivo" in doc.page_content.lower()])
        return f"Os principais objetivos do projeto do Santander são:\n{objectives}"
    
    elif "equipe do projeto" in query_lower:
        relevant_docs = retriever.get_relevant_documents("equipe do projeto")
        team = "\n".join([doc.page_content for doc in relevant_docs if "equipe" in doc.page_content.lower()])
        return f"A equipe do projeto do Santander é composta por:\n{team}"
    
    elif "cronograma do projeto" in query_lower:
        relevant_docs = retriever.get_relevant_documents("cronograma do projeto")
        timeline = "\n".join([doc.page_content for doc in relevant_docs if "cronograma" in doc.page_content.lower()])
        return f"O cronograma do projeto do Santander é o seguinte:\n{timeline}"
    
    elif "benefícios do projeto" in query_lower:
        relevant_docs = retriever.get_relevant_documents("benefícios do projeto")
        benefits = "\n".join([doc.page_content for doc in relevant_docs if "benefício" in doc.page_content.lower()])
        return f"Os principais benefícios do projeto do Santander incluem:\n{benefits}"
    
    elif "desafios do projeto" in query_lower:
        relevant_docs = retriever.get_relevant_documents("desafios do projeto")
        challenges = "\n".join([doc.page_content for doc in relevant_docs if "desafio" in doc.page_content.lower()])
        return f"Alguns dos desafios enfrentados pelo projeto do Santander são:\n{challenges}"
    
    else:
        return None

# Fim do código