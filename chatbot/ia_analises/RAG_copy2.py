import pandas as pd
import numpy as np
import streamlit as st
from groq import Groq
import os
import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import CharacterTextSplitter
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

# Criar um corpus de documentos
documents = []

# Diretório a ser analisado
directory_path = r'C:\Users\servidor\Downloads\dm2\document-streaming-main\documentacoes'

# Iterar sobre todos os arquivos no diretório
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    if os.path.isfile(file_path):
        try:
            if filename.endswith('.txt'):
                content = read_text_file(file_path)
                documents.append(f"Arquivo de texto ({filename}): {content}")
            
            elif filename.endswith('.docx'):
                content = docx2txt.process(file_path)
                documents.append(f"Arquivo Word ({filename}): {content}")
            
            elif filename.endswith('.py'):
                content = read_text_file(file_path)
                documents.append(f"Arquivo Python ({filename}): {content}")
            
            elif filename.endswith('.yaml') or filename.endswith('.yml'):
                yaml_content = load_yaml_file(file_path)
                formatted_yaml = yaml_to_formatted_string(yaml_content)
                documents.append(f"Arquivo YAML ({filename}): {formatted_yaml}")
            
            
            
        except Exception as e:
            print(f"Erro ao processar o arquivo {filename}: {str(e)}")

# Preparar documentos para o LangChain
text_splitter = CharacterTextSplitter(chunk_size=20500, chunk_overlap=0)
texts = text_splitter.create_documents(documents)

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
retriever = vectorstore.as_retriever()



# Classe wrapper para o Groq
class GroqLLM(LLM):
    client: Any
    model: str = "llama-3.1-70b-versatile"  # Substitua pelo modelo fine-tuned específico do projeto do Santander
    
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
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Configurar o prompt
from langchain.prompts import PromptTemplate
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
from langchain.chains import ConversationalRetrievalChain
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
st.set_page_config(page_title="Chatbot do Projeto Santander", page_icon=":bank:", layout="wide")
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
import docx2txt

documentation_path = "C:/Users/servidor/Downloads/dm2/document-streaming-main/documentacoes/documentacao.docx"
documentation_content = docx2txt.process(documentation_path)

# Pré-processamento do texto da documentação
# Realize aqui as etapas de pré-processamento, como remoção de caracteres especiais, tokenização, remoção de stopwords, etc.
# Aplique técnicas de normalização, como lowercase e lematização, para melhorar a qualidade do texto.
# Divida o texto em chunks menores para facilitar o fine-tuning do modelo.

# Adicionar o conteúdo da documentação ao corpus
documents.append(f"Documentação do Projeto: {documentation_content}")

# Atualizar a função get_precise_answer para incluir informações da documentação
def get_precise_answer(query):
    query_lower = query.lower()
    
    if "objetivos do projeto" in query_lower:
        # Extrair os objetivos do projeto da documentação
        objectives = "Os principais objetivos do projeto do Santander são: [insira aqui os objetivos extraídos da documentação]"
        return objectives
    
    elif "equipe do projeto" in query_lower:
        # Extrair informações sobre a equipe do projeto da documentação
        team = "A equipe do projeto do Santander é composta por: [insira aqui os membros da equipe extraídos da documentação]"
        return team
    
    elif "cronograma do projeto" in query_lower:
        # Extrair o cronograma do projeto da documentação
        timeline = "O cronograma do projeto do Santander é o seguinte: [insira aqui o cronograma extraído da documentação]"
        return timeline
    
    elif "benefícios do projeto" in query_lower:
        # Extrair os benefícios do projeto da documentação
        benefits = "Os principais benefícios do projeto do Santander incluem: [insira aqui os benefícios extraídos da documentação]"
        return benefits
    
    elif "desafios do projeto" in query_lower:
        # Extrair os desafios do projeto da documentação
        challenges = "Alguns dos desafios enfrentados pelo projeto do Santander são: [insira aqui os desafios extraídos da documentação]"
        return challenges
    
    else:
        return None

# Fim do código

