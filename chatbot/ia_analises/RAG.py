import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from groq import Groq
import os
import json
import logging

# Configuração de logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Função para ler o conteúdo de um arquivo de texto
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read()

# Carregando as tabelas existentes
df_dummy = pd.read_csv("C:/Users/servidor/Downloads/dm2/document-streaming-main/dummys_tables_create/tabela_dummy_func.csv")
df_auxiliar = pd.read_csv("C:/Users/servidor/Downloads/dm2/document-streaming-main/dummys_tables_create/tabela_auxiliar_siglas.csv")
df_correlacao = pd.read_csv("C:/Users/servidor/Downloads/dm2/document-streaming-main/dummys_tables_create/tabela_correlacao.csv")
df_amarracoes = pd.read_csv("C:/Users/servidor/Downloads/dm2/document-streaming-main/dummys_tables_create/tabela_amarracoes.csv")

# Criar um corpus de documentos
documents = []

# Função para adicionar dados do DataFrame ao corpus
def add_dataframe_to_corpus(df, prefix):
    for _, row in df.iterrows():
        doc = f"{prefix}: " + ", ".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(doc)

# Adicionar dados de todos os DataFrames ao corpus
add_dataframe_to_corpus(df_dummy, "Funcionário")
add_dataframe_to_corpus(df_auxiliar, "Sigla")
add_dataframe_to_corpus(df_correlacao, "Correlação")
add_dataframe_to_corpus(df_amarracoes, "Amarração")

# Adicionar conteúdo dos arquivos de texto
new_files = [
    "comprovantes_de_pagamento.txt",
    "credito_imobiliario.txt",
    "duvidas_comprovantes_de_pagamento.txt",
    "extrato_santander_como_emitir.txt",
    "guia_imposto_de_renda.txt",
    "imposto_de_renda_2024.txt",
    "ne_10m_admin_0_countries_bra.VERSION.txt",
    "seguro_cartao.txt",
    "transacoes_seguras.txt"
]

base_path = "C:/Users/servidor/Downloads/dm2/document-streaming-main/Streamlit/map/rags/"

for file in new_files:
    file_path = os.path.join(base_path, file)
    content = read_text_file(file_path)
    documents.append(f"Arquivo: {file}, Conteúdo: {content}")

# Criação de Embeddings
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(documents)

# Armazenamento de embeddings
embeddings_array = embeddings.toarray()

# Função para recuperar documentos relevantes
def get_relevant_documents(query, top_k=10):
    query_vector = vectorizer.transform([query]).toarray()
    similarities = cosine_similarity(query_vector, embeddings_array)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]

# Configuração do Streamlit e Groq
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

st.title("💬 Chatbot Empresarial com RAG Avançado")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Digite sua pergunta sobre a empresa, funcionários ou serviços bancários:")

# Adicione esta função após a definição de get_relevant_documents
def get_precise_answer(query):
    query_lower = query.lower()
    if "total de funcionarios" in query_lower or "quantos funcionarios" in query_lower:
        return f"O total de funcionários cadastrados é {len(df_dummy)}."
    elif "total de siglas" in query_lower or "quantas siglas" in query_lower:
        return f"O total de siglas cadastradas é {len(df_auxiliar)}."
    elif "total de correlações" in query_lower or "quantas correlações" in query_lower:
        return f"O total de correlações cadastradas é {len(df_correlacao)}."
    elif "total de amarrações" in query_lower or "quantas amarrações" in query_lower:
        return f"O total de amarrações cadastradas é {len(df_amarracoes)}."
    elif "funcionários na sigla" in query_lower or "funcionários de" in query_lower:
        sigla = query_lower.split("de")[-1].strip().upper()
        count = len(df_dummy[df_dummy['Sigla'] == sigla])
        return f"Há {count} funcionários na sigla/departamento {sigla}."
    elif "cargo de" in query_lower:
        nome = query_lower.split("cargo de")[-1].strip()
        funcionario = df_dummy[df_dummy['Nome'].str.lower() == nome.lower()]
        if not funcionario.empty:
            return f"O cargo de {nome} é {funcionario.iloc[0]['Cargo']}."
        else:
            return f"Não foi encontrado um funcionário com o nome {nome}."
    else:
        return None

    
    return None

# Modifique a parte do código onde processamos a entrada do usuário
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    precise_answer = get_precise_answer(user_input)
    
    if precise_answer:
        full_response = precise_answer
        with st.chat_message("assistant"):
            st.markdown(full_response)
    else:
        relevant_docs = get_relevant_documents(user_input)
        context = [
            {"role": "system", "content": "Você é um assistente AI especializado em informações sobre nossa empresa, funcionários e serviços bancários. Use os dados fornecidos para responder às perguntas com precisão, detalhe e contexto. Sempre forneça respostas completas e abrangentes."},
            {"role": "system", "content": f"Informações relevantes: {' '.join(relevant_docs)}"},
            {"role": "system", "content": 'df_dummy é a tabela de funcionários com as informações e sigla de cada, df_auxiliar é a tabela de ajuda para compreender, df_correlacao é a correlação entre os arquivos, df_amarracoes são as amarrações entre os arquivos'},
            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        ]
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            for response in client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=context,
                max_tokens=8000,
                temperature=0.3,
                top_p=0.9,
                frequency_penalty=0.2,
                presence_penalty=0.1,
                stream=True
            ):
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    logging.info(f"User Input: {user_input}")
    logging.info(f"Bot Response: {full_response}")

# Adicione esta seção para depuração
st.sidebar.header("Depuração")
if st.sidebar.checkbox("Mostrar última query e resposta"):
    st.sidebar.write("Última query:", user_input)
    st.sidebar.write("Última resposta:", full_response)

if st.button("Limpar Histórico"):
    st.session_state.messages = []
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

# Opção para visualizar tabelas
table_option = st.sidebar.selectbox(
    "Selecione uma tabela para visualizar:",
    ["Funcionários", "Siglas", "Correlações", "Amarrações"]
)

if table_option == "Funcionários":
    st.sidebar.dataframe(df_dummy)
elif table_option == "Siglas":
    st.sidebar.dataframe(df_auxiliar)
elif table_option == "Correlações":
    st.sidebar.dataframe(df_correlacao)
elif table_option == "Amarrações":
    st.sidebar.dataframe(df_amarracoes)

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

# Adicionar informações de depuração
st.sidebar.header("Informações de Depuração")
st.sidebar.write(f"Total de documentos no corpus: {len(documents)}")
st.sidebar.write(f"Total de funcionários: {len(df_dummy)}")
st.sidebar.write(f"Total de siglas: {len(df_auxiliar)}")
st.sidebar.write(f"Total de correlações: {len(df_correlacao)}")
st.sidebar.write(f"Total de amarrações: {len(df_amarracoes)}")

# Fim do código
