import os
from groq import Groq

def configure_groq_client():
    """
    Configura o cliente Groq usando a chave da API.
    """
    api_key = os.getenv("GROQ_API_KEY", "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")
    
    if not api_key:
        raise ValueError("A chave da API do Groq não está definida. Defina a variável de ambiente GROQ_API_KEY ou forneça a chave diretamente no código.")
    
    return Groq(api_key=api_key)

def analyze_python_code(file_path, client):
    """
    Analisa um arquivo Python usando a API do Groq com o modelo LLaMA.
    
    :param file_path: Caminho para o arquivo Python a ser analisado
    :param client: Cliente Groq configurado
    :return: String contendo a análise detalhada do código
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            code = file.read()

    prompt = f"""Analise o seguinte código Python e forneça uma explicação detalhada, etapa por etapa, do que ele está fazendo. 
    Inclua comentários sobre a funcionalidade de cada parte do código, analise todo o código, preciso que seja feita as resposta visando ajudar posteriormente um chatbot para responder perguntas e respostas sobre o código analisado:

    {code}

    Explicação detalhada:"""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Você é um especialista em análise de código Python. Forneça explicações detalhadas e técnicas."},
            {"role": "system", "content": "Comente o código intiero com explicação do que ele está fazendo em cada etapa detalhada."},
            {"role": "system", "content": "Ao análisar o código crie respostas bem direcionadas o passo a passo de cada etapa do código"},
            {"role": "system", "content": "O máximo de tokens e tamanho de resposta possível."},
            {"role": "system", "content": "Faça uma explicação completa visando ser utilizado esse texto por um chatbot de perguntas e respostas, quando perguntar sobre algo no código, essa resposta ira auxiliar a IA para responder."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-70b-versatile",
        temperature=0.5,
        max_tokens=8000,
        top_p=1,
        stream=False
    )

    return chat_completion.choices[0].message.content

def save_analysis_to_file(analysis, output_path):
    """
    Salva a análise gerada em um arquivo .txt.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(analysis)
    except IOError as e:
        raise Exception(f"Erro ao salvar o arquivo: {str(e)}")

if __name__ == "__main__":
    try:
        client = configure_groq_client()
        
        source_file = input("Digite o caminho completo do arquivo Python a ser analisado: ")
        output_file = input("Digite o caminho completo para salvar a análise: ")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        analysis = analyze_python_code(source_file, client)
        save_analysis_to_file(analysis, output_file)
        
        print(f"Análise concluída e salva em: {output_file}")
        
    except ValueError as ve:
        print(f"Erro de configuração: {str(ve)}")
        print("Certifique-se de que a chave da API está correta e tente novamente.")
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
