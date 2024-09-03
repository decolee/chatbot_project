import os
import json
import yaml
from typing import List, Dict, Any, Union
import hashlib
import random
import re
import docx2txt

def read_file(file_path: str) -> Union[dict, str]:
    print(f"Lendo arquivo: {file_path}")
    _, file_extension = os.path.splitext(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        if file_extension in ['.yml', '.yaml']:
            return yaml.safe_load(file)
        elif file_extension == '.json':
            return json.load(file)
        elif file_extension in ['.py', '.txt']:
            return file.read()
        elif file_extension in ['.docx', '.doc']:
            return docx2txt.process(file_path)
        else:
            return file.read()
def get_service_functionality(service: str, config: dict) -> str:
    functionality = ""
    if 'image' in config:
        image = config['image']
        if 'postgres' in image:
            functionality = "um banco de dados PostgreSQL para armazenamento persistente de dados"
        elif 'redis' in image:
            functionality = "um cache em memória Redis para melhorar o desempenho de operações frequentes"
        elif 'nginx' in image:
            functionality = "um servidor web Nginx para servir conteúdo estático e atuar como proxy reverso"
        elif 'mongo' in image:
            functionality = "um banco de dados MongoDB para armazenamento de dados não-relacionais"
        elif 'elasticsearch' in image:
            functionality = "um mecanismo de busca e análise Elasticsearch para indexação e consulta de dados"
        elif 'rabbitmq' in image:
            functionality = "um message broker RabbitMQ para gerenciar filas e mensagens entre serviços"
        else:
            functionality = f"funcionalidades específicas baseadas na imagem {image}"
    elif 'build' in config:
        functionality = "funcionalidades personalizadas definidas no Dockerfile especificado"
    else:
        functionality = "funcionalidades específicas do serviço que não podem ser inferidas diretamente da configuração"
    
    return functionality

def generate_questions(file_content: Union[dict, str], iteration: int) -> List[Dict]:
    print(f"Gerando perguntas - Iteração {iteration}")
    
    if isinstance(file_content, dict):
        return generate_docker_compose_questions(file_content, iteration)
    else:
        return generate_text_questions(file_content, iteration)

def generate_docker_compose_questions(file_content: dict, iteration: int) -> List[Dict]:
    questions = []
    services = file_content.get('services', {})
    
    for service, config in services.items():
        questions.extend(generate_service_questions(service, config, iteration))
    
    questions.extend(generate_service_relation_questions(services, iteration))
    questions.extend(generate_general_questions(file_content, iteration))
    questions.extend(generate_best_practices_questions(file_content, iteration))
    
    print(f"Geradas {len(questions)} perguntas estruturadas na iteração {iteration}")
    return questions


def generate_code_analysis_questions(text_content: str, iteration: int) -> List[Dict]:
    questions = []
    
    lines = text_content.split('\n')
    
    code_purpose_question = {
        "pergunta": "Qual parece ser o propósito principal deste código?",
        "resposta": f"Com base na análise do código, o propósito principal parece ser {infer_code_purpose(text_content)}.",
        "tags": ["análise de código", "propósito"],
        "dificuldade": "médio",
        "categoria": "compreensão de código",
        "id": hashlib.md5(f"code_purpose_{iteration}".encode()).hexdigest()
    }
    questions.append(code_purpose_question)
    
    complexity_question = {
        "pergunta": "Qual é a complexidade aparente deste código?",
        "resposta": f"A complexidade aparente deste código é {assess_code_complexity(text_content)}.",
        "tags": ["análise de código", "complexidade"],
        "dificuldade": "difícil",
        "categoria": "métricas de código",
        "id": hashlib.md5(f"code_complexity_{iteration}".encode()).hexdigest()
    }
    questions.append(complexity_question)
    
    improvement_question = {
        "pergunta": "Quais são algumas possíveis melhorias ou otimizações para este código?",
        "resposta": f"Algumas possíveis melhorias ou otimizações para este código incluem: {suggest_code_improvements(text_content)}",
        "tags": ["análise de código", "otimização", "melhores práticas"],
        "dificuldade": "expert",
        "categoria": "refatoração de código",
        "id": hashlib.md5(f"code_improvements_{iteration}".encode()).hexdigest()
    }
    questions.append(improvement_question)
    
    return questions

def infer_code_purpose(text_content: str) -> str:
    # Implemente a lógica para inferir o propósito do código
    # Esta é uma implementação simplificada
    if "def main" in text_content:
        return "um programa principal com várias funções"
    elif "class" in text_content:
        return "definir uma ou mais classes"
    elif "import" in text_content and "def" in text_content:
        return "um módulo com funções importáveis"
    else:
        return "um script com várias operações"

def assess_code_complexity(text_content: str) -> str:
    # Implemente a lógica para avaliar a complexidade do código
    # Esta é uma implementação simplificada
    lines = text_content.split('\n')
    if len(lines) < 50:
        return "baixa"
    elif len(lines) < 200:
        return "média"
    else:
        return "alta"

def suggest_code_improvements(text_content: str) -> str:
    # Implemente a lógica para sugerir melhorias no código
    # Esta é uma implementação simplificada
    suggestions = []
    if "global" in text_content:
        suggestions.append("reduzir o uso de variáveis globais")
    if text_content.count("for") > 5:
        suggestions.append("otimizar loops para melhor performance")
    if "print" in text_content:
        suggestions.append("substituir prints por logging para melhor depuração")
    return ", ".join(suggestions) if suggestions else "nenhuma melhoria óbvia identificada"


def generate_text_questions(text_content: str, iteration: int) -> List[Dict]:
    questions = []
    
    lines = text_content.split('\n')
    
    total_lines_question = {
        "pergunta": "Quantas linhas de código ou texto este arquivo contém?",
        "resposta": f"Este arquivo contém {len(lines)} linhas de código ou texto.",
        "tags": ["análise de código", "estatísticas"],
        "dificuldade": "fácil",
        "categoria": "métricas de código",
        "id": hashlib.md5(f"total_lines_{iteration}".encode()).hexdigest()
    }
    questions.append(total_lines_question)
    
    purpose_question = {
        "pergunta": "Qual parece ser o propósito geral deste arquivo com base em seu conteúdo?",
        "resposta": f"Com base no conteúdo, este arquivo parece {infer_file_purpose(text_content)}.",
        "tags": ["análise de código", "propósito"],
        "dificuldade": "médio",
        "categoria": "compreensão de código",
        "id": hashlib.md5(f"file_purpose_{iteration}".encode()).hexdigest()
    }
    questions.append(purpose_question)
    
    # Adicione as perguntas de análise de código se o arquivo parecer conter código
    if "def " in text_content or "class " in text_content or "import " in text_content:
        questions.extend(generate_code_analysis_questions(text_content, iteration))
    
    return questions

def infer_file_purpose(text_content: str) -> str:
    if "def " in text_content and "class " in text_content:
        return "conter definições de classes e funções em Python"
    elif "import " in text_content and "def " in text_content:
        return "ser um módulo Python com importações e definições de funções"
    elif "<html>" in text_content.lower():
        return "ser um arquivo HTML"
    elif "select " in text_content.lower() and "from " in text_content.lower():
        return "conter consultas SQL"
    else:
        return "conter texto ou código genérico"

def get_service_purpose(service: str, config: dict) -> str:
    if 'image' in config:
        return f"executar a imagem Docker '{config['image']}'"
    elif 'build' in config:
        return f"construir e executar um contêiner personalizado"
    else:
        return "fornecer funcionalidades específicas no ambiente Docker"


def generate_service_questions(service: str, config: dict, iteration: int) -> List[Dict]:
    questions = []
    
    purpose_question = {
        "pergunta": f"Qual é o propósito principal do serviço '{service}' no contexto deste arquivo docker-compose?",
        "resposta": f"O serviço '{service}' é definido para {get_service_purpose(service, config)}. " +
                    f"Ele desempenha um papel importante na arquitetura geral do sistema, " +
                    f"fornecendo {get_service_functionality(service, config)}.",
        "tags": [service, "docker-compose", "arquitetura", "propósito"],
        "dificuldade": "médio",
        "categoria": "arquitetura de serviços",
        "id": hashlib.md5(f"purpose_{service}_{iteration}".encode()).hexdigest()
    }
    questions.append(purpose_question)
    
    for key, value in config.items():
        questions.extend(generate_config_questions(service, key, value, iteration))
    
    if 'depends_on' in config:
        dep_question = {
            "pergunta": f"Quais são as dependências do serviço '{service}' e por que elas são necessárias?",
            "resposta": f"O serviço '{service}' depende de {', '.join(config['depends_on'])}. " +
                        f"Essas dependências são necessárias porque {get_dependency_reason(service, config['depends_on'])}.",
            "tags": [service, "dependências", "arquitetura"],
            "dificuldade": "difícil",
            "categoria": "relações entre serviços",
            "id": hashlib.md5(f"dependencies_{service}_{iteration}".encode()).hexdigest()
        }
        questions.append(dep_question)
    
    return questions

def generate_config_questions(service: str, key: str, value: Any, iteration: int) -> List[Dict]:
    questions = []
    
    if isinstance(value, (str, int, float, bool)):
        config_question = {
            "pergunta": f"Qual é a configuração de '{key}' para o serviço '{service}' e qual seu impacto no funcionamento do contêiner?",
            "resposta": f"A configuração de '{key}' para o serviço '{service}' é '{value}'. " +
                        f"Isso afeta o contêiner da seguinte forma: {get_config_impact(key, value)}.",
            "tags": [service, key, "configuração", "impacto"],
            "dificuldade": "médio",
            "categoria": "configuração de serviço",
            "id": hashlib.md5(f"{service}_{key}_{iteration}".encode()).hexdigest()
        }
        questions.append(config_question)
    elif isinstance(value, list):
        list_question = {
            "pergunta": f"Quais são os valores configurados para '{key}' no serviço '{service}' e como eles interagem com o ambiente Docker?",
            "resposta": f"Os valores configurados para '{key}' no serviço '{service}' são: {', '.join(map(str, value))}. " +
                        f"Esses valores {get_list_config_impact(key, value)} no ambiente Docker.",
            "tags": [service, key, "configuração", "lista", "ambiente Docker"],
            "dificuldade": "difícil",
            "categoria": "configuração avançada de serviço",
            "id": hashlib.md5(f"{service}_{key}_list_{iteration}".encode()).hexdigest()
        }
        questions.append(list_question)
    elif isinstance(value, dict):
        dict_question = {
            "pergunta": f"Quais são as configurações detalhadas de '{key}' para o serviço '{service}' e como elas contribuem para a funcionalidade do serviço?",
            "resposta": f"As configurações detalhadas de '{key}' para o serviço '{service}' são: {json.dumps(value, indent=2)}. " +
                        f"Essas configurações são importantes porque {get_dict_config_importance(key, value)}.",
            "tags": [service, key, "configuração", "detalhada", "funcionalidade"],
            "dificuldade": "expert",
            "categoria": "configuração avançada de serviço",
            "id": hashlib.md5(f"{service}_{key}_dict_{iteration}".encode()).hexdigest()
        }
        questions.append(dict_question)
    
    return questions

def generate_service_relation_questions(services: dict, iteration: int) -> List[Dict]:
    questions = []
    
    if len(services) > 1:
        service_names = list(services.keys())
        random.shuffle(service_names)
        service1, service2 = service_names[:2]
        
        relation_question = {
            "pergunta": f"Como os serviços '{service1}' e '{service2}' interagem entre si na arquitetura definida por este docker-compose?",
            "resposta": f"Os serviços '{service1}' e '{service2}' interagem da seguinte forma: {get_service_interaction(service1, service2, services)}. " +
                        f"Esta interação é crucial para {get_interaction_importance(service1, service2)}.",
            "tags": [service1, service2, "interação", "arquitetura"],
            "dificuldade": "expert",
            "categoria": "arquitetura de microsserviços",
            "id": hashlib.md5(f"interaction_{service1}_{service2}_{iteration}".encode()).hexdigest()
        }
        questions.append(relation_question)
    
    return questions

def generate_general_questions(file_content: dict, iteration: int) -> List[Dict]:
    questions = []
    
    services = file_content.get('services', {})
    networks = file_content.get('networks', {})
    volumes = file_content.get('volumes', {})

    structure_question = {
        "pergunta": "Qual é a estrutura geral deste arquivo docker-compose e como ela reflete a arquitetura do sistema?",
        "resposta": f"Este arquivo docker-compose define {len(services)} serviços, {len(networks)} redes e {len(volumes)} volumes. " +
                    f"A arquitetura refletida por esta estrutura é {get_architecture_description(services, networks, volumes)}. " +
                    f"Isso sugere um sistema {get_system_complexity(services)} com {get_scalability_assessment(services)}.",
        "tags": ["docker-compose", "arquitetura", "estrutura", "complexidade"],
        "dificuldade": "difícil",
        "categoria": "visão geral da arquitetura",
        "id": hashlib.md5(f"structure_overview_{iteration}".encode()).hexdigest()
    }
    questions.append(structure_question)

    scalability_question = {
        "pergunta": "Como este docker-compose aborda questões de escalabilidade e resiliência?",
        "resposta": f"Este docker-compose aborda escalabilidade e resiliência através de {get_scalability_features(services)}. " +
                    f"Além disso, {get_resilience_features(services, networks)} são implementados para aumentar a robustez do sistema.",
        "tags": ["escalabilidade", "resiliência", "docker-compose", "arquitetura"],
        "dificuldade": "expert",
        "categoria": "design de sistemas distribuídos",
        "id": hashlib.md5(f"scalability_resilience_{iteration}".encode()).hexdigest()
    }
    questions.append(scalability_question)

    security_question = {
        "pergunta": "Quais aspectos de segurança são abordados neste docker-compose e como eles poderiam ser melhorados?",
        "resposta": f"Os aspectos de segurança abordados neste docker-compose incluem {get_security_features(services, networks)}. " +
                    f"Possíveis melhorias incluem: {get_security_improvements(services, networks)}.",
        "tags": ["segurança", "docker-compose", "melhores práticas"],
        "dificuldade": "expert",
        "categoria": "segurança de contêineres",
        "id": hashlib.md5(f"security_aspects_{iteration}".encode()).hexdigest()
    }
    questions.append(security_question)

    return questions

def generate_best_practices_questions(file_content: dict, iteration: int) -> List[Dict]:
    questions = []
    
    services = file_content.get('services', {})

    best_practices_question = {
        "pergunta": "Quais melhores práticas de Docker e docker-compose são seguidas neste arquivo, e quais poderiam ser adicionadas?",
        "resposta": f"As melhores práticas seguidas neste docker-compose incluem: {get_followed_best_practices(services)}. " +
                    f"Práticas adicionais que poderiam ser implementadas são: {get_additional_best_practices(services)}.",
        "tags": ["docker", "docker-compose", "melhores práticas", "otimização"],
        "dificuldade": "expert",
        "categoria": "otimização de configuração Docker",
        "id": hashlib.md5(f"best_practices_{iteration}".encode()).hexdigest()
    }
    questions.append(best_practices_question)

    resource_optimization_question = {
        "pergunta": "Como os recursos são alocados e otimizados neste docker-compose, e quais estratégias poderiam melhorar a eficiência?",
        "resposta": f"Neste docker-compose, os recursos são alocados da seguinte forma: {get_resource_allocation(services)}. " +
                    f"Estratégias para melhorar a eficiência incluem: {get_efficiency_strategies(services)}.",
        "tags": ["recursos", "otimização", "eficiência", "docker-compose"],
        "dificuldade": "expert",
        "categoria": "gerenciamento de recursos em contêineres",
        "id": hashlib.md5(f"resource_optimization_{iteration}".encode()).hexdigest()
    }
    questions.append(resource_optimization_question)

    return questions

# Funções auxiliares (implementações omitidas para brevidade)

def save_questions(questions: List[Dict], output_file: str):
    print(f"Salvando perguntas em: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(questions, file, indent=2, ensure_ascii=False)

def main():
    file_path = input("Digite o caminho do arquivo a ser analisado: ")
    output_file = input("Digite o nome do arquivo de saída para as perguntas: ")
    num_iterations = int(input("Digite o número de iterações para refinamento (recomendado: 1-10): "))

    file_content = read_file(file_path)
    all_questions = []

    for iteration in range(1, num_iterations + 1):
        print(f"Iniciando iteração {iteration} de {num_iterations}")
        new_questions = generate_questions(file_content, iteration)
        all_questions.extend(new_questions)

    unique_questions = list({q['id']: q for q in all_questions}.values())

    save_questions(unique_questions, output_file)
    print(f"Perguntas geradas e salvas em {output_file}")
    print(f"Total de perguntas únicas geradas: {len(unique_questions)}")

if __name__ == "__main__":
    main()

# Implementações das funções auxiliares

def get_architecture_description(services: dict, networks: dict, volumes: dict) -> str:
    architecture_type = "monolítica" if len(services) == 1 else "de microsserviços"
    network_type = "isolada" if not networks else "interconectada"
    data_management = "com persistência de dados" if volumes else "sem persistência de dados explícita"
    
    return f"uma arquitetura {architecture_type} {network_type} {data_management}"

def get_system_complexity(services: dict) -> str:
    if len(services) <= 2:
        return "relativamente simples"
    elif len(services) <= 5:
        return "de complexidade moderada"
    else:
        return "complexo e potencialmente desafiador de gerenciar"

def get_scalability_assessment(services: dict) -> str:
    scalable_services = sum(1 for service in services.values() if 'deploy' in service)
    if scalable_services == len(services):
        return "alta capacidade de escalabilidade em todos os serviços"
    elif scalable_services > 0:
        return f"potencial de escalabilidade em {scalable_services} de {len(services)} serviços"
    else:
        return "limitações de escalabilidade, pois nenhum serviço tem configurações explícitas de implantação"

def get_scalability_features(services: dict) -> str:
    features = []
    for service, config in services.items():
        if 'deploy' in config:
            if 'replicas' in config['deploy']:
                features.append(f"replicação do serviço {service}")
            if 'update_config' in config['deploy']:
                features.append(f"configuração de atualização para {service}")
    
    if not features:
        return "não há características explícitas de escalabilidade definidas"
    
    return ", ".join(features)

def get_resilience_features(services: dict, networks: dict) -> str:
    features = []
    for service, config in services.items():
        if 'restart' in config:
            features.append(f"política de reinicialização para {service}")
        if 'healthcheck' in config:
            features.append(f"verificação de saúde para {service}")
    
    if networks:
        features.append("uso de redes personalizadas para isolamento e segurança")
    
    if not features:
        return "não há características explícitas de resiliência definidas"
    
    return ", ".join(features)

def get_security_features(services: dict, networks: dict) -> str:
    features = []
    for service, config in services.items():
        if 'secrets' in config:
            features.append(f"uso de secrets para {service}")
        if 'user' in config:
            features.append(f"execução com usuário não-root para {service}")
    
    if networks:
        features.append("isolamento de rede através de redes personalizadas")
    
    if not features:
        return "não há características explícitas de segurança definidas"
    
    return ", ".join(features)

def get_security_improvements(services: dict, networks: dict) -> str:
    improvements = [
        "implementar o princípio do menor privilégio para todos os serviços",
        "utilizar secrets para todas as informações sensíveis",
        "configurar limites de recursos para prevenir ataques de negação de serviço",
        "implementar verificações de saúde para todos os serviços críticos",
        "utilizar redes personalizadas para isolar grupos de serviços"
    ]
    return ", ".join(improvements)

def get_followed_best_practices(services: dict) -> str:
    practices = []
    for service, config in services.items():
        if 'image' in config and ':' in config['image']:
            practices.append("uso de tags específicas para imagens")
        if 'healthcheck' in config:
            practices.append("implementação de verificações de saúde")
        if 'restart' in config:
            practices.append("configuração de políticas de reinicialização")
    
    if not practices:
        return "não foram identificadas práticas recomendadas explícitas"
    
    return ", ".join(practices)

def get_additional_best_practices(services: dict) -> str:
    practices = [
        "usar multi-stage builds para reduzir o tamanho das imagens",
        "implementar logging centralizado",
        "utilizar secrets para gerenciar informações sensíveis",
        "configurar limites de recursos para todos os serviços",
        "implementar verificações de saúde para todos os serviços",
        "usar redes personalizadas para isolar grupos de serviços",
        "implementar políticas de reinicialização adequadas",
        "utilizar volumes nomeados em vez de bind mounts",
        "minimizar o número de camadas nas imagens Docker",
        "usar a diretiva 'depends_on' para gerenciar dependências entre serviços"
    ]
    return ", ".join(practices)

def get_resource_allocation(services: dict) -> str:
    allocations = []
    for service, config in services.items():
        if 'deploy' in config and 'resources' in config['deploy']:
            resources = config['deploy']['resources']
            limits = resources.get('limits', {})
            reservations = resources.get('reservations', {})
            allocation = f"Para {service}: "
            if limits:
                allocation += f"limites de {', '.join([f'{k}={v}' for k, v in limits.items()])}; "
            if reservations:
                allocation += f"reservas de {', '.join([f'{k}={v}' for k, v in reservations.items()])}"
            allocations.append(allocation)
    
    if not allocations:
        return "não há alocações de recursos explícitas definidas"
    
    return "; ".join(allocations)

def get_efficiency_strategies(services: dict) -> str:
    strategies = [
        "otimizar as imagens Docker para reduzir o tamanho e melhorar o tempo de inicialização",
        "implementar cache eficiente para builds e layers",
        "utilizar healthchecks para garantir que apenas serviços saudáveis recebam tráfego",
        "configurar limites de recursos apropriados para evitar sobre-alocação",
        "implementar estratégias de logging eficientes para reduzir o overhead de I/O",
        "utilizar redes overlay eficientes para comunicação entre serviços",
        "implementar balanceamento de carga para distribuir o tráfego de forma equitativa",
        "utilizar volumes para persistência de dados de forma eficiente",
        "otimizar consultas de banco de dados e implementar caching quando apropriado",
        "utilizar compressão de dados para reduzir o tráfego de rede"
    ]
    return ", ".join(strategies)

def get_service_functionality(service: str, config: dict) -> str:
    functionality = ""
    if 'image' in config:
        image = config['image']
        if 'postgres' in image:
            functionality = "um banco de dados PostgreSQL para armazenamento persistente de dados"
        elif 'redis' in image:
            functionality = "um cache em memória Redis para melhorar o desempenho de operações frequentes"
        elif 'nginx' in image:
            functionality = "um servidor web Nginx para servir conteúdo estático e atuar como proxy reverso"
        elif 'mongo' in image:
            functionality = "um banco de dados MongoDB para armazenamento de dados não-relacionais"
        elif 'elasticsearch' in image:
            functionality = "um mecanismo de busca e análise Elasticsearch para indexação e consulta de dados"
        elif 'rabbitmq' in image:
            functionality = "um message broker RabbitMQ para gerenciar filas e mensagens entre serviços"
        else:
            functionality = f"funcionalidades específicas baseadas na imagem {image}"
    elif 'build' in config:
        functionality = "funcionalidades personalizadas definidas no Dockerfile especificado"
    else:
        functionality = "funcionalidades específicas do serviço que não podem ser inferidas diretamente da configuração"
    
    return functionality

def get_dependency_reason(service: str, dependencies: List[str]) -> str:
    reasons = []
    for dep in dependencies:
        if 'database' in dep or 'db' in dep:
            reasons.append(f"precisa do serviço de banco de dados {dep} para persistência de dados")
        elif 'cache' in dep or 'redis' in dep:
            reasons.append(f"utiliza {dep} para caching e melhorar o desempenho")
        elif 'queue' in dep or 'rabbitmq' in dep:
            reasons.append(f"depende de {dep} para processamento assíncrono e comunicação entre serviços")
        elif 'auth' in dep:
            reasons.append(f"requer {dep} para autenticação e autorização")
        else:
            reasons.append(f"tem uma dependência funcional de {dep}")
    
    return " e ".join(reasons)

def get_config_impact(key: str, value: Any) -> str:
    impacts = {
        'ports': "permite a exposição do serviço para acesso externo",
        'environment': "configura variáveis de ambiente que afetam o comportamento do serviço",
        'volumes': "gerencia persistência de dados e compartilhamento de arquivos com o host",
        'networks': "define a conectividade e isolamento de rede do serviço",
        'depends_on': "estabelece a ordem de inicialização e dependências entre serviços",
        'deploy': "configura estratégias de implantação e recursos para o serviço",
        'healthcheck': "define como o Docker deve verificar a saúde do serviço",
        'restart': "especifica a política de reinicialização do contêiner em caso de falha"
    }
    return impacts.get(key, f"afeta o comportamento do serviço de maneira específica relacionada a {key}")

def get_list_config_impact(key: str, value: List[Any]) -> str:
    if key == 'ports':
        return f"expõem as portas {', '.join(map(str, value))} do contêiner, permitindo comunicação externa"
    elif key == 'volumes':
        return f"montam os volumes {', '.join(map(str, value))}, gerenciando persistência e compartilhamento de dados"
    elif key == 'networks':
        return f"conectam o serviço às redes {', '.join(map(str, value))}, definindo sua topologia de rede"
    else:
        return f"configuram múltiplos valores para {key}, afetando o comportamento do serviço de forma correspondente"

def get_dict_config_importance(key: str, value: dict) -> str:
    importance = []
    if key == 'environment':
        importance.append("define variáveis de ambiente cruciais para a configuração e comportamento do serviço")
    elif key == 'volumes':
        importance.append("gerencia o armazenamento persistente e compartilhamento de dados entre o host e o contêiner")
    elif key == 'ports':
        importance.append("controla o mapeamento de portas entre o host e o contêiner, permitindo a comunicação externa")
    elif key == 'deploy':
        importance.append("especifica configurações de implantação para orquestradores como Docker Swarm")
    else:
        importance.append("fornece configurações adicionais que afetam o comportamento e a integração do serviço")
    
    return " e ".join(importance)

def get_service_interaction(service1: str, service2: str, services: dict) -> str:
    interactions = []
    
    # Verificar dependências diretas
    if service2 in services.get(service1, {}).get('depends_on', []):
        interactions.append(f"{service1} depende diretamente de {service2}")
    elif service1 in services.get(service2, {}).get('depends_on', []):
        interactions.append(f"{service2} depende diretamente de {service1}")
    
    # Verificar redes compartilhadas
    networks1 = set(services.get(service1, {}).get('networks', []))
    networks2 = set(services.get(service2, {}).get('networks', []))
    shared_networks = networks1.intersection(networks2)
    if shared_networks:
        interactions.append(f"compartilham as seguintes redes: {', '.join(shared_networks)}")
    
    # Verificar volumes compartilhados
    volumes1 = set(services.get(service1, {}).get('volumes', []))
    volumes2 = set(services.get(service2, {}).get('volumes', []))
    shared_volumes = volumes1.intersection(volumes2)
    if shared_volumes:
        interactions.append(f"compartilham os seguintes volumes: {', '.join(shared_volumes)}")
    
    if not interactions:
        return "não têm uma interação direta aparente, mas podem se comunicar através da rede Docker padrão"
    
    return "; ".join(interactions)

def get_interaction_importance(service1: str, service2: str) -> str:
    return f"garantir a funcionalidade integrada e o fluxo de dados adequado entre os serviços {service1} e {service2}, contribuindo para a coesão e eficiência do sistema como um todo"

# Função principal
if __name__ == "__main__":
    main()
