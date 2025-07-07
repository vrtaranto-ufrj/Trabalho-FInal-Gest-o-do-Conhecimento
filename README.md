
# Comparação de Algoritmos de Agrupamento

Este repositório contém as implementações "do zero" dos algoritmos de agrupamento **K-means**, **DBSCAN** e **SOM** (Mapas Auto-Organizáveis), como parte do trabalho final da disciplina de *Gestão do Conhecimento*.

## Projeto Inclui

- `kmeans.py`: Implementação do algoritmo K-means.
- `dbscan.py`: Implementação do algoritmo DBSCAN.
- `som.py`: Implementação do algoritmo SOM.
- `utils.py`: Funções auxiliares para carregar dados e plotar gráficos.
- `analise_experimentos.py`: Script principal para executar todos os testes, comparar os algoritmos com diferentes parâmetros e gerar os gráficos para o relatório.
- `iris/`: Pasta contendo o Iris Dataset.
- `graficos/`: Pasta onde os gráficos gerados pela análise são salvos.

## Como Rodar o Projeto

### 1. Pré-requisitos

- Python 3.8 ou superior

### 2. Configuração do Ambiente

Primeiro, clone o repositório e navegue até a pasta do projeto.

É altamente recomendado criar um ambiente virtual (venv) para isolar as dependências do projeto:

```bash
# Criar o ambiente virtual
python -m venv .venv

# Ativar o ambiente virtual
# No Windows:
.venv\Scripts\activate
# No macOS/Linux:
source .venv/bin/activate
```

### 3. Instalação das Dependências

Com o ambiente virtual ativado, instale as bibliotecas necessárias usando o arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Executando os Experimentos

#### Opção A: Gerar todos os gráficos e análises de uma vez

Para executar a análise completa, que testa todos os algoritmos com diferentes parâmetros e salva os gráficos na pasta `graficos/`, rode o script principal:

```bash
python analise_experimentos.py
```

#### Opção B: Rodar cada algoritmo individualmente

Você também pode executar cada arquivo de modelo separadamente para ver um exemplo de saída para aquele algoritmo específico:

```bash
# Para testar o K-means
python kmeans.py

# Para testar o DBSCAN
python dbscan.py

# Para testar o SOM
python som.py
```
