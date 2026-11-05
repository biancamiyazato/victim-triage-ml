# Configurações compartilhadas entre as etapas da Tarefa 2 (Regressão de sobr)

from pathlib import Path

SEED = 42

# Dataset de treinamento/validação (etapa 1)
PASTA_TREINO = Path("./datasets/vict/10000v")
CSV_TREINO = PASTA_TREINO / "data.csv"

# Parâmetros de geração do dataset de treinamento/validação
DISTRIB_TRI = {
    0: 2500,  # Verde
    1: 2500,  # Amarelo
    2: 2500,  # Vermelho
    3: 2500,  # Preto
}
MEDIA_IDADE = 40
DESVIO_IDADE = 25
NIVEL_RUIDO = 0.05

# Dataset de teste cego (usado SOMENTE na etapa 6).
# Parâmetros do __main__ de gerar_dados_vitimas.py; semente distinta da do treino.
PASTA_TESTE = Path("./datasets/vict/1300v")
CSV_TESTE = PASTA_TESTE / "data.csv"
DISTRIB_TRI_TESTE = {0: 100, 1: 390, 2: 405, 3: 405}
SEED_TESTE = 2026

# Validação cruzada
K_FOLDS = 5

# Variáveis de entrada (1 a 10) e saída (14).
# gcs, avpu e tri (11, 12 e 13) NÃO podem ser usadas.
FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']
TARGET = 'sobr'

# Resultados
PASTA_RESULTADOS = Path("./resultados")
