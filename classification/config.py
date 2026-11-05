# Configurações compartilhadas por todas as etapas do Trabalho 1 (Classificação)

from pathlib import Path

SEED = 42                   # semente fixa para reprodutibilidade
N_FOLDS = 5                 # número de folds da validação cruzada (StratifiedKFold)

# Dataset de treinamento/validação (etapa 1)
TREINO_DIR = Path("./datasets/vict/10000v")
TREINO_CSV = TREINO_DIR / "data.csv"

# Dataset de teste cego: usar SOMENTE na etapa 5
TESTE_CEGO_CSV = Path("./datasets/vict/1300v/data.csv")

# Resultados das etapas (tabelas do relatório)
RESULTADOS_DIR = Path("./resultados")

# Entradas 1 a 10 (gcs, avpu, tri e sobr são proibidas como entrada)
FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']
TARGET = 'tri'
CLASSES = {0: 'verde', 1: 'amarelo', 2: 'vermelho', 3: 'preto'}
