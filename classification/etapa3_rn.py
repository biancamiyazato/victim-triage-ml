# Etapa 3: treina a Rede Neural MLP com validação cruzada usando 3 hiperparametrizações:
# subajustada (U), equilibrada (E) e sobreajustada (O).
# As entradas são padronizadas (StandardScaler) dentro do pipeline, de modo que o
# scaler é ajustado apenas nos folds de treino (sem vazamento para a validação).

import os
import warnings

# Silencia ConvergenceWarning no processo principal e nos workers do n_jobs=-1
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import SEED, N_FOLDS, TREINO_CSV, RESULTADOS_DIR, FEATURES, TARGET
from validacao_cruzada import avaliar_cv, imprimir_resultado, imprimir_tabela, salvar_json

HIPERPARAMETROS_RN = {
    # 1 camada oculta com só 2 neurônios, sgd lento e poucas épocas: não aprende as fronteiras
    "U": {"hidden_layer_sizes": (2,), "activation": "logistic", "learning_rate_init": 0.001,
          "solver": "sgd", "alpha": 0.0001, "max_iter": 100, "early_stopping": False},
    # rede pequena com regularização L2 e parada antecipada: generaliza bem
    "E": {"hidden_layer_sizes": (32, 16), "activation": "relu", "learning_rate_init": 0.001,
          "solver": "adam", "alpha": 0.001, "max_iter": 500, "early_stopping": True},
    # rede grande, sem regularização e treinada até decorar o treino
    "O": {"hidden_layer_sizes": (256, 256, 256), "activation": "relu", "learning_rate_init": 0.001,
          "solver": "adam", "alpha": 0.0, "max_iter": 1000, "early_stopping": False,
          "tol": 1e-6, "n_iter_no_change": 50},
}


def criar_rn(params):
    return make_pipeline(StandardScaler(), MLPClassifier(random_state=SEED, **params))


if __name__ == "__main__":
    df = pd.read_csv(TREINO_CSV)
    X, y = df[FEATURES], df[TARGET]

    resultados = {}
    for nome, params in HIPERPARAMETROS_RN.items():
        print(f"\nTreinando RN {nome}... (O pode levar alguns minutos)", flush=True)
        resultados[nome] = avaliar_cv(criar_rn(params), X, y)
        imprimir_resultado(f"RN {nome} {params}", resultados[nome])

    print("\n===== TABELA 4: HIPERPARAMETRIZAÇÕES REDE NEURAL =====")
    tabela = pd.DataFrame(HIPERPARAMETROS_RN).rename(index={
        "hidden_layer_sizes": "Topologia", "activation": "Função de ativação",
        "learning_rate_init": "Learning rate", "solver": "solver"})
    tabela.loc["Topologia"] = [str(list(t)) for t in tabela.loc["Topologia"]]
    print(tabela.fillna("-").to_string())
    imprimir_tabela("TABELA 5: RESULTADOS DOS MODELOS RN", resultados)

    salvar_json(RESULTADOS_DIR / "etapa3_rn.json", {
        "n_folds": N_FOLDS,
        "preprocessamento": "StandardScaler",
        "hiperparametros": HIPERPARAMETROS_RN,
        "resultados": resultados,
    })
    print(f"\nResultados salvos em {RESULTADOS_DIR / 'etapa3_rn.json'}")
