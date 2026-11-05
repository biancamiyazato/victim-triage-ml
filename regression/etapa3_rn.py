# Etapa 3: Rede Neural MLP (MLPRegressor) com validação cruzada k-fold em
# 3 hiperparametrizações: subajustado (U), equilibrado (E) e sobreajustado (O).
# O StandardScaler fica dentro do Pipeline, sendo ajustado apenas com os dados
# de treino de cada fold (sem vazamento para a validação).

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import SEED, K_FOLDS, PASTA_RESULTADOS
from validacao import (carregar_treino, validacao_cruzada, imprimir_folds,
                       tabela_resultados, folds_dataframe, media_dpa)

# Hiperparametrizações (as demais ficam no default do scikit-learn)
HIPERPARAMETROS_RN = {
    'RN U': {'hidden_layer_sizes': (1,), 'activation': 'identity', 'solver': 'sgd',
             'learning_rate_init': 1e-3, 'alpha': 1e-4, 'max_iter': 100,
             'early_stopping': False},
    'RN E': {'hidden_layer_sizes': (32, 16), 'activation': 'tanh', 'solver': 'adam',
             'learning_rate_init': 1e-3, 'alpha': 1e-3, 'max_iter': 500,
             'early_stopping': True},
    'RN O': {'hidden_layer_sizes': (128, 128), 'activation': 'relu', 'solver': 'adam',
             'learning_rate_init': 1e-3, 'alpha': 0.0, 'max_iter': 300,
             'early_stopping': False, 'tol': 1e-9, 'n_iter_no_change': 300},
}


def criar_rn(params):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(random_state=SEED, **params)),
    ])


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    X, y = carregar_treino()
    print(f"Dataset: {X.shape[0]} vítimas, {X.shape[1]} features, k={K_FOLDS} folds")

    resultados = {}
    for nome, params in HIPERPARAMETROS_RN.items():
        resultados[nome] = validacao_cruzada(criar_rn(params), X, y)
        imprimir_folds(f"{nome} {params}", resultados[nome])

    # Tabela 4: hiperparametrizações
    tab_hp = pd.DataFrame({
        nome: {
            'Topologia': '[' + ' '.join(map(str, p['hidden_layer_sizes'])) + ']',
            'Função de ativação': p['activation'],
            'Learning rate': p['learning_rate_init'],
            'solver': p['solver'],
            'alpha (L2)': p['alpha'],
            'max_iter': p['max_iter'],
            'early_stopping': p['early_stopping'],
            'scaler': 'StandardScaler',
        } for nome, p in HIPERPARAMETROS_RN.items()
    }).rename_axis('MODELO')
    tab_hp.to_csv(PASTA_RESULTADOS / "etapa3_rn_hiperparametros.csv")

    # Tabela 5: resultados
    tab_res = tabela_resultados(resultados)
    tab_res.to_csv(PASTA_RESULTADOS / "etapa3_rn_resultados.csv")
    folds_dataframe(resultados).to_csv(PASTA_RESULTADOS / "etapa3_rn_folds.csv", index=False)

    print("\n===== TABELA 4: HIPERPARAMETRIZAÇÕES REDE NEURAL =====")
    print(tab_hp.to_string())
    print("\n===== TABELA 5: RESULTADOS DOS MODELOS RN =====")
    print(tab_res.to_string())

    melhor = min(resultados, key=lambda n: media_dpa(resultados[n]['validacao'])[0])
    print(f"\nMelhor RN (menor MSE médio de validação): {melhor}")

    # Gráfico: MSE médio de treino x validação por hiperparametrização
    nomes = list(resultados)
    pos = np.arange(len(nomes))
    larg = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    for desl, chave, rotulo, cor in [(-larg / 2, 'treino', 'Treino', '#4C78A8'),
                                     (larg / 2, 'validacao', 'Validação', '#F58518')]:
        medias = [media_dpa(resultados[n][chave])[0] for n in nomes]
        dpas = [media_dpa(resultados[n][chave])[1] for n in nomes]
        ax.bar(pos + desl, medias, larg, yerr=dpas, capsize=4, label=rotulo, color=cor)
    ax.set_xticks(pos, nomes)
    ax.set_ylabel('MSE médio (k folds)')
    ax.set_title(f'RN MLP: MSE de treino x validação ({K_FOLDS}-fold)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "etapa3_rn_mse.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
