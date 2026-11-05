# Etapa 2: CART (DecisionTreeRegressor) com validação cruzada k-fold em
# 3 hiperparametrizações: subajustado (U), equilibrado (E) e sobreajustado (O).

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from config import SEED, K_FOLDS, PASTA_RESULTADOS
from validacao import (carregar_treino, validacao_cruzada, imprimir_folds,
                       tabela_resultados, folds_dataframe, media_dpa)

# Hiperparametrizações (as demais ficam no default do scikit-learn)
HIPERPARAMETROS_CART = {
    'CART U': {'max_depth': 2,    'min_samples_leaf': 1,  'min_samples_split': 2,
               'criterion': 'squared_error'},
    'CART E': {'max_depth': 10,   'min_samples_leaf': 20, 'min_samples_split': 2,
               'criterion': 'squared_error'},
    'CART O': {'max_depth': None, 'min_samples_leaf': 1,  'min_samples_split': 2,
               'criterion': 'squared_error'},
}


def criar_cart(params):
    return DecisionTreeRegressor(random_state=SEED, **params)


def main():
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    X, y = carregar_treino()
    print(f"Dataset: {X.shape[0]} vítimas, {X.shape[1]} features, k={K_FOLDS} folds")

    resultados = {}
    for nome, params in HIPERPARAMETROS_CART.items():
        resultados[nome] = validacao_cruzada(criar_cart(params), X, y)
        imprimir_folds(f"{nome} {params}", resultados[nome])

    # Tabela 2: hiperparametrizações
    tab_hp = pd.DataFrame(HIPERPARAMETROS_CART).rename(index={
        'min_samples_leaf': 'Min_samples_leaf', 'max_depth': 'Max_depth',
        'min_samples_split': 'Min_samples_split', 'criterion': 'Criterion'})
    tab_hp = tab_hp.fillna('None').rename_axis('MODELO')
    tab_hp.to_csv(PASTA_RESULTADOS / "etapa2_cart_hiperparametros.csv")

    # Tabela 3: resultados
    tab_res = tabela_resultados(resultados)
    tab_res.to_csv(PASTA_RESULTADOS / "etapa2_cart_resultados.csv")
    folds_dataframe(resultados).to_csv(PASTA_RESULTADOS / "etapa2_cart_folds.csv", index=False)

    print("\n===== TABELA 2: HIPERPARAMETRIZAÇÕES CART =====")
    print(tab_hp.to_string())
    print("\n===== TABELA 3: RESULTADOS DOS MODELOS CART =====")
    print(tab_res.to_string())

    melhor = min(resultados, key=lambda n: media_dpa(resultados[n]['validacao'])[0])
    print(f"\nMelhor CART (menor MSE médio de validação): {melhor}")

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
    ax.set_title(f'CART: MSE de treino x validação ({K_FOLDS}-fold)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "etapa2_cart_mse.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
