# Etapa 5: retreina o melhor CART e a melhor RN (escolhidos na etapa 4) com
# todas as 10.000 vítimas do dataset de treinamento/validação, sem validação
# cruzada nem split, e salva os modelos para o teste cego (etapa 6).
#
# Saídas (pasta ./modelos):
#   melhor_cart.joblib  - DecisionTreeRegressor
#   melhor_rn.joblib    - MLPRegressor (espera entradas já padronizadas)
#   scaler.joblib       - StandardScaler ajustado nas 10.000 vítimas

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from config import SEED, FEATURES, PASTA_RESULTADOS
from etapa2_cart import HIPERPARAMETROS_CART, criar_cart
from etapa3_rn import HIPERPARAMETROS_RN
from validacao import carregar_treino, mse

PASTA_MODELOS = Path("./modelos")


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)

    with open(PASTA_RESULTADOS / "etapa4_melhores.json", encoding="utf-8") as f:
        melhores = json.load(f)
    nome_cart, nome_rn = melhores['cart'], melhores['rn']

    X, y = carregar_treino()
    print(f"Retreino com {X.shape[0]} vítimas e features {FEATURES}")

    # CART
    params_cart = HIPERPARAMETROS_CART[nome_cart]
    cart = criar_cart(params_cart).fit(X, y)
    joblib.dump(cart, PASTA_MODELOS / "melhor_cart.joblib")
    print(f"\n{nome_cart} {params_cart}")
    print(f"  folhas={cart.get_n_leaves()} profundidade={cart.get_depth()}")
    print(f"  MSE no treino completo: {mse(y, cart.predict(X)):.5f}")

    # RN (scaler salvo separadamente, como pede a entrega)
    params_rn = dict(HIPERPARAMETROS_RN[nome_rn])
    scaler = StandardScaler().fit(X)
    X_pad = scaler.transform(X)
    print(f"\n{nome_rn} {params_rn}")

    # Com early_stopping=True o MLPRegressor separa internamente 10% dos dados
    # para validação e ajusta os pesos com apenas 90%. Para que o fit final use
    # as 10.000 vítimas, o early stopping serve só para definir o número de
    # épocas (melhor época de validação); o modelo final é treinado sem ele.
    if params_rn.get('early_stopping'):
        rn_es = MLPRegressor(random_state=SEED, **params_rn).fit(X_pad, y)
        epocas = int(np.argmax(rn_es.validation_scores_)) + 1
        print(f"  early stopping: parou em {rn_es.n_iter_} épocas, melhor época={epocas}")
        params_rn.update(early_stopping=False, max_iter=epocas,
                         n_iter_no_change=epocas)  # roda exatamente 'epocas' épocas

    rn = MLPRegressor(random_state=SEED, **params_rn).fit(X_pad, y)
    joblib.dump(scaler, PASTA_MODELOS / "scaler.joblib")
    joblib.dump(rn, PASTA_MODELOS / "melhor_rn.joblib")
    print(f"  fit final: {X_pad.shape[0]} vítimas, épocas={rn.n_iter_}, "
          f"early_stopping={rn.early_stopping}")
    print(f"  MSE no treino completo: {mse(y, rn.predict(X_pad)):.5f}")

    print(f"\nModelos salvos em {PASTA_MODELOS.resolve()}")


if __name__ == "__main__":
    main()
