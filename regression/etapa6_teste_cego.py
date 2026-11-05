# Etapa 6: teste cego dos modelos retreinados na etapa 5 com o dataset de
# 1300 vítimas. ÚNICO script que lê o dataset de teste cego.
#
# Métricas conforme o enunciado:
#   MSE  = sqrt((1/n) * sum((y - y_pred)^2))
#   RMSE = sqrt(MSE)
#   Ganho % (somente para o melhor modelo) = (1 - erro_melhor / erro_outro) * 100

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import CSV_TESTE, FEATURES, TARGET, PASTA_RESULTADOS
from validacao import mse, rmse

PASTA_MODELOS = Path("./modelos")


def fmt(v, casas=5):
    return f"{v:.{casas}f}".replace('.', ',')


def ganhos(erro_cart, erro_rn):
    """Ganho % apenas na coluna do melhor modelo; a outra fica em branco."""
    if erro_cart < erro_rn:
        return f"{fmt((1 - erro_cart / erro_rn) * 100, 2)}%", ""
    if erro_rn < erro_cart:
        return "", f"{fmt((1 - erro_rn / erro_cart) * 100, 2)}%"
    return "0%", "0%"


def grafico_dispersao(y, y_pred, titulo, arquivo, cor):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color='#9A9A9A', linewidth=1.5, linestyle='--',
            label='Predição perfeita (y = ŷ)', zorder=1)
    ax.scatter(y, y_pred, s=14, alpha=0.45, color=cor, edgecolors='none', zorder=2,
               label='Vítimas')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_xlabel('sobr real')
    ax.set_ylabel('sobr predito')
    ax.set_title(titulo)
    ax.grid(linestyle='--', alpha=0.4)
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / arquivo, dpi=150)
    plt.close(fig)


def main():
    with open(PASTA_RESULTADOS / "etapa4_melhores.json", encoding="utf-8") as f:
        melhores = json.load(f)
    letra_cart = melhores['cart'].split()[-1]
    letra_rn = melhores['rn'].split()[-1]

    df = pd.read_csv(CSV_TESTE)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    print(f"Teste cego: {len(df)} vítimas")

    cart = joblib.load(PASTA_MODELOS / "melhor_cart.joblib")
    rn = joblib.load(PASTA_MODELOS / "melhor_rn.joblib")
    scaler = joblib.load(PASTA_MODELOS / "scaler.joblib")

    pred_cart = cart.predict(X)
    pred_rn = rn.predict(scaler.transform(X))

    mse_cart, mse_rn = mse(y, pred_cart), mse(y, pred_rn)
    rmse_cart, rmse_rn = rmse(y, pred_cart), rmse(y, pred_rn)
    ganho_mse = ganhos(mse_cart, mse_rn)
    ganho_rmse = ganhos(rmse_cart, rmse_rn)

    col_cart, col_rn = f"Melhor CART ({letra_cart})", f"Melhor RN ({letra_rn})"
    tab = pd.DataFrame({
        col_cart: [fmt(mse_cart), ganho_mse[0], fmt(rmse_cart), ganho_rmse[0]],
        col_rn: [fmt(mse_rn), ganho_mse[1], fmt(rmse_rn), ganho_rmse[1]],
    }, index=['MSE', 'Ganho % de MSE', 'RMSE', 'Ganho % de RMSE']).rename_axis('MODELO')
    tab.to_csv(PASTA_RESULTADOS / "etapa6_teste_cego.csv")

    print("\n===== TABELA 7: RESULTADOS DO TESTE CEGO =====")
    print(tab.to_string())

    # Predições salvas para conferência
    pd.DataFrame({'sobr_real': y, 'pred_cart': pred_cart, 'pred_rn': pred_rn}).to_csv(
        PASTA_RESULTADOS / "etapa6_predicoes.csv", index=False)

    grafico_dispersao(y, pred_cart, f"Teste cego: CART {letra_cart} (MSE = {fmt(mse_cart)})",
                      "etapa6_dispersao_cart.png", '#2A6FB0')
    grafico_dispersao(y, pred_rn, f"Teste cego: RN {letra_rn} (MSE = {fmt(mse_rn)})",
                      "etapa6_dispersao_rn.png", '#2A6FB0')
    print(f"\nGráficos salvos em {PASTA_RESULTADOS.resolve()}")


if __name__ == "__main__":
    main()
