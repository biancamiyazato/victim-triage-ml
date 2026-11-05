# Etapa 4: compara o melhor CART com a melhor RN a partir dos MSEs por fold
# salvos nas etapas 2 e 3 (não há novo treinamento aqui).
# Critério de escolha: menor MSE médio de validação; a diferença treino x
# validação é usada para confirmar que o modelo escolhido não está sobreajustado.

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import K_FOLDS, PASTA_RESULTADOS
from validacao import media_dpa, tabela_resultados


def carregar_folds(arquivo):
    """Lê o CSV de folds e devolve {modelo: {'treino', 'validacao', 'diferenca'}}."""
    df = pd.read_csv(PASTA_RESULTADOS / arquivo)
    return {
        nome: {
            'treino': g.sort_values('fold')['mse_treino'].to_numpy(),
            'validacao': g.sort_values('fold')['mse_validacao'].to_numpy(),
            'diferenca': g.sort_values('fold')['dif_abs'].to_numpy(),
        }
        for nome, g in df.groupby('modelo', sort=False)
    }


def escolher_melhor(resultados):
    return min(resultados, key=lambda n: media_dpa(resultados[n]['validacao'])[0])


def main():
    cart = carregar_folds("etapa2_cart_folds.csv")
    rn = carregar_folds("etapa3_rn_folds.csv")

    print("===== Viés: MSE médio de treino x validação =====")
    for nome, r in {**cart, **rn}.items():
        tr, va = media_dpa(r['treino'])[0], media_dpa(r['validacao'])[0]
        print(f"{nome}: treino={tr:.5f} validação={va:.5f} "
              f"dif={media_dpa(r['diferenca'])[0]:.5f}")

    melhor_cart = escolher_melhor(cart)
    melhor_rn = escolher_melhor(rn)
    letra_cart = melhor_cart.split()[-1]
    letra_rn = melhor_rn.split()[-1]

    # Tabela 6: melhor CART x melhor RN
    tab = tabela_resultados({
        f"Melhor CART ({letra_cart})": cart[melhor_cart],
        f"Melhor RN ({letra_rn})": rn[melhor_rn],
    })
    tab.to_csv(PASTA_RESULTADOS / "etapa4_comparacao.csv")

    # Registra a escolha para o retreino (etapa 5)
    with open(PASTA_RESULTADOS / "etapa4_melhores.json", "w", encoding="utf-8") as f:
        json.dump({'cart': melhor_cart, 'rn': melhor_rn}, f, ensure_ascii=False, indent=2)

    print("\n===== TABELA 6: MELHOR MODELO CART X MELHOR MODELO RN =====")
    print(tab.to_string())

    va_cart = media_dpa(cart[melhor_cart]['validacao'])[0]
    va_rn = media_dpa(rn[melhor_rn]['validacao'])[0]
    vencedor, perdedor = (melhor_rn, melhor_cart) if va_rn < va_cart else (melhor_cart, melhor_rn)
    ganho = (1 - min(va_cart, va_rn) / max(va_cart, va_rn)) * 100
    print(f"\n{vencedor} tem MSE médio de validação {ganho:.1f}% menor que {perdedor}")

    # Gráfico: MSE de validação por fold dos dois melhores modelos
    fig, ax = plt.subplots(figsize=(6, 4))
    folds = range(1, K_FOLDS + 1)
    for nome, r, cor in [(melhor_cart, cart[melhor_cart], '#4C78A8'),
                         (melhor_rn, rn[melhor_rn], '#F58518')]:
        ax.plot(folds, r['validacao'], 'o-', color=cor, label=f"{nome} (validação)")
        ax.plot(folds, r['treino'], 'o--', color=cor, alpha=0.6, label=f"{nome} (treino)")
    ax.set_xticks(list(folds))
    ax.set_xlabel('Fold')
    ax.set_ylabel('MSE')
    ax.set_title('Melhor CART x melhor RN: MSE por fold')
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "etapa4_comparacao.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
