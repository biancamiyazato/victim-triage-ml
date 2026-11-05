# Etapa 1: gera o dataset de treinamento/validação com exatamente 10.000 vítimas
# usando o gerador fornecido (gerar_dados_vitimas.py) e registra os parâmetros
# de criação e a definição dos folds da validação cruzada.

import matplotlib
matplotlib.use("Agg")  # evita que plt.show() do gerador bloqueie a execução
import matplotlib.pyplot as plt
import pandas as pd

import gerar_dados_vitimas as gerador
from config import (SEED, PASTA_TREINO, CSV_TREINO, DISTRIB_TRI, MEDIA_IDADE,
                    DESVIO_IDADE, NIVEL_RUIDO, K_FOLDS, FEATURES, TARGET,
                    PASTA_RESULTADOS)

CORES = {0: 'verde', 1: 'amarelo', 2: 'vermelho', 3: 'preto'}


def main():
    assert sum(DISTRIB_TRI.values()) == 10000, "O dataset deve ter exatamente 10.000 vítimas"

    PASTA_TREINO.mkdir(parents=True, exist_ok=True)
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # Redireciona a saída do gerador original (que aponta para 1300v)
    gerador.OUTPUT_CSV = CSV_TREINO

    df = gerador.gerar_dataset_vitimas(
        distrib_tri=DISTRIB_TRI,
        media_idade=MEDIA_IDADE,
        desvio_idade=DESVIO_IDADE,
        nivel_ruido=NIVEL_RUIDO,
        seed=SEED,
    )
    plt.savefig(PASTA_RESULTADOS / "etapa1_histograma_sobr.png", dpi=120)
    plt.close('all')

    # Verificações de integridade
    assert len(df) == 10000
    assert list(df.columns[:10]) == FEATURES and TARGET in df.columns
    assert df[FEATURES + [TARGET]].isna().sum().sum() == 0

    # Tabela 1 do relatório
    contagem = df['tri'].value_counts().sort_index()
    linhas = [(f"Vítimas tri={CORES[k]}", int(contagem.get(k, 0))) for k in CORES]
    linhas += [
        ("Idade das vítimas (média amostral)", f"{df['idade'].mean():.2f}"),
        ("Desvio padrão amostral da idade", f"{df['idade'].std(ddof=1):.2f}"),
        ("ruído", NIVEL_RUIDO),
        ("Média da idade (parâmetro do gerador)", MEDIA_IDADE),
        ("Desvio padrão da idade (parâmetro do gerador)", DESVIO_IDADE),
        ("Semente", SEED),
        ("Número de folds (validação cruzada)", K_FOLDS),
    ]
    tabela = pd.DataFrame(linhas, columns=["MODELO", "VALOR"])
    tabela.to_csv(PASTA_RESULTADOS / "etapa1_dataset.csv", index=False)

    print("\n===== TABELA 1: DATASET DE TREINAMENTO/VALIDAÇÃO =====")
    print(tabela.to_string(index=False))

    print("\n===== Estatísticas de sobr =====")
    print(df[TARGET].describe().round(4).to_string())


if __name__ == "__main__":
    main()
