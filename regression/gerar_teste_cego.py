# Gera o dataset de teste cego com 1300 vítimas usando o gerador fornecido
# (gerar_dados_vitimas.py) com os parâmetros do seu __main__.
# Este dataset é usado SOMENTE na etapa 6 (teste cego).

import matplotlib
matplotlib.use("Agg")  # evita que plt.show() do gerador bloqueie a execução
import matplotlib.pyplot as plt

import gerar_dados_vitimas as gerador
from config import (PASTA_TESTE, CSV_TESTE, DISTRIB_TRI_TESTE, SEED_TESTE,
                    MEDIA_IDADE, DESVIO_IDADE, NIVEL_RUIDO, FEATURES, TARGET)


def main():
    assert sum(DISTRIB_TRI_TESTE.values()) == 1300

    PASTA_TESTE.mkdir(parents=True, exist_ok=True)
    gerador.OUTPUT_CSV = CSV_TESTE

    df = gerador.gerar_dataset_vitimas(
        distrib_tri=DISTRIB_TRI_TESTE,
        media_idade=MEDIA_IDADE,
        desvio_idade=DESVIO_IDADE,
        nivel_ruido=NIVEL_RUIDO,
        seed=SEED_TESTE,
    )
    plt.close('all')

    assert len(df) == 1300
    assert list(df.columns[:10]) == FEATURES and TARGET in df.columns
    assert df[FEATURES + [TARGET]].isna().sum().sum() == 0

    print(f"Idade: média={df['idade'].mean():.2f} dpa={df['idade'].std(ddof=1):.2f}")
    print(df[TARGET].describe().round(4).to_string())


if __name__ == "__main__":
    main()
