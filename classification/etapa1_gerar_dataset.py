# Etapa 1: gera o dataset de treinamento/validação (10.000 vítimas) com o
# programa gerar_dados_vitimas.py e registra os parâmetros para o relatório.

import json
import warnings

import matplotlib
matplotlib.use("Agg")  # evita a janela bloqueante do plt.show() do gerador
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

import gerar_dados_vitimas as gerador
from config import SEED, N_FOLDS, TREINO_DIR, TREINO_CSV, FEATURES, TARGET, CLASSES

# Parâmetros de criação do dataset
N_VITIMAS = 10000
DISTRIB_TRI = {0: 2500, 1: 2500, 2: 2500, 3: 2500}  # classes balanceadas
MEDIA_IDADE = 40
DESVIO_IDADE = 25
NIVEL_RUIDO = 0.05

assert sum(DISTRIB_TRI.values()) == N_VITIMAS

# Redireciona a saída do gerador para não sobrescrever a pasta do teste cego (1300v)
TREINO_DIR.mkdir(parents=True, exist_ok=True)
gerador.OUTPUT_CSV = TREINO_CSV

df = gerador.gerar_dataset_vitimas(
    distrib_tri=DISTRIB_TRI,
    media_idade=MEDIA_IDADE,
    desvio_idade=DESVIO_IDADE,
    nivel_ruido=NIVEL_RUIDO,
    seed=SEED,
)
plt.savefig(TREINO_DIR / "hist_sobr.png", dpi=120)
plt.close("all")

# Estatísticas para a tabela 1 do relatório
contagem = df[TARGET].value_counts().sort_index()
resumo = {
    "n_vitimas": len(df),
    "seed": SEED,
    "n_folds": N_FOLDS,
    "parametros_gerador": {
        "distrib_tri": DISTRIB_TRI,
        "media_idade": MEDIA_IDADE,
        "desvio_idade": DESVIO_IDADE,
        "nivel_ruido": NIVEL_RUIDO,
    },
    "vitimas_por_tri": {CLASSES[k]: int(v) for k, v in contagem.items()},
    "idade_media_amostral": round(float(df["idade"].mean()), 2),
    "idade_dpa": round(float(df["idade"].std(ddof=1)), 2),
}
with open(TREINO_DIR / "resumo_dataset.json", "w", encoding="utf-8") as f:
    json.dump(resumo, f, indent=2, ensure_ascii=False)

print("\n===== TABELA 1: DATASET DE TREINAMENTO/VALIDAÇÃO =====")
for nome, qtd in resumo["vitimas_por_tri"].items():
    print(f"Vítimas tri={nome:<9} {qtd}")
print(f"Idade das vítimas (média)        {resumo['idade_media_amostral']}")
print(f"Desvio padrão amostral da idade  {resumo['idade_dpa']}")
print(f"Ruído                            {NIVEL_RUIDO}")
print(f"Folds (validação cruzada)        {N_FOLDS}")
print(f"Entradas: {FEATURES} -> saída: {TARGET}")
