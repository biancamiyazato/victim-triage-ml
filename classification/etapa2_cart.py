# Etapa 2: treina o CART com validação cruzada usando 3 hiperparametrizações:
# subajustada (U), equilibrada (E) e sobreajustada (O).

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from config import SEED, N_FOLDS, TREINO_CSV, RESULTADOS_DIR, FEATURES, TARGET
from validacao_cruzada import avaliar_cv, imprimir_resultado, imprimir_tabela, salvar_json

HIPERPARAMETROS_CART = {
    # árvore rasa: não consegue separar as 4 classes (treino e validação baixos)
    "U": {"min_samples_leaf": 50, "max_depth": 2, "criterion": "gini"},
    # profundidade moderada e folhas com no mínimo 10 amostras: melhor validação e diferença pequena
    "E": {"min_samples_leaf": 10, "max_depth": 8, "criterion": "entropy"},
    # árvore sem limites: decora o treino (f1=1.0) e aprende o ruído
    "O": {"min_samples_leaf": 1, "max_depth": None, "criterion": "gini"},
}

df = pd.read_csv(TREINO_CSV)
X, y = df[FEATURES], df[TARGET]

resultados = {}
for nome, params in HIPERPARAMETROS_CART.items():
    modelo = DecisionTreeClassifier(random_state=SEED, **params)
    resultados[nome] = avaliar_cv(modelo, X, y)
    imprimir_resultado(f"CART {nome} {params}", resultados[nome])

print("\n===== TABELA 2: HIPERPARAMETRIZAÇÕES CART =====")
print(pd.DataFrame(HIPERPARAMETROS_CART).fillna("None").to_string())
imprimir_tabela("TABELA 3: RESULTADOS DOS MODELOS CART", resultados)

salvar_json(RESULTADOS_DIR / "etapa2_cart.json", {
    "n_folds": N_FOLDS,
    "hiperparametros": HIPERPARAMETROS_CART,
    "resultados": resultados,
})
print(f"\nResultados salvos em {RESULTADOS_DIR / 'etapa2_cart.json'}")
