# Etapa 5: retreina o melhor CART e a melhor RN com as 10.000 amostras (sem CV/split),
# salva os modelos e avalia no teste cego (1300v), usado SOMENTE aqui.

import json
import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, classification_report)
from sklearn.tree import DecisionTreeClassifier

from config import SEED, TREINO_CSV, TESTE_CEGO_CSV, RESULTADOS_DIR, FEATURES, TARGET, CLASSES
from etapa3_rn import criar_rn
from validacao_cruzada import salvar_json

with open(RESULTADOS_DIR / "etapa4_melhores.json", encoding="utf-8") as f:
    melhores = json.load(f)

params_cart = melhores["cart"]["hiperparametros"]
params_rn = dict(melhores["rn"]["hiperparametros"])
params_rn["hidden_layer_sizes"] = tuple(params_rn["hidden_layer_sizes"])  # json salva como lista

# --- Retreino com todo o dataset de treinamento/validação ---
treino = pd.read_csv(TREINO_CSV)
X_treino, y_treino = treino[FEATURES], treino[TARGET]

modelos = {
    f"Melhor CART ({melhores['cart']['modelo']})":
        DecisionTreeClassifier(random_state=SEED, **params_cart).fit(X_treino, y_treino),
    f"Melhor RN ({melhores['rn']['modelo']})":
        criar_rn(params_rn).fit(X_treino, y_treino),
}
arquivos = ["melhor_cart.joblib", "melhor_rn.joblib"]
for modelo, arquivo in zip(modelos.values(), arquivos):
    joblib.dump(modelo, arquivo)
print(f"Modelos salvos: {', '.join(arquivos)}")

# --- Teste cego ---
teste = pd.read_csv(TESTE_CEGO_CSV)
X_teste, y_teste = teste[FEATURES], teste[TARGET]
rotulos = sorted(CLASSES)
siglas = ["G", "Y", "R", "B"]

resultados = {}
for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_teste)
    p, r, f1, _ = precision_recall_fscore_support(y_teste, y_pred, labels=rotulos,
                                                  average="macro", zero_division=0)
    resultados[nome] = {
        "precisao_macro": round(float(p), 5),
        "recall_macro": round(float(r), 5),
        "f1_macro": round(float(f1), 5),
        "acuracia": round(float(accuracy_score(y_teste, y_pred)), 5),
        "matriz_confusao": confusion_matrix(y_teste, y_pred, labels=rotulos).tolist(),
    }
    print(f"\n--- {nome}: relatório por classe ---")
    print(classification_report(y_teste, y_pred, labels=rotulos,
                                target_names=list(CLASSES.values()), digits=5, zero_division=0))

print(f"\n===== TABELA 7: TESTE CEGO ({len(teste)} vítimas) =====")
print(f"{'MODELO':<24}" + "".join(f"{n:<22}" for n in resultados))
for chave, rotulo in [("precisao_macro", "MÉDIA PRECISÃO (macro)"), ("recall_macro", "MÉDIA RECALL (macro)"),
                      ("f1_macro", "F1 SCORE (macro)"), ("acuracia", "ACURÁCIA")]:
    print(f"{rotulo:<24}" + "".join(f"{r[chave]:<22.5f}" for r in resultados.values()))

print("\n===== TABELA 8: MATRIZES DE CONFUSÃO (linhas=real, colunas=predito) =====")
for nome, r in resultados.items():
    print(f"\n{nome}")
    print(pd.DataFrame(r["matriz_confusao"], index=siglas, columns=siglas).to_string())

salvar_json(RESULTADOS_DIR / "etapa5_teste_cego.json", {
    "n_vitimas_teste": len(teste),
    "resultados": resultados,
})
print(f"\nResultados salvos em {RESULTADOS_DIR / 'etapa5_teste_cego.json'}")
