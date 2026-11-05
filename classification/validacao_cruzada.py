# Validação cruzada estratificada com f1 macro de treino e de validação por fold.
# Usado pelas etapas 2 (CART) e 3 (RN).

import json

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

from config import SEED, N_FOLDS


def _resumo(valores):
    # dpa calculado como no exemplo do enunciado (np.std, ddof=0)
    return {"folds": [round(float(v), 5) for v in valores],
            "media": round(float(np.mean(valores)), 5),
            "dpa": round(float(np.std(valores)), 5)}


def avaliar_cv(modelo, X, y):
    """Retorna f1 macro por fold (treino, validação e diferença absoluta) com média e dpa."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    r = cross_validate(modelo, X, y, cv=cv, scoring="f1_macro",
                       return_train_score=True, n_jobs=-1)
    treino, valid = r["train_score"], r["test_score"]
    return {"treino": _resumo(treino),
            "validacao": _resumo(valid),
            "difs": _resumo(np.abs(treino - valid))}


def imprimir_resultado(nome, res):
    print(f"\n--- {nome} (k={N_FOLDS}) ---")
    rotulos = {"treino": "Treino:   ", "validacao": "Validação:", "difs": "Difs abs: "}
    for chave, rotulo in rotulos.items():
        r = res[chave]
        folds = ", ".join(f"{v:.5f}" for v in r["folds"])
        print(f"{rotulo} F1 por fold: [{folds}] média={r['media']:.5f} dpa={r['dpa']:.5f}")


def imprimir_tabela(titulo, resultados):
    print(f"\n===== {titulo} =====")
    print(f"{'MODELO':<24}" + "".join(f"{m:<22}" for m in resultados))
    linhas = {"treino": "TREINO f1 (DPA)", "validacao": "VALIDAÇÃO f1 (DPA)",
              "difs": "MÉDIA DAS DIFS. (DPA)"}
    for chave, rotulo in linhas.items():
        print(f"{rotulo:<24}" + "".join(
            f"{r[chave]['media']:.5f} ({r[chave]['dpa']:.5f})".ljust(22) for r in resultados.values()))


def salvar_json(caminho, conteudo):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(conteudo, f, indent=2, ensure_ascii=False)
