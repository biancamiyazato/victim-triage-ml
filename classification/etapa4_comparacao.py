# Etapa 4: compara os modelos CART x RN pelo viés (f1 macro médio de treino x validação)
# e escolhe o melhor CART e a melhor RN para o retreino da etapa 5.

import json

from config import RESULTADOS_DIR
from validacao_cruzada import imprimir_tabela, salvar_json

LIMITE_SOBREAJUSTE = 0.05   # diferença média treino-validação acima disso => sobreajuste
LIMITE_SUBAJUSTE = 0.05     # validação abaixo do melhor da família por mais que isso => subajuste


def carregar(nome_arquivo):
    with open(RESULTADOS_DIR / nome_arquivo, encoding="utf-8") as f:
        return json.load(f)


def diagnosticar(resultados):
    """Classifica cada modelo pelo viés e escolhe o de maior f1 de validação sem sobreajuste."""
    melhor_valid = max(r["validacao"]["media"] for r in resultados.values())
    diagnostico = {}
    for nome, r in resultados.items():
        if r["difs"]["media"] > LIMITE_SOBREAJUSTE:
            diagnostico[nome] = "sobreajustado"
        elif r["validacao"]["media"] < melhor_valid - LIMITE_SUBAJUSTE:
            diagnostico[nome] = "subajustado"
        else:
            diagnostico[nome] = "equilibrado"

    candidatos = [n for n, d in diagnostico.items() if d != "sobreajustado"]
    melhor = max(candidatos, key=lambda n: resultados[n]["validacao"]["media"])
    return diagnostico, melhor


def imprimir_diagnostico(familia, resultados, diagnostico, melhor):
    print(f"\n--- Viés {familia} ---")
    for nome, r in resultados.items():
        marca = "  <== melhor" if nome == melhor else ""
        print(f"{familia} {nome}: treino={r['treino']['media']:.5f} "
              f"validação={r['validacao']['media']:.5f} "
              f"dif={r['difs']['media']:.5f} -> {diagnostico[nome]}{marca}")


if __name__ == "__main__":
    cart, rn = carregar("etapa2_cart.json"), carregar("etapa3_rn.json")

    diag_cart, melhor_cart = diagnosticar(cart["resultados"])
    diag_rn, melhor_rn = diagnosticar(rn["resultados"])

    imprimir_diagnostico("CART", cart["resultados"], diag_cart, melhor_cart)
    imprimir_diagnostico("RN", rn["resultados"], diag_rn, melhor_rn)

    imprimir_tabela("TABELA 6: MELHOR MODELO CART X MELHOR MODELO RN", {
        f"Melhor CART ({melhor_cart})": cart["resultados"][melhor_cart],
        f"Melhor RN ({melhor_rn})": rn["resultados"][melhor_rn],
    })

    v_cart = cart["resultados"][melhor_cart]["validacao"]["media"]
    v_rn = rn["resultados"][melhor_rn]["validacao"]["media"]
    vencedor = "RN" if v_rn > v_cart else "CART"
    print(f"\nNa validação cruzada, {vencedor} teve o maior f1 macro médio "
          f"(diferença de {abs(v_rn - v_cart):.5f}).")

    salvar_json(RESULTADOS_DIR / "etapa4_melhores.json", {
        "criterio": (f"maior f1 macro médio de validação entre os modelos com "
                     f"diferença média treino-validação <= {LIMITE_SOBREAJUSTE}"),
        "cart": {"modelo": melhor_cart, "diagnostico": diag_cart,
                 "hiperparametros": cart["hiperparametros"][melhor_cart],
                 "resultados": cart["resultados"][melhor_cart]},
        "rn": {"modelo": melhor_rn, "diagnostico": diag_rn,
               "hiperparametros": rn["hiperparametros"][melhor_rn],
               "resultados": rn["resultados"][melhor_rn]},
    })
    print(f"Escolhas salvas em {RESULTADOS_DIR / 'etapa4_melhores.json'}")
