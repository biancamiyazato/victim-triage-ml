# Validação cruzada k-fold compartilhada entre CART (etapa 2) e RN (etapa 3).
# Os mesmos folds (KFold embaralhado com SEED) são usados para todos os modelos.

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from config import SEED, K_FOLDS, CSV_TREINO, FEATURES, TARGET


def mse(y_real, y_pred):
    """MSE conforme a fórmula do enunciado: sqrt((1/n) * sum((y - y_pred)^2))."""
    return float(np.sqrt(mean_squared_error(y_real, y_pred)))


def rmse(y_real, y_pred):
    """RMSE conforme a tabela 7 do enunciado: sqrt(MSE), com o MSE definido acima."""
    return float(np.sqrt(mse(y_real, y_pred)))


def carregar_treino():
    df = pd.read_csv(CSV_TREINO)
    return df[FEATURES].to_numpy(dtype=float), df[TARGET].to_numpy(dtype=float)


def validacao_cruzada(modelo, X, y, k=K_FOLDS, seed=SEED):
    """Treina um clone do modelo em cada fold e retorna os MSEs de treino e validação."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    mse_treino, mse_valid = [], []
    for idx_tr, idx_va in kf.split(X):
        m = clone(modelo)
        m.fit(X[idx_tr], y[idx_tr])
        mse_treino.append(mse(y[idx_tr], m.predict(X[idx_tr])))
        mse_valid.append(mse(y[idx_va], m.predict(X[idx_va])))
    mse_treino, mse_valid = np.array(mse_treino), np.array(mse_valid)
    return {
        'treino': mse_treino,
        'validacao': mse_valid,
        'diferenca': np.abs(mse_valid - mse_treino),
    }


def media_dpa(valores):
    """Média e desvio padrão amostral (ddof=1)."""
    return float(np.mean(valores)), float(np.std(valores, ddof=1))


def fmt(valores, casas=5):
    m, d = media_dpa(valores)
    return f"{m:.{casas}f} ({d:.{casas}f})".replace('.', ',')


def imprimir_folds(nome, res):
    print(f"\n--- {nome} ---")
    for chave, rotulo in [('treino', 'Treino    MSE por fold'),
                          ('validacao', 'Validação MSE por fold'),
                          ('diferenca', 'Diferença absoluta MSE')]:
        v = res[chave]
        m, d = media_dpa(v)
        print(f"{rotulo}: {np.round(v, 5).tolist()} média={m:.5f} dpa={d:.5f}")


def tabela_resultados(resultados):
    """Monta a tabela de resultados (formato do relatório) a partir de {nome: res}."""
    linhas = {
        'TREINO MSE (dpa)': {n: fmt(r['treino']) for n, r in resultados.items()},
        'VALIDAÇÃO MSE (dpa)': {n: fmt(r['validacao']) for n, r in resultados.items()},
        'MÉDIA DAS DIFS. (dpa)': {n: fmt(r['diferenca']) for n, r in resultados.items()},
    }
    return pd.DataFrame(linhas).T.rename_axis('MODELO')


def folds_dataframe(resultados):
    """MSEs por fold de cada modelo, em formato longo, para salvar em CSV."""
    linhas = []
    for nome, r in resultados.items():
        for i in range(len(r['treino'])):
            linhas.append({'modelo': nome, 'fold': i + 1,
                           'mse_treino': r['treino'][i],
                           'mse_validacao': r['validacao'][i],
                           'dif_abs': r['diferenca'][i]})
    return pd.DataFrame(linhas)
