import os, glob, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold

ROOT = 'd:/hack/s6e11'
CP = 'C:/Users/Admin/.cache/kagglehub/competitions/playground-series-s6e11'
tr = pd.read_csv(f'{CP}/train.csv', usecols=['id', 'is_fraud'])
te = pd.read_csv(f'{CP}/test.csv', usecols=['id'])
y = tr.is_fraud.astype(int).values
NTR, NTE = len(tr), len(te)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(NTR), y))
STK = 'stack|blend|nested|winner|wide|fusion|distill'


def lg(r):
    r = np.clip(np.asarray(r, np.float32), 1e-6, 1 - 1e-6)
    return np.log(r / (1 - r))


def rk(v):
    return (rankdata(v) / len(v)).astype(np.float32)


class Bank:
    def __init__(s, dirs):
        s.O, s.T, M = [], [], []
        for d in dirs:
            m = pd.read_csv(f'{d}/members.csv')
            m['bank'], m['k'], m['j'] = os.path.basename(d), len(s.O), np.arange(len(m))
            s.O.append(np.load(f'{d}/O.npy', mmap_mode='r'))
            s.T.append(np.load(f'{d}/T.npy', mmap_mode='r'))
            M.append(m)
        s.m = pd.concat(M, ignore_index=True)

    def add(s, name, o, t, src='own'):
        s.O.append(rk(o)[:, None])
        s.T.append(rk(t)[:, None])
        s.m = pd.concat([s.m, pd.DataFrame([dict(src=src, name=name, auc=auc(y, o), split_ok=True, bank=src, k=len(s.O) - 1, j=0)])],
                        ignore_index=True)

    def cols(s, idx, side='O', rows=slice(None)):
        A = s.O if side == 'O' else s.T
        n = len(A[0][rows]) if isinstance(rows, slice) else len(rows)
        X = np.empty((n, len(idx)), np.float32)
        for c, i in enumerate(idx):
            X[:, c] = A[s.m.k.iat[i]][rows, s.m.j.iat[i]]
        return X

    def stacky(s):
        return (s.m.name.str.contains(STK, case=False) | s.m.src.str.contains(STK, case=False)
                & ~s.m.name.str.contains('oof_|lgb|xgb|glm|cat|gpt|lm', case=False)).values & (s.m.src != 'own').values


def dedupe(B, order, thr=0.99995):
    Z = B.cols(order, rows=slice(None, None, 5))
    Z = (Z - Z.mean(0)) / Z.std(0)
    G = Z.T @ Z / len(Z)
    keep = []
    for i in range(len(order)):
        if all(G[i, j] < thr for j in keep):
            keep.append(i)
    return order[keep]


def lr(X, yy, rows, C, it=40, ch=65536):
    d = X.shape[1] + 1
    w = np.zeros(d)
    for _ in range(it):
        g, H = np.zeros(d), np.zeros((d, d))
        w32 = w[:-1].astype(np.float32)
        for s in range(0, len(rows), ch):
            r = rows[s:s + ch]
            Xc = X[r]
            p = 1 / (1 + np.exp(-(Xc @ w32 + w[-1]).astype(np.float64)))
            e, q = p - yy[r], p * (1 - p)
            g[:-1] += Xc.T @ e.astype(np.float32)
            g[-1] += e.sum()
            Xq = Xc * q.astype(np.float32)[:, None]
            H[:-1, :-1] += Xq.T @ Xc
            H[:-1, -1] += Xq.sum(0)
            H[-1, -1] += q.sum()
        H[-1, :-1] = H[:-1, -1]
        g[:-1] += w[:-1] / C
        H[np.arange(d - 1), np.arange(d - 1)] += 1 / C
        dw = np.linalg.solve(H, g)
        w -= dw
        if np.abs(dw).max() < 1e-6:
            break
    return w


def dec(X, w, ch=131072):
    return np.concatenate([X[s:s + ch] @ w[:-1].astype(np.float32) + w[-1] for s in range(0, len(X), ch)])


def run(B, gate, Cs, tag, sub=True, mask=None, verbose=True):
    t0 = time.time()
    m = B.m
    ok = (m.auc >= m.auc.max() - gate).values & m.split_ok.values & (mask if mask is not None else True)
    order = np.where(ok)[0]
    order = order[np.argsort(-m.auc.values[order])]
    idx = dedupe(B, order)
    X = lg(B.cols(idx))
    res = {}
    for C in Cs:
        v = np.zeros(NTR)
        for a, b in FOLDS:
            v[b] = rankdata(dec(X[b], lr(X, y, a, C))) / len(b)
        res[C] = v
        if verbose:
            print(f'{tag} gate {gate} n {len(idx)} C {C}: nested {auc(y, v):.6f} folds {[round(auc(y[b], v[b]), 5) for _, b in FOLDS]} {time.time() - t0:.0f}s', flush=True)
    Cb = max(res, key=lambda c: auc(y, res[c]))
    if not sub:
        return res[Cb], None, idx
    w = lr(X, y, np.arange(NTR), Cb)
    del X
    p = rankdata(dec(lg(B.cols(idx, 'T')), w)) / NTE
    os.makedirs(f'{ROOT}/sub', exist_ok=True)
    pd.DataFrame({'id': te.id, 'is_fraud': p}).to_csv(f'{ROOT}/sub/{tag}.csv', index=False)
    np.save(f'{ROOT}/sub/{tag}_nested.npy', res[Cb])
    pd.DataFrame({'src': m.src.values[idx], 'name': m.name.values[idx], 'auc': m.auc.values[idx], 'coef': w[:-1]}).sort_values(
        'coef', key=abs, ascending=False).to_csv(f'{ROOT}/sub/{tag}_coef.csv', index=False)
    return res[Cb], p, idx


def banks():
    return Bank(sorted(glob.glob(f'{ROOT}/out/dev[12]')))


if __name__ == '__main__':
    B = banks()
    print(len(B.m), 'best', B.m.auc.max(), 'stacky', B.stacky().sum(), flush=True)
    run(B, 0.015, [0.01, 0.02, 0.04], 'wide_ns', mask=~B.stacky())
