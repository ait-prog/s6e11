import os, glob, re, json, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold

t0 = time.time()
IN, OUT = '/kaggle/input', '/kaggle/working'
cmp = [p for p in glob.glob(f'{IN}/**/train.csv', recursive=True) if 'playground-series-s6e11' in p and 'datasets' not in p]
CD = os.path.dirname(sorted(cmp, key=len)[0])
tr, te = pd.read_csv(f'{CD}/train.csv'), pd.read_csv(f'{CD}/test.csv')
y = tr.is_fraud.astype(int).values
NTR, NTE = len(tr), len(te)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(NTR), y))
fid = np.zeros(NTR, int)
for k, (_, b) in enumerate(FOLDS):
    fid[b] = k
print('comp', CD, NTR, NTE, flush=True)


def src_of(p):
    q = os.path.relpath(p, IN).split(os.sep)
    if q[0] in ('datasets', 'notebooks', 'kernels', 'competitions') and len(q) > 3:
        return '/'.join(q[1:3]), os.sep.join(q[3:])
    return q[0], os.sep.join(q[1:])


STRIP = ['predictions', 'prediction', 'preds', 'pred', 'submission', 'oof', 'test', 'full', 'refit', 'foldavg',
         'scores', 'proba', 'probs']


def norm(s):
    s = s.lower()
    for w in STRIP:
        s = s.replace(w, '')
    return re.sub(r'[^a-z0-9]+', '_', s).strip('_')


def kind(s):
    s = s.lower()
    return 'R' if any(w in s for w in ('full', 'refit')) else 'F'


def vec(a):
    a = np.asarray(a)
    if a.dtype.kind not in 'fiu':
        return None
    a = a.astype(np.float64)
    if a.ndim == 3 and a.shape[2] == 2:
        return np.logaddexp.reduce(a[:, :, 1], 1) - np.logaddexp.reduce(a[:, :, 0], 1)
    if a.ndim == 2 and a.shape[1] == 2 and a.shape[0] in (NTR, NTE):
        return a[:, 1] - a[:, 0]
    return a.ravel() if a.ndim == 1 else None


def good(v, n):
    return v is not None and len(v) == n and np.isfinite(v).all() and np.unique(v[:5000]).size > 20


C = {}


def put(src, name, v, n, tag):
    side = 'o' if n == NTR else 't'
    C.setdefault(src, []).append((side, name, v, tag))


SKIPF = re.compile(r'(^|[\\/])(train|test|sample_submission|train_fe|test_fe|original.*|credit_card_fraud_2026)\.csv(\.gz)?$', re.I)
DROPC = {'id', 'fold', 'folds', 'target', 'y', 'row', 'index', 'unnamed: 0', 'kfold', 'transaction_id'}


def frame(src, rel, df):
    n = len(df)
    if n not in (NTR, NTE):
        return
    if 'id' in df.columns:
        ref = tr.id.values if n == NTR else te.id.values
        if set(df.id.values) != set(ref):
            return
        df = df.set_index('id').loc[ref].reset_index()
    cs = [c for c in df.columns if c.lower() not in DROPC and df[c].dtype.kind == 'f']
    if n == NTR:
        cs = [c for c in cs if not np.array_equal(df[c].values, y)]
    stem = re.sub(r'\.(csv|gz|parquet)', '', rel)
    for c in cs:
        v = df[c].values.astype(np.float64)
        if good(v, n):
            put(src, f'{stem}:{c}' if len(cs) > 1 or c != 'is_fraud' else stem, v, n, kind(stem))


files = [p for p in glob.glob(f'{IN}/**/*', recursive=True) if os.path.isfile(p) and not p.startswith(CD)]
split_bad = set()
for p in files:
    src, rel = src_of(p)
    low = rel.lower()
    try:
        if low.endswith('.npy'):
            if re.search(r'_f\d_(val_idx|params)', low):
                continue
            a = np.load(p, mmap_mode='r', allow_pickle=False)
            if a.shape[0] == NTR and a.ndim == 1 and a.dtype.kind in 'iu' and ('fold' in low):
                if not all(len(set(a[fid == k])) == 1 for k in range(5)):
                    split_bad.add(src)
                continue
            if a.shape[0] not in (NTR, NTE):
                continue
            if a.ndim == 2 and a.shape[1] > 2:
                for j in range(a.shape[1]):
                    v = vec(a[:, j])
                    if good(v, a.shape[0]):
                        put(src, f'{rel[:-4]}#{j}', v, a.shape[0], kind(rel))
                continue
            v = vec(a)
            if good(v, a.shape[0]):
                put(src, rel[:-4], v, a.shape[0], kind(rel))
        elif low.endswith('.npz'):
            z = np.load(p, allow_pickle=False)
            nm = None
            for mp in glob.glob(os.path.join(os.path.dirname(p), '*manifest*.json')):
                try:
                    m = json.load(open(mp))
                    nm = m.get('names') if isinstance(m, dict) else None
                except Exception:
                    pass
            for k in z.files:
                a = z[k]
                if a.ndim == 0 or a.shape[0] not in (NTR, NTE) or a.dtype.kind != 'f':
                    continue
                if a.ndim == 2 and a.shape[1] > 2:
                    for j in range(a.shape[1]):
                        nj = nm[j] if nm and j < len(nm) else str(j)
                        put(src, f'{rel[:-4]}#{k}#{nj}', a[:, j].astype(np.float64), a.shape[0], kind(k))
                else:
                    v = vec(a)
                    if good(v, a.shape[0]):
                        put(src, f'{rel[:-4]}#{k}', v, a.shape[0], kind(k))
        elif re.search(r'\.(csv|csv\.gz|parquet)$', low) and not SKIPF.search(low) and '_fe' not in low and 'feature' not in low:
            if os.path.getsize(p) > 6e8:
                continue
            df = pd.read_parquet(p) if low.endswith('.parquet') else pd.read_csv(p)
            if 'fold' in df.columns and len(df) == NTR:
                f = df['fold'].values
                if 'id' in df.columns:
                    f = df.set_index('id').loc[tr.id.values, 'fold'].values
                if not all(len(set(f[fid == k])) == 1 for k in range(5)):
                    split_bad.add(src)
            frame(src, rel, df)
    except Exception as e:
        print('skip', src, rel, repr(e)[:120], flush=True)

for p in glob.glob(f'{IN}/**/*_f0_val_idx.npy', recursive=True):
    src, rel = src_of(p)
    base = p[:-len('_f0_val_idx.npy')]
    for oo, tt in (('_oof.npy', '_test.npy'), ('_oof_scores.npy', '_test_scores.npy')):
        if not os.path.exists(f'{base}_f0{oo}'):
            continue
        try:
            o, t = np.full(NTR, np.nan), []
            for k in range(5):
                ix = np.load(f'{base}_f{k}_val_idx.npy')
                o[ix] = vec(np.load(f'{base}_f{k}{oo}'))
                t.append(rankdata(vec(np.load(f'{base}_f{k}{tt}'))))
            if not all(np.array_equal(np.sort(np.load(f'{base}_f{k}_val_idx.npy')), np.sort(FOLDS[k][1])) for k in range(5)):
                split_bad.add(src)
            if good(o, NTR):
                put(src, os.path.basename(base) + oo[:-4], o, NTR, 'F')
                put(src, os.path.basename(base) + oo[:-4], np.mean(t, 0), NTE, 'F')
        except Exception as e:
            print('skip folds', base, repr(e)[:120], flush=True)

print('candidates', sum(len(v) for v in C.values()), 'sources', len(C), 'split_bad', split_bad, f'{time.time() - t0:.0f}s', flush=True)

rk = lambda v: (rankdata(v) / len(v)).astype(np.float32)
M = []
for src, L in C.items():
    oo = [(n, v) for s, n, v, _ in L if s == 'o']
    tt = [(n, v, g) for s, n, v, g in L if s == 't']
    used = set()
    for n, v in oo:
        k = norm(n)
        hit = [(m, w, g) for m, w, g in tt if norm(m) == k]
        if not hit and len(oo) == 1 and len(tt) >= 1:
            hit = tt if len(tt) == 1 else [x for x in tt if norm(x[0]) == ''] or []
        if not hit:
            continue
        M.append((src, n, v, np.mean([rankdata(w) for _, w, _ in hit], 0), '+'.join(sorted(g for _, _, g in hit))))

print('pairs', len(M), flush=True)
rows, O, T = [], [], []
for src, n, o, t, g in M:
    a = auc(y, o)
    if a < 0.5:
        o, t, a = -o, -t, 1 - a
    fa = [auc(y[b], o[b]) for _, b in FOLDS]
    rows.append(dict(src=src, name=n, auc=a, fmin=min(fa), fmax=max(fa), test=g, split_ok=src not in split_bad))
    O.append(rk(o))
    T.append(rk(t))
meta = pd.DataFrame(rows)
O, T = np.column_stack(O), np.column_stack(T)
print('raw members', O.shape, flush=True)

keep = (meta.auc >= 0.93).values & (meta.auc <= 0.9535).values
z = lambda A: (A - A.mean(0)) / A.std(0)
cons_o = np.median(O[:, keep & (meta.auc > 0.948).values], 1)
cons_t = np.median(T[:, keep & (meta.auc > 0.948).values], 1)
co = np.array([np.corrcoef(O[:, j], cons_o)[0, 1] for j in range(O.shape[1])])
ct = np.array([np.corrcoef(T[:, j], cons_t)[0, 1] for j in range(T.shape[1])])
meta['co'], meta['ct'] = co, ct
keep &= (np.abs(co - ct) < 0.03) & (ct > 0.7)
meta['keep'] = keep
idx = np.where(keep)[0]
print('kept', len(idx), f'{time.time() - t0:.0f}s', flush=True)
meta.to_csv(f'{OUT}/meta.csv', index=False)
np.save(f'{OUT}/O.npy', O[:, idx])
np.save(f'{OUT}/T.npy', T[:, idx])
meta.iloc[idx].reset_index(drop=True).to_csv(f'{OUT}/members.csv', index=False)
print(meta.iloc[idx].sort_values('auc', ascending=False).head(60).to_string(), flush=True)
print(meta.iloc[idx].groupby('src').size().sort_values(ascending=False).to_string(), flush=True)
print('done', f'{time.time() - t0:.0f}s')
