import os, sys, time, glob, math, subprocess
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

E = os.environ.get
J = [{'NAME': 'a', 'D': 256, 'L': 6, 'H': 8, 'EP': 6, 'BS': 512, 'LR': 0.001, 'P': 0.1, 'FOLDS': 0, 'NVAL': 40000, 'K': 4}, {'NAME': 'b', 'D': 384, 'L': 8, 'H': 8, 'EP': 6, 'BS': 512, 'LR': 0.0007, 'P': 0.1, 'FOLDS': 0, 'NVAL': 40000, 'K': 4}]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
D2 = ['amount_usd', 'hours_since_last_txn', 'distance_from_home_km', 'account_balance_usd']
D1 = ['velocity_score', 'merchant_risk_score']
PAD, END, LAB, DOT, DIG = 0, 1, 2, 4, 5
LMAX = 10


def fmt(c, s):
    if c in D2:
        return f'{float(s):.2f}'
    if c in D1:
        return f'{float(s):.1f}'
    return s


def find(n, h):
    r = sorted(p for p in glob.glob(f'{E("IN", "/kaggle/input")}/**/{n}', recursive=True) if h in p)
    return min(r, key=len)


def build():
    rd = lambda n: pd.read_csv(find(n, 'playground-series-s6e11'), dtype=str, keep_default_na=False)
    tr, te = rd('train.csv'), rd('test.csv')
    cols = [c for c in tr.columns if c not in ('id', 'is_fraud')]
    al = pd.concat([tr[cols], te[cols]], ignore_index=True)
    nxt, X = DIG + 10, []
    for j, c in enumerate(cols):
        cd, un = pd.factorize(al[c])
        num = all(ch.isdigit() or ch == '.' for v in un for ch in v)
        if num:
            ids = [[DOT if ch == '.' else DIG + int(ch) for ch in fmt(c, v)] for v in un]
        else:
            ids = [[nxt + i] for i in range(len(un))]
            nxt += len(un)
        t = np.full((len(un), LMAX), -1, np.int16)
        for i, x in enumerate(ids):
            t[i, 1:1 + len(x)] = x
        X.append(t[cd])
    X = np.stack(X, 1)
    col0 = nxt
    X[:, :, 0] = col0 + np.arange(len(cols))
    y = tr.is_fraud.astype(float).astype(int).values
    return X, y, len(tr), col0, col0 + len(cols)


class Rows:
    def __init__(s, X, nv):
        s.t = torch.from_numpy(X).to(DEV)
        s.nv, s.nc = nv, X.shape[1]
        s.len = torch.from_numpy((X >= 0).sum((1, 2)) + 1).to(DEV)

    def mk(s, idx, lab, gen):
        B = idx.numel()
        p = torch.rand(B, s.nc, device=DEV, generator=gen).argsort(1)
        seg = s.t[idx].long().gather(1, p[:, :, None].expand(-1, -1, LMAX)).reshape(B, -1)
        ok = seg >= 0
        T = int(s.len[idx].max())
        pos = torch.where(ok, ok.cumsum(1), torch.full_like(seg, T + 1))
        buf = torch.full((B, T + 2), PAD, dtype=torch.long, device=DEV)
        buf.scatter_(1, pos, seg.clamp_min(0))
        buf = buf[:, :T + 1]
        buf[:, 0] = LAB + lab
        x, nx = buf[:, :-1], buf[:, 1:]
        tg = torch.where(nx < s.nv, nx, torch.full_like(nx, END))
        tg = torch.where(nx == PAD, torch.where(x == PAD, torch.full_like(nx, -100), torch.full_like(nx, END)), tg)
        tg[:, 0] = -100
        return x.contiguous(), tg


class Blk(nn.Module):
    def __init__(s, d, h, p):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.f = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.h, s.p, s.dr = h, p, nn.Dropout(p)

    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=s.p if s.training else 0.0, is_causal=True)
        x = x + s.dr(s.o(a.transpose(1, 2).reshape(B, T, D)))
        return x + s.dr(s.f(s.n2(x)))


class LM(nn.Module):
    def __init__(s, ntok, nout, d, L, h, p, T=96):
        super().__init__()
        s.e, s.pe = nn.Embedding(ntok, d), nn.Embedding(T, d)
        s.b = nn.ModuleList(Blk(d, h, p) for _ in range(L))
        s.n, s.hd, s.dr = nn.LayerNorm(d), nn.Linear(d, nout, bias=False), nn.Dropout(p)

    def forward(s, x):
        h = s.dr(s.e(x) + s.pe.weight[:x.shape[1]])
        for b in s.b:
            h = b(h)
        return s.hd(s.n(h)).float()


def train(m, R, y, rows, ep, bs, lr, seed, log, cb=None):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    sc = torch.amp.GradScaler()
    per = len(rows) // bs
    tot, gen, yd = per * ep, torch.Generator(device=DEV), torch.as_tensor(y, device=DEV)
    rs = np.random.RandomState(seed)
    t0, st = time.time(), 0
    for e in range(ep):
        m.train()
        gen.manual_seed(seed * 100 + e)
        o = torch.as_tensor(rs.permutation(rows)[:per * bs].reshape(per, bs), device=DEV)
        run = torch.zeros((), device=DEV)
        for i in range(per):
            for g in opt.param_groups:
                g['lr'] = lr * min(1, (st + 1) / 300) * 0.5 * (1 + math.cos(math.pi * st / tot))
            idx = o[i]
            x, tg = R.mk(idx, yd[idx], gen)
            with torch.autocast('cuda', dtype=torch.float16):
                lg = m(x)
            loss = F.cross_entropy(lg.view(-1, lg.shape[-1]), tg.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            sc.scale(loss).backward()
            sc.unscale_(opt)
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            sc.step(opt)
            sc.update()
            run += loss.detach()
            st += 1
        log(f'ep {e + 1}/{ep} loss {run.item() / per:.4f} {(time.time() - t0) / 60:.1f}m' + (cb() if cb else ''))


@torch.no_grad()
def score(m, R, rows, K, seed, bs=1024):
    m.eval()
    out = np.zeros((len(rows), K, 2), np.float32)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rt = torch.as_tensor(rows, device=DEV)
    for k in range(K):
        for s in range(0, len(rows), bs):
            idx = rt[s:s + bs]
            n = idx.numel()
            x0, tg = R.mk(idx, torch.zeros(n, dtype=torch.long, device=DEV), gen)
            x1 = x0.clone()
            x1[:, 0] = LAB + 1
            with torch.autocast('cuda', dtype=torch.float16):
                lg = m(torch.cat([x0, x1]))
            nll = F.cross_entropy(lg.view(-1, lg.shape[-1]), tg.repeat(2, 1).reshape(-1), ignore_index=-100,
                                  reduction='none').view(2 * n, -1).sum(1)
            out[s:s + n, k, 0] = -nll[:n].cpu().numpy()
            out[s:s + n, k, 1] = -nll[n:].cpu().numpy()
    return out


def bayes(ll, pr):
    return (np.logaddexp.reduce(ll[:, :, 1].astype(np.float64), 1) - np.logaddexp.reduce(ll[:, :, 0].astype(np.float64), 1)
            + np.log(pr / (1 - pr)))


def main():
    out = E('OUT', '/kaggle/working')
    os.makedirs(out, exist_ok=True)
    lf = open(f'{out}/log.txt', 'a')

    def log(*a):
        s = time.strftime('%H:%M:%S ') + ' '.join(map(str, a))
        print(s, flush=True)
        lf.write(s + '\n')
        lf.flush()

    X, y, ntr, col0, ntok = build()
    R = Rows(X, col0)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))
    d, L, h, p = int(E('D', 256)), int(E('L', 6)), int(E('H', 8)), float(E('P', 0.1))
    ep, bs, lr = int(E('EP', 10)), int(E('BS', 512)), float(E('LR', 1e-3))
    K, KT, seed = int(E('K', 8)), int(E('KT', 4)), int(E('SEED', 0))
    sub, nval = float(E('SUB', 1)), int(E('NVAL', 0))
    yy = np.r_[y, np.zeros(len(X) - ntr, int)]
    log(f'ntok {ntok} d{d} L{L} h{h} p{p} ep{ep} bs{bs} lr{lr} K{K} seed{seed}')
    for f in [int(v) for v in E('FOLDS', '0,1,2,3,4').split(',')]:
        if os.path.exists(f'{out}/te_{f}.npy'):
            continue
        a, b = folds[f]
        rs = np.random.RandomState(seed + f)
        if sub < 1:
            a = rs.choice(a, int(len(a) * sub), replace=False)
        if nval:
            b = np.r_[b[y[b] == 1], rs.choice(b[y[b] == 0], nval, replace=False)]
        torch.manual_seed(seed * 10 + f)
        m = LM(ntok, col0, d, L, h, p).to(DEV)
        log(f'fold {f} params {sum(q.numel() for q in m.parameters()) / 1e6:.2f}M train {len(a)}')
        cb = (lambda: f' val {roc_auc_score(y[b], bayes(score(m, R, b, 2, 7), 0.5)):.5f}') if nval else None
        train(m, R, yy, a, ep, bs, lr, seed * 10 + f, log, cb)
        ll = score(m, R, b, K, seed * 10 + f + 1)
        z = bayes(ll, y[a].mean())
        log(f'fold {f} AUC {roc_auc_score(y[b], z):.5f} one-order {roc_auc_score(y[b], ll[:, 0, 1] - ll[:, 0, 0]):.5f}')
        np.save(f'{out}/oof_{f}.npy', ll)
        np.save(f'{out}/idx_{f}.npy', b)
        if not nval:
            np.save(f'{out}/te_{f}.npy', score(m, R, np.arange(ntr, len(X)), KT, seed * 10 + f + 2))
        torch.save(m.state_dict(), f'{out}/m_{f}.pt')
        del m
        torch.cuda.empty_cache()


def launch():
    ng = max(1, torch.cuda.device_count())
    ps = []
    for i, j in enumerate(J):
        env = {**os.environ, 'ROLE': '1', 'CUDA_VISIBLE_DEVICES': str(i % ng), 'OUT': f'/kaggle/working/{j.get("NAME", i)}'}
        env.update({k: str(v) for k, v in j.items()})
        ps.append(subprocess.Popen([sys.executable, os.path.abspath(sys.argv[0])], env=env))
    sys.exit(max(p.wait() for p in ps))


if __name__ == '__main__':
    main() if E('ROLE') or not J else launch()
