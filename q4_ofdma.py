"""Q4: OFDMA. Part 1 numerical orthogonality + cyclic prefix. Part 2 allocation of 64 subcarriers to 8 users over 1000 channel realisations (seed 2026)."""
import numpy as np
from common import *
# ---------- Part 1 ----------
N = 64; n = np.arange(N)
X = np.exp(2j*np.pi*np.outer([3, 4, 5], n)/N)             # subcarriers k=3,4,5 over one useful symbol
G = np.abs(X @ X.conj().T)                                  # |inner products|
frac = abs(np.sum(np.exp(2j*np.pi*4.3*n/N)*np.conj(np.exp(2j*np.pi*5*n/N))))  # tone with 0.3-bin offset leaks into bin 5 (should be 0 if orthogonal)
rng = np.random.default_rng(5)
h = np.zeros(8, complex); h[0] = 1; h[3] = 0.5; h[7] = 0.3j            # 3-path channel, delay spread 7 samples
def ofdm_evm(cp, nsym=300):
    bits = rng.integers(0, 4, (nsym, N)); S = np.exp(1j*(np.pi/4 + np.pi/2*bits))   # QPSK
    tx = np.fft.ifft(S, axis=1)*np.sqrt(N)
    txcp = np.concatenate([tx[:, N-cp:], tx], axis=1) if cp else tx
    rx = np.convolve(txcp.reshape(-1), h)[:txcp.size].reshape(nsym, N+cp)[:, cp:]
    Y = np.fft.fft(rx, axis=1)/np.sqrt(N); H = np.fft.fft(h, N)
    E = Y/H - S; return float(np.sqrt(np.mean(np.abs(E)**2)))
evm = {cp: ofdm_evm(cp) for cp in (0, 4, 7, 8, 16)}
save_json('q4_orthogonality.json', dict(gram=G.round(6).tolist(), leakage_0p3bin=float(frac), evm_vs_cp={str(k): v for k, v in evm.items()}))
print('Gram', G.round(3).tolist(), 'leak', frac, 'EVM', evm)
# ---------- Part 2 ----------
NS = 64; R = 1000
def channels(K, mean_db, rng, R=R):
    taps = 8; pdp = np.exp(-np.arange(taps)/2.0); pdp /= pdp.sum()
    h = (rng.normal(size=(R, K, taps)) + 1j*rng.normal(size=(R, K, taps)))*np.sqrt(pdp/2)
    g = np.abs(np.fft.fft(h, NS, axis=2))**2                     # |H_k|^2, mean 1
    snr = 10**(np.array(mean_db)/10)[None, :, None]
    return np.log2(1 + snr*g)                                    # bits / subcarrier / OFDM symbol
def alloc(rate, scheme, mean_db=None, rmin=None):
    R_, K, _ = rate.shape; out = np.zeros((R_, K))
    if scheme == 'contiguous':
        for u in range(K): out[:, u] = rate[:, u, u*(NS//K):(u+1)*(NS//K)].sum(1)
    elif scheme == 'interleaved':
        for u in range(K): out[:, u] = rate[:, u, u::K].sum(1)
    elif scheme == 'best_subcarrier':
        w = rate.argmax(1)
        for u in range(K): out[:, u] = np.where(w == u, rate[:, u, :], 0).sum(1)
    elif scheme == 'min_rate_guaranteed':
        order = np.argsort(mean_db)                                # weakest mean SNR served first
        for r in range(R_):
            free = np.ones(NS, bool); got = np.zeros(K)
            for u in order:
                idx = np.argsort(-rate[r, u]); 
                for k in idx:
                    if got[u] >= rmin: break
                    if free[k]: free[k] = False; got[u] += rate[r, u, k]
            fk = np.flatnonzero(free)
            if len(fk):
                w = rate[r][:, fk].argmax(0)
                for j, k in enumerate(fk): got[w[j]] += rate[r, w[j], k]
            out[r] = got
    return out
def jain(x): return float(x.sum()**2/(len(x)*np.sum(x**2)))
SCH = ['contiguous', 'interleaved', 'best_subcarrier']
scen = {'equal_10dB': [10]*8, 'unequal_22to1dB': [22, 19, 16, 13, 10, 7, 4, 1]}
RMIN = 6.0                                                       # bits per OFDM symbol (= 0.094 bit/subcarrier averaged over the whole band)
rows, res = [], {}
for name, mdb in scen.items():
    rng = np.random.default_rng(2026); rate = channels(8, mdb, rng)
    for s in SCH + ['min_rate_guaranteed']:
        th = alloc(rate, s, mdb, RMIN)
        pu = th.mean(0); sumth = th.sum(1).mean(); j_mean = jain(pu); j_real = float(np.mean([jain(t) for t in th]))
        outage = float((th < RMIN).mean()); worst = float((th[:, int(np.argmin(mdb))] < RMIN).mean())
        res[f'{name}|{s}'] = dict(sum=float(sumth), per_user=pu.tolist(), jain_of_means=j_mean, jain_per_real=j_real, outage_all=outage, outage_weakest=worst, min_user=float(pu.min()))
        rows.append([name, s, round(sumth, 3), round(j_mean, 4), round(j_real, 4), round(outage, 4)] + [round(x, 3) for x in pu])
write_csv('q4_allocation.csv', rows, ['scenario','scheme','sum_bits_per_symbol','jain_of_mean_user_throughput','mean_jain_per_realisation','outage_prob_rate_lt_Rmin']+[f'user{u}' for u in range(8)])
# multi-user diversity: equal mean SNR, K users
mud = {}; mrows = []
for K in (2, 4, 8, 16, 32):
    rng = np.random.default_rng(2026); rate = channels(K, [10]*K, rng)
    bs = alloc(rate, 'best_subcarrier').sum(1).mean(); il = alloc(rate, 'interleaved').sum(1).mean()
    mud[K] = dict(best=float(bs), interleaved=float(il), gain_pct=float(100*(bs/il-1))); mrows.append([K, round(il, 3), round(bs, 3), round(100*(bs/il-1), 2)])
write_csv('q4_multiuser_diversity.csv', mrows, ['users','interleaved','best_subcarrier','gain_pct'])
save_json('q4_summary.json', dict(res=res, mud=mud, rmin=RMIN))
for k, v in res.items(): print(k, round(v['sum'], 2), [round(x, 2) for x in v['per_user']], 'J', round(v['jain_of_means'], 3), 'out', round(v['outage_all'], 3), round(v['outage_weakest'], 3))
print(mud)
