"""Q3: DS-CDMA uplink, BPSK data, spreading code length L, AWGN Eb/N0 = 10 dB, user 0 is the desired (unit-power) user.
20,000 bits per run for user 0. Conditions: walsh_sync, random_sync, walsh_sync_10x, walsh_async, walsh_async_10x, random_sync_10x.
'10x' = user 1 transmits at 10x power. 'async' = each interferer has a random non-zero chip offset."""
import numpy as np
from common import *
NB, CH, EBN0 = 20000, 1000, 10**(10/10)
def hadamard(n):
    H = np.array([[1]])
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H
def codes_for(L, K, kind, rng):
    if kind == 'walsh':
        H = hadamard(L); order = list(range(1, L)) + [0]; return H[order[:K]].astype(float)
    return rng.choice([-1.0, 1.0], size=(K, L))
def run(L, K, cond, seed):
    kind = 'walsh' if cond.startswith('walsh') else 'random'
    asyn = 'async' in cond; strong = '10x' in cond
    rng = np.random.default_rng(seed)
    amp = np.ones(K)
    if strong and K >= 2: amp[1] = np.sqrt(10)
    sigma = np.sqrt(L/(2*EBN0)); err = 0
    for _ in range(NB//CH):                      # fresh random code set / chip offsets for every 1000-bit block (averages over code draws)
        C = codes_for(L, K, kind, rng)
        off = rng.integers(1, L, K) if asyn else np.zeros(K, int); off[0] = 0
        b = rng.choice([-1.0, 1.0], size=(K, CH))
        chips = (b[:, :, None]*C[:, None, :]).reshape(K, CH*L)*amp[:, None]
        for i in range(1, K):
            if off[i]: chips[i] = np.roll(chips[i], off[i])
        rx = chips.sum(0) + rng.normal(0, sigma, CH*L)
        y = rx.reshape(CH, L) @ C[0]/L
        err += int(np.sum(np.where(y >= 0, 1.0, -1.0) != b[0]))
    return err
KS = {8: [1,2,3,4,6,8], 16: [1,2,3,4,6,8,12,16], 64: [1,2,4,6,8,10,12,16,24,32,48,64]}
CONDS = ['walsh_sync','random_sync','walsh_sync_10x','walsh_async','walsh_async_10x','random_sync_10x']
rows = []; ber = {}
for L, ks in KS.items():
    for ci, cond in enumerate(CONDS):
        for K in ks:
            e = run(L, K, cond, 3000 + 100*L + 10*ci + K)
            rows.append([L, cond, K, NB, e, e/NB, round(10*np.log10(L), 2)]); ber[(L, cond, K)] = e/NB
write_csv('q3_cdma_ber.csv', rows, ['L','condition','users','bits','errors','BER','processing_gain_dB'])
# correlation values (normalised periodic cross-correlation of user 0 code with others)
crow = []; csum = {}
for L in KS:
    for kind in ('walsh', 'random'):
        rng = np.random.default_rng(99+L); C = codes_for(L, min(L, 16), kind, rng)
        auto = float(C[0]@C[0]/L)
        zero = [abs(C[0]@C[i]/L) for i in range(1, len(C))]
        def apc(c0, ci):   # rms aperiodic cross-correlation per bit over all chip offsets 1..L-1
            v = []
            for t in range(1, L):
                p1 = np.dot(ci[L-t:], c0[:t]); p2 = np.dot(ci[:L-t], c0[t:]); v.append(np.sqrt((p1**2+p2**2)/2)/L)
            return v
        mx = [np.mean(apc(C[0], C[i])) for i in range(1, len(C))]
        mxx = [np.max(apc(C[0], C[i])) for i in range(1, len(C))]
        crow.append([L, kind, auto, float(np.mean(zero)), float(np.max(zero)), float(np.mean(mx)), float(np.max(mxx)), round(1/np.sqrt(L), 3)])
        csum[f'{L}|{kind}'] = dict(auto=auto, zero_mean=float(np.mean(zero)), zero_max=float(np.max(zero)), shift_mean=float(np.mean(mx)), shift_max=float(np.max(mxx)))
write_csv('q3_correlation.csv', crow, ['L','codes','autocorr_zero_shift','mean_abs_xcorr_zero','max_abs_xcorr_zero','mean_rms_aperiodic_xcorr_over_offsets','max_rms_aperiodic_xcorr','1_over_sqrtL'])
# capacity at BER target 1e-2 (equal power, random sync / walsh async)
cap = {}
for L, ks in KS.items():
    for cond in ('random_sync', 'walsh_async'):
        ok = [K for K in ks if ber[(L, cond, K)] <= 1e-2]; cap[f'{L}|{cond}'] = max(ok) if ok else 0
summ = dict(ber={f'{L}|{c}|{K}': v for (L, c, K), v in ber.items()}, corr=csum, capacity=cap)
save_json('q3_summary.json', summ); print(cap)
for L, ks in KS.items():
    print('L', L)
    for cond in CONDS: print(' ', cond.ljust(16), [f'{ber[(L,cond,K)]:.4f}' for K in ks])
for r in crow: print(r)
