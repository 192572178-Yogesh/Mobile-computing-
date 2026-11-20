"""Q5: SDMA with N-element uniform linear array, d = lambda/2, uniform weights steered to theta0."""
import numpy as np
from common import *
d = 0.5; TH = np.linspace(-90, 90, 36001)
def AF(N, th, th0):                       # |AF| normalised so that AF(th0)=1
    psi = 2*np.pi*d*(np.sin(np.radians(th)) - np.sin(np.radians(th0)))
    n = np.arange(N)
    return np.abs(np.exp(1j*np.outer(psi, n)).sum(1))/N
def metrics(N, th0):
    a = AF(N, TH, th0); p = 20*np.log10(np.maximum(a, 1e-9)); i0 = np.argmin(np.abs(TH - th0))
    j = i0
    while j < len(TH)-1 and a[j] > 1/np.sqrt(2): j += 1
    k = i0
    while k > 0 and a[k] > 1/np.sqrt(2): k -= 1
    hp = TH[j] - TH[k]
    j = i0
    while j < len(TH)-1 and a[j+1] < a[j]: j += 1
    k = i0
    while k > 0 and a[k-1] < a[k]: k -= 1
    bnn = TH[j] - TH[k]
    side = np.concatenate([a[:k], a[j:]]); sll = 20*np.log10(side.max())
    return hp, bnn, sll
THR = 10.0   # dB, assumed SIR threshold
rows = []; out = {}
for th0 in (0, 30):
    for N in (2, 4, 8, 16):
        hp, bnn, sll = metrics(N, th0)
        sep = np.arange(0.1, 90-th0, 0.1); sir = -20*np.log10(np.maximum(AF(N, th0+sep, th0), 1e-9))
        ok = sir >= THR; bad = np.flatnonzero(~ok); mins = (float(sep[bad[-1]+1]) if bad[-1]+1 < len(sep) else float('nan')) if len(bad) else float(sep[0])
        first = float(sep[np.argmax(ok)]) if ok.any() else float('nan')
        sirs = {int(s): float(-20*np.log10(max(AF(N, np.array([th0+s]), th0)[0], 1e-9))) for s in range(5, 61, 5) if th0+s <= 90}
        out[f'{N}|{th0}'] = dict(hpbw=hp, bwnn=bnn, sll=sll, min_sep=mins, first_cross=first, sir=sirs, theory_hpbw=101.6/N/np.cos(np.radians(th0)))
        rows.append([N, th0, round(hp, 2), round(bnn, 2), round(sll, 2), round(first, 1), round(mins, 1)] + [round(sirs.get(s, float('nan')), 1) for s in range(5, 61, 5)])
write_csv('q5_sdma_summary.csv', rows, ['N','steer_deg','HPBW_deg','first_null_BW_deg','peak_sidelobe_dB','first_sep_meeting_SIR_deg','min_sep_always_meeting_SIR_deg']+[f'SIR_dB_at_{s}deg' for s in range(5, 61, 5)])
# SIR curves (fine) for plotting, steer 0
curves = [[N, round(s, 1), float(-20*np.log10(max(AF(N, np.array([s]), 0)[0], 1e-9)))] for N in (2, 4, 8, 16) for s in np.arange(0.5, 60.01, 0.5)]
write_csv('q5_sir_curves.csv', curves, ['N','separation_deg','SIR_dB'])
# angular spread of user A's arrival (Gaussian sigma), beam fixed at nominal A; B at 15 deg; mean SIR (power ratio of means)
rng = np.random.default_rng(55); spread = {}
for N in (8, 16):
    for sg in (0, 1, 2, 5):
        a = rng.normal(0, sg, 20000) if sg else np.zeros(1); b = rng.normal(0, sg, 20000) if sg else np.zeros(1)
        sig = np.mean(AF(N, a, 0)**2); intf = np.mean(AF(N, 15 + b, 0)**2)
        spread[f'{N}|{sg}'] = float(10*np.log10(sig/intf))
out['spread_sir_B15deg'] = spread
# toll plaza geometry: lateral lane separation w at range r
plaza = {f'w={w}m,r={r}m': float(np.degrees(np.arctan2(w, r))) for w in (3.5, 7.0) for r in (30, 60, 120)}
out['plaza_sep_deg'] = plaza; save_json('q5_summary.json', out)
for k, v in out.items():
    if '|' in k and not k.startswith(('8|','16|')) or True: pass
for k in [f'{N}|0' for N in (2,4,8,16)]+[f'{N}|30' for N in (2,4,8,16)]: print(k, {a: (round(b,2) if not isinstance(b,dict) else '') for a, b in out[k].items()})
print(spread); print(plaza)
