"""Q6: weighted decision matrix + +/-20% sensitivity. Ratings 1-5 (higher = better), each traced to Q2-Q5 evidence (see report)."""
import numpy as np, itertools
from common import *
TECH = ['SDMA', 'TDMA', 'FDMA', 'CDMA', 'OFDMA']
CRIT = ['C1 capacity (continuous traffic)', 'C2 efficiency, bursty/small reports', 'C3 interference & jam resistance', 'C4 high rate / frequency-selective channel', 'C5 device simplicity & signalling cost', 'C6 scalability in users']
RATE = {  #           C1 C2 C3 C4 C5 C6
 'SDMA':  [4, 2, 3, 2, 1, 3],
 'TDMA':  [3, 2, 2, 2, 3, 2],
 'FDMA':  [4, 1, 2, 2, 4, 1],
 'CDMA':  [3, 3, 5, 3, 2, 4],
 'OFDMA': [5, 3, 2, 5, 3, 4]}
DEP = {
 '1 Wide-area cellular voice':   [0.30, 0.05, 0.15, 0.10, 0.20, 0.20],
 '2 Jam-resistant tactical link':[0.10, 0.05, 0.50, 0.10, 0.10, 0.15],
 '3 High-rate stadium downlink': [0.25, 0.05, 0.10, 0.35, 0.10, 0.15],
 '4 5,000 low-rate sensors':     [0.05, 0.40, 0.10, 0.05, 0.25, 0.15]}
M = np.array([RATE[t] for t in TECH], float)
res = {}; rows = []
for dep, w in DEP.items():
    w = np.array(w); sc = M @ w; rank = np.argsort(-sc)
    base = TECH[rank[0]]; flips = []
    for c in range(6):
        for f in (0.8, 1.2):
            w2 = w.copy(); w2[c] *= f; w2 /= w2.sum(); s2 = M @ w2; top = TECH[int(np.argmax(s2))]
            gap = float(np.sort(s2)[-1] - np.sort(s2)[-2])
            rows.append([dep, CRIT[c][:2], f, top, round(gap, 3)])
            if top != base: flips.append((CRIT[c][:2], f, top))
    rng = np.random.default_rng(6); keep = 0
    for _ in range(5000):
        w2 = w*rng.uniform(0.8, 1.2, 6); w2 /= w2.sum(); keep += TECH[int(np.argmax(M @ w2))] == base
    res[dep] = dict(stable_pct_all_weights=100*keep/5000, weights=w.tolist(), scores={t: float(s) for t, s in zip(TECH, sc)}, best=base, runner_up=TECH[rank[1]],
                    margin=float(sc[rank[0]]-sc[rank[1]]), flips=flips, n_cases=12)
write_csv('q6_sensitivity.csv', rows, ['deployment','criterion','weight_factor','winner','margin_to_runner_up'])
save_json('q6_summary.json', dict(ratings=RATE, criteria=CRIT, res=res))
for d, r in res.items(): print(d, {k: round(v, 2) for k, v in r['scores'].items()}, r['best'], round(r['margin'], 2), r['flips'], r['stable_pct_all_weights'])
