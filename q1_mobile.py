"""Q1: case (e) offline meter-reading app. Two-state Markov link, 1 reading/min for an 8 h shift (480 readings).
Mean outage length fixed at 3 min (assumed coverage-hole length); availability a in {0.9,0.6,0.3}.
Strategy A: send immediately (reading lost if link down at generation).
Strategy B: store-and-forward queue (cap 60 readings, drop-oldest on overflow), flush whenever link is up.
After the shift the app keeps trying for 60 more minutes (grace period); anything still queued is 'lost'."""
import numpy as np
from common import *
N, GRACE, CAP, MEAN_DOWN, RUNS = 480, 60, 60, 3.0, 300
def link(n, a, rng):
    p_up = 1/MEAN_DOWN; p_down = (1-a)/(a*MEAN_DOWN)
    s = np.empty(n, bool); s[0] = rng.random() < a
    for t in range(1, n):
        s[t] = (rng.random() >= p_down) if s[t-1] else (rng.random() < p_up)
    return s
rows, agg = [], {}
for a in (0.9, 0.6, 0.3):
    for strat in ('immediate', 'store_forward'):
        res = []
        for run in range(RUNS):
            rng = np.random.default_rng(10_000 + run)      # same link trace for both strategies
            s = link(N + GRACE, a, rng)
            ontime = delayed = lost = 0; delays = []
            if strat == 'immediate':
                for t in range(N):
                    if s[t]: ontime += 1; delays.append(0)
                    else: lost += 1
            else:
                q = []
                for t in range(N + GRACE):
                    if t < N:
                        if s[t]: ontime += 1; delays.append(0)
                        else:
                            q.append(t)
                            if len(q) > CAP: q.pop(0); lost += 1
                    if s[t] and q:
                        for g in q: delayed += 1; delays.append(t - g)
                        q = []
                lost += len(q)
            md = float(np.mean(delays)) if delays else 0.0
            md_delayed = float(np.mean([d for d in delays if d > 0])) if delayed else 0.0
            rows.append([a, strat, run, ontime, delayed, lost, round(md, 3), round(md_delayed, 3)])
            res.append((ontime, delayed, lost, md, md_delayed))
        r = np.array(res).mean(0)
        agg[f'{a}|{strat}'] = dict(ontime=r[0], delayed=r[1], lost=r[2], mean_delay=r[3], mean_delay_of_delayed=r[4],
                                   delivery_pct=100*(r[0]+r[1])/N)
write_csv('q1_store_forward.csv', rows, ['availability','strategy','run','delivered_on_time','delivered_delayed','lost','mean_delay_min','mean_delay_of_delayed_min'])
save_json('q1_summary.json', agg)
for k, v in agg.items(): print(k, {x: round(y, 2) for x, y in v.items()})
