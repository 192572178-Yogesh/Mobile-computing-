"""Q7: bursty sessions on a 70-slot TDMA cell. Original = dedicated slot per session. Redesign = statistical admission + burst-level slot pool.
Time step dt=0.1 s. Session: Poisson arrivals, exponential holding (mean 120 s), ON/OFF activity: ON mean 1 s, OFF mean 9 s (duty 10%)."""
import numpy as np
from scipy.stats import binom
from common import *
C, DT, HOLD, ON_M, OFF_M, EPS = 70, 0.1, 120.0, 1.0, 9.0, 0.01
DUTY = ON_M/(ON_M+OFF_M)
NMAX = int(max(n for n in range(C, 2000) if binom.sf(C, n, DUTY) <= EPS))     # admission limit: P(active > C) <= 1%
def sim(mode, erl, T=4200.0, warm=600.0, seed=1):
    rng = np.random.default_rng(seed); M = 900
    alive = np.zeros(M, bool); on = np.zeros(M, bool); has = np.zeros(M, bool); waiting = np.zeros(M, bool); ws = np.zeros(M)
    lam = erl/HOLD; steps = int(T/DT); w0 = int(warm/DT); lim = C if mode == 'original' else NMAX
    arr = adm = 0; served = occ = demand = 0.0; bursts = req_msgs = lost = 0; delays = []; nalive_sum = 0.0
    for st in range(steps):
        t = st*DT; rec = st >= w0
        dep = alive & (rng.random(M) < DT/HOLD)
        if dep.any():
            lost += int((dep & waiting).sum()) if rec else 0
            alive[dep] = on[dep] = has[dep] = waiting[dep] = False
        for _ in range(rng.poisson(lam*DT)):
            if rec: arr += 1
            if alive.sum() < lim:
                i = int(np.flatnonzero(~alive)[0]); alive[i] = True; on[i] = rng.random() < DUTY; has[i] = False; waiting[i] = False
                if rec: adm += 1
                if on[i] and mode == 'redesign': waiting[i] = True; ws[i] = t; req_msgs += rec
                if on[i] and rec: bursts += 1
        r = rng.random(M)
        off_ev = alive & on & (r < DT/ON_M); on_ev = alive & ~on & (r < DT/OFF_M)
        on[off_ev] = False; on[on_ev] = True
        if mode == 'redesign':
            lost += int((off_ev & waiting).sum()) if rec else 0
            waiting[off_ev] = False; has[off_ev] = False                      # release at end of burst (H=0)
            nb = np.flatnonzero(on_ev)
            waiting[nb] = True; ws[nb] = t
            if rec: bursts += len(nb); req_msgs += len(nb)
            free = C - int(has.sum()); w = np.flatnonzero(waiting)
            if len(w) and free > 0:
                g = w[np.argsort(ws[w])][:free]; has[g] = True; waiting[g] = False
                if rec: delays.extend((t - ws[g] + DT).tolist())              # +1 step request/grant latency
        else:
            has[:] = alive
            if rec: bursts += int(on_ev.sum())
        if rec:
            served += float((on & has & alive).sum()); occ += float(has.sum()); demand += float((on & alive).sum()); nalive_sum += float(alive.sum())
    n = steps - w0; d = np.array(delays) if delays else np.array([0.0])
    msgs = 2*adm + (req_msgs*2 if mode == 'redesign' else 0)
    return dict(offered_erl=erl, arrivals=arr, admitted=adm, blocking=1-adm/max(arr, 1), mean_sessions=nalive_sum/n, useful_util=served/n/C,
                slot_occupancy=occ/n/C, demand_met=served/max(demand, 1e-9), mean_burst_wait_s=float(d.mean()) if mode == 'redesign' else 0.0,
                p_wait_gt_0p5s=float((d > 0.5).mean()) if mode == 'redesign' else 0.0, p95_wait_s=float(np.percentile(d, 95)) if mode == 'redesign' else 0.0,
                lost_bursts_pct=100*lost/max(bursts, 1), msgs_per_session=msgs/max(adm, 1))
rows = []; agg = {}
LOADS = [30, 60, 90, 150, 250, 400, 500, 650]
for erl in LOADS:
    for mode in ('original', 'redesign'):
        rs = [sim(mode, erl, seed=700+s) for s in range(3)]
        m = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}; agg[f'{erl}|{mode}'] = m
        for s, r in enumerate(rs): rows.append([erl, mode, s] + [round(r[k], 5) for k in list(rs[0])[1:]])
        print(erl, mode, {k: round(v, 3) for k, v in m.items()}, flush=True)
write_csv('q7_original_vs_redesign.csv', rows, ['offered_erl', 'mode', 'seed'] + list(rs[0])[1:])
agg['params'] = dict(C=C, NMAX=NMAX, duty=DUTY, eps=EPS); save_json('q7_summary.json', agg); print('NMAX', NMAX)
