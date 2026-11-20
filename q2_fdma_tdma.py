"""Q2: dedicated-channel allocation, FDMA (76 usable channels) vs TDMA (70 usable slots). 2 MHz per direction.
Nominal capacity = 80 channel-units. Wasted capacity is decomposed so that all parts sum to 80*T."""
import numpy as np, heapq
from common import *
T, SEEDS, NOMINAL = 20000.0, 10, 80
SCHEMES = {  # C usable, payload fraction, setup time (s, channel held but unused)  [assumed values justified in report]
 'FDMA': dict(C=76, payload=0.95, setup=0.4),
 'TDMA': dict(C=70, payload=114/156.25, setup=0.6)}
# datasets: name -> (arrival spec, list of (prob, mean talk s, duty))
DATASETS = {
 'A_voice':  (('poisson', 65/120), [(1.0, 120, 1.0)]),
 'B_bursty': (('poisson', 65/120), [(1.0, 120, 0.1)]),
 'C_mix7030':(('poisson', 65/120), [(0.7, 120, 1.0), (0.3, 120, 0.1)]),
 'D_edge_sync_short': (('batch', (60.0, 90)), [(1.0, 15, 1.0)])}   # 90 call attempts at the same instant every 60 s
def arrivals(spec, rng):
    if spec[0] == 'poisson':
        n = rng.poisson(spec[1]*T); return np.sort(rng.uniform(0, T, n))
    per, size = spec[1]; return np.repeat(np.arange(per, T, per), size)
def run(scheme, ds, seed):
    p = SCHEMES[scheme]; spec, mix = DATASETS[ds]
    rng = np.random.default_rng(seed)                       # same arrivals/holds for both schemes
    arr = arrivals(spec, rng); n = len(arr)
    kind = rng.choice(len(mix), n, p=[m[0] for m in mix])
    talk = rng.exponential(np.array([mix[k][1] for k in kind]))
    duty = np.array([mix[k][2] for k in kind])
    heap = []; busy = setup_t = silent = active = 0.0; acc = 0
    for i in range(n):
        while heap and heap[0] <= arr[i]: heapq.heappop(heap)
        if len(heap) < p['C']:
            acc += 1; hold = p['setup'] + talk[i]; heapq.heappush(heap, arr[i] + hold)
            busy += hold; setup_t += p['setup']; silent += talk[i]*(1-duty[i]); active += talk[i]*duty[i]
    tot = NOMINAL*T
    comp = dict(reserved=(NOMINAL-p['C'])*T, idle_channels=p['C']*T-busy, setup=setup_t, held_silent=silent,
                overhead=active*(1-p['payload']), useful=active*p['payload'])
    out = dict(blocking=1-acc/n, occupancy=busy/(p['C']*T), useful_util=comp['useful']/tot, attempts=n)
    out.update({k: v/tot for k, v in comp.items()})
    return out
rows, agg = [], {}
for ds in DATASETS:
    for sch in SCHEMES:
        rs = [run(sch, ds, 500+s) for s in range(SEEDS)]
        for s, r in enumerate(rs): rows.append([ds, sch, s] + [round(r[k], 5) for k in ('blocking','occupancy','useful_util','reserved','idle_channels','setup','held_silent','overhead','useful')])
        agg[f'{ds}|{sch}'] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
write_csv('q2_fdma_tdma.csv', rows, ['dataset','scheme','seed','blocking','occupancy','useful_util','reserved','idle_channels','setup','held_silent','overhead','useful'])
# Design decision: telemetry via slotted ALOHA on ONE shared 50 kbps channel (assumed), 1000 devices (assumed), 200 B + 20 B header per minute
rng = np.random.default_rng(77); NDEV, PKT_BITS, RATE = 1000, (200+20)*8, 50_000
slot = PKT_BITS/RATE; slots_per_min = int(60/slot); MAXTRY = 6; MINUTES = 30
succ = fail = tries = 0
for m in range(MINUTES):
    pending = {d: int(rng.integers(0, slots_per_min)) for d in range(NDEV)}; att = {d: 0 for d in range(NDEV)}
    sched = {}
    for d, s in pending.items(): sched.setdefault(s, []).append(d)
    s = 0
    while sched and s < slots_per_min + 20000:
        tx = sched.pop(s, [])
        if tx:
            for d in tx: att[d] += 1; tries += 1
            if len(tx) == 1: succ += 1
            else:
                for d in tx:
                    if att[d] < MAXTRY: sched.setdefault(s+1+int(rng.integers(1, min(100*2**att[d], 1500))), []).append(d)
                    else: fail += 1
        s += 1
tot = NDEV*MINUTES
aloha = dict(devices=NDEV, slot_ms=slot*1000, slots_per_min=slots_per_min, G=NDEV/slots_per_min, delivered=succ/tot, failed=fail/tot,
             mean_attempts=tries/tot, duty_per_device=PKT_BITS/RATE/60, dedicated_channels_needed=NDEV)
agg['aloha'] = aloha; save_json('q2_summary.json', agg)
for k, v in agg.items(): print(k, {x: round(y, 4) for x, y in v.items()})
