import json, numpy as np, csv, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import *
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 150})
J = lambda n: json.load(open(f'{DATA}/{n}'))
# F1
q1 = J('q1_summary.json'); fig, ax = plt.subplots(figsize=(6.5, 3.2)); x = 0; ticks = []; lab = []
for a in ('0.9', '0.6', '0.3'):
    for s, nm in (('immediate', 'Immediate'), ('store_forward', 'Store-fwd')):
        r = q1[f'{a}|{s}']; ax.bar(x, r['ontime'], color='#2a9d8f', label='On time' if x == 0 else '')
        ax.bar(x, r['delayed'], bottom=r['ontime'], color='#e9c46a', label='Delayed' if x == 0 else '')
        ax.bar(x, r['lost'], bottom=r['ontime']+r['delayed'], color='#e76f51', label='Lost' if x == 0 else ''); ticks.append(x); lab.append(f'{int(float(a)*100)}%\n{nm}'); x += 1
    x += .6
ax.set_xticks(ticks); ax.set_xticklabels(lab); ax.set_ylabel('Readings (of 480)'); ax.legend(ncol=3, fontsize=8); ax.set_title('Q1: updates delivered / delayed / lost (mean of 300 runs)')
fig.tight_layout(); fig.savefig(f'{FIG}/q1.png'); plt.close()
# F2
q2 = J('q2_summary.json'); comps = ['useful', 'overhead', 'setup', 'held_silent', 'idle_channels', 'reserved']; cols = ['#2a9d8f', '#8ab17d', '#e9c46a', '#f4a261', '#bdbdbd', '#6d6875']
fig, ax = plt.subplots(figsize=(6.8, 3.4)); x = 0; tk = []; lb = []
for ds in ('A_voice', 'B_bursty', 'C_mix7030', 'D_edge_sync_short'):
    for sc in ('FDMA', 'TDMA'):
        r = q2[f'{ds}|{sc}']; b = 0
        for c, col in zip(comps, cols): ax.bar(x, 100*r[c], bottom=b, color=col, label=c if x == 0 else ''); b += 100*r[c]
        tk.append(x); lb.append(f'{ds.split("_")[0]}\n{sc}'); x += 1
    x += .5
ax.set_xticks(tk); ax.set_xticklabels(lb, fontsize=8); ax.set_ylabel('% of nominal 80-channel capacity'); ax.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.2)); ax.set_title('Q2: where the nominal capacity goes')
fig.tight_layout(); fig.savefig(f'{FIG}/q2.png'); plt.close()
# F3
rows = list(csv.DictReader(open(f'{DATA}/q3_cdma_ber.csv'))); conds = ['walsh_sync', 'random_sync', 'walsh_sync_10x', 'walsh_async', 'walsh_async_10x', 'random_sync_10x']
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
for ax, L in zip(axs, (8, 16, 64)):
    for c in conds:
        pts = [(int(r['users']), max(float(r['BER']), 2e-5)) for r in rows if int(r['L']) == L and r['condition'] == c]
        ax.semilogy(*zip(*pts), marker='o', ms=3, label=c)
    ax.axhline(1e-2, color='k', ls=':', lw=1); ax.set_title(f'L = {L}'); ax.set_xlabel('users K')
axs[0].set_ylabel('BER (user 0, floor 2e-5)'); axs[2].legend(fontsize=6); fig.tight_layout(); fig.savefig(f'{FIG}/q3.png'); plt.close()
# F4
q4 = J('q4_summary.json'); res = q4['res']; fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for s, col in zip(('contiguous', 'interleaved', 'best_subcarrier', 'min_rate_guaranteed'), ('#8ab17d', '#2a9d8f', '#e76f51', '#264653')):
    axs[0].semilogy(range(1, 9), np.maximum(res[f'unequal_22to1dB|{s}']['per_user'], 1e-2), marker='o', ms=3, label=s, color=col)
axs[0].axhline(q4['rmin'], color='k', ls=':'); axs[0].set_xlabel('user (1 = strongest, 8 = weakest)'); axs[0].set_ylabel('mean bits / OFDM symbol'); axs[0].legend(fontsize=6); axs[0].set_title('Per-user throughput (unequal SNR)')
ks = sorted(q4['mud'], key=int); axs[1].plot([int(k) for k in ks], [q4['mud'][k]['gain_pct'] for k in ks], marker='o'); axs[1].set_xscale('log', base=2)
axs[1].set_xlabel('number of users (64 subcarriers)'); axs[1].set_ylabel('best-subcarrier gain over interleaved (%)'); axs[1].set_title('Multi-user diversity'); fig.tight_layout(); fig.savefig(f'{FIG}/q4.png'); plt.close()
# F5
sys_ = __import__('q5_sdma') if False else None
d = .5; th = np.linspace(-90, 90, 3601)
def AF(N, t, t0):
    psi = 2*np.pi*d*(np.sin(np.radians(t)) - np.sin(np.radians(t0))); return np.abs(np.exp(1j*np.outer(psi, np.arange(N))).sum(1))/N
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for N in (2, 4, 8, 16): axs[0].plot(th, 20*np.log10(np.maximum(AF(N, th, 0), 1e-3)), label=f'N={N}')
axs[0].set_ylim(-40, 2); axs[0].set_xlabel('angle (deg)'); axs[0].set_ylabel('pattern (dB)'); axs[0].legend(fontsize=7); axs[0].set_title('Array pattern, steered to 0°')
cur = list(csv.DictReader(open(f'{DATA}/q5_sir_curves.csv')))
for N in (2, 4, 8, 16):
    p = [(float(r['separation_deg']), float(r['SIR_dB'])) for r in cur if int(r['N']) == N]; axs[1].plot(*zip(*p), label=f'N={N}')
axs[1].axhline(10, color='k', ls=':'); axs[1].set_ylim(0, 40); axs[1].set_xlabel('separation Δ (deg)'); axs[1].set_ylabel('SIR at user B (dB)'); axs[1].legend(fontsize=7); axs[1].set_title('SIR vs angular separation')
fig.tight_layout(); fig.savefig(f'{FIG}/q5.png'); plt.close()
# F6
q6 = J('q6_summary.json'); fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=True); T = ['SDMA', 'TDMA', 'FDMA', 'CDMA', 'OFDMA']
for ax, (dep, r) in zip(axs, q6['res'].items()):
    v = [r['scores'][t] for t in T]; ax.bar(T, v, color=['#e76f51' if t == r['best'] else '#9aa5b1' for t in T]); ax.set_title(dep[:2] + ' ' + dep[2:].split()[0] + ' ' + dep[2:].split()[1] if False else dep.split(' ')[0], fontsize=8); ax.tick_params(axis='x', labelrotation=60, labelsize=7)
axs[0].set_ylabel('weighted score'); fig.suptitle('Q6: weighted scores (red = winner)', fontsize=9); fig.tight_layout(); fig.savefig(f'{FIG}/q6.png'); plt.close()
# F7
q7 = J('q7_summary.json'); L = [30, 60, 90, 150, 250, 400, 500, 650]; fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9))
for m, c in (('original', '#e76f51'), ('redesign', '#2a9d8f')):
    axs[0].plot(L, [q7[f'{l}|{m}']['mean_sessions'] for l in L], marker='o', ms=3, color=c, label=m)
    axs[1].plot(L, [100*q7[f'{l}|{m}']['useful_util'] for l in L], marker='o', ms=3, color=c)
    axs[2].plot(L, [100*q7[f'{l}|{m}']['blocking'] for l in L], marker='o', ms=3, color=c)
axs[0].set_title('Sessions carried'); axs[1].set_title('Useful slot utilisation (%)'); axs[2].set_title('Session blocking (%)')
for a in axs: a.set_xlabel('offered sessions (Erlang)')
axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f'{FIG}/q7_results.png'); plt.close()
# flowchart
fig, ax = plt.subplots(figsize=(7.2, 4.4)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(-0.3, 6.4)
def box(x, y, w, h, t, c='#e8f1f2'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', fc=c, ec='#264653')); ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=7.5)
def arr(x1, y1, x2, y2): ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', color='#264653'))
box(0.2, 5.2, 2.8, 0.9, 'Session request with\ntraffic descriptor (duty, burst)'); box(3.6, 5.2, 2.8, 0.9, 'Admit?  active sessions < N_max\n(P[Bin(N_max,duty) > C] ≤ 1%)', '#fff3c4'); box(7.2, 5.2, 2.6, 0.9, 'Reject (blocked)', '#f9d5d3')
arr(3.0, 5.65, 3.6, 5.65); arr(6.4, 5.65, 7.2, 5.65); ax.text(6.55, 5.8, 'no', fontsize=7)
box(3.6, 3.9, 2.8, 0.8, 'Admitted: no slot held;\nsession enters IDLE'); arr(5.0, 5.2, 5.0, 4.7); ax.text(5.1, 4.95, 'yes', fontsize=7)
box(3.6, 2.6, 2.8, 0.8, 'Burst starts → send\nREQUEST (mini-slot)', '#d7ecd9'); arr(5.0, 3.9, 5.0, 3.4)
box(7.0, 2.6, 2.8, 0.8, 'Queue FIFO by arrival time\n(slot pool full?)', '#fff3c4'); arr(6.4, 3.0, 7.0, 3.0)
box(3.6, 1.2, 2.8, 0.8, 'GRANT free slot\n(slot from pool of C)', '#d7ecd9'); arr(5.0, 2.6, 5.0, 2.0); arr(8.4, 2.6, 6.4, 1.6); ax.text(7.7, 2.0, 'slot freed', fontsize=7)
box(3.6, 0.0, 2.8, 0.8, 'Transmit burst → end-of-burst\nflag releases slot at once', '#d7ecd9'); arr(5.0, 1.2, 5.0, 0.8)
ax.annotate('', (3.55, 3.0), (3.55, 0.4), arrowprops=dict(arrowstyle='->', color='#264653', connectionstyle='arc3,rad=-0.9'))
ax.text(0.25, 1.6, 'next burst → REQUEST again', fontsize=7)
fig.tight_layout(); fig.savefig(f'{FIG}/q7_flowchart.png'); plt.close()
print('ok')
