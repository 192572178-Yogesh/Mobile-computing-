"""Q3 extra: near-far. L=64, K=8 users, random codes, synchronous, user 1 power = R x user 0 power. BER of user 0; with and without power control."""
import numpy as np
from common import *
L, K, NB, CH, EBN0 = 64, 8, 20000, 1000, 10.0
rows = []; out = {}
for kind in ('random', 'walsh_async'):
    for R_db in (0, 3, 6, 10, 15, 20, 30):
        rng = np.random.default_rng(4100 + R_db); err = 0; sigma = np.sqrt(L/(2*EBN0)); amp = np.ones(K); amp[1] = 10**(R_db/20)
        for _ in range(NB//CH):
            if kind == 'random': C = rng.choice([-1.0, 1.0], (K, L)); off = np.zeros(K, int)
            else:
                H = np.array([[1]]); 
                while H.shape[0] < L: H = np.block([[H, H], [H, -H]])
                C = H[list(range(1, K+1))].astype(float); off = rng.integers(1, L, K); off[0] = 0
            b = rng.choice([-1.0, 1.0], (K, CH)); ch = (b[:, :, None]*C[:, None, :]).reshape(K, CH*L)*amp[:, None]
            for i in range(1, K):
                if off[i]: ch[i] = np.roll(ch[i], off[i])
            rx = ch.sum(0) + rng.normal(0, sigma, CH*L); y = rx.reshape(CH, L) @ C[0]/L; err += int(np.sum(np.where(y >= 0, 1.0, -1.0) != b[0]))
        rows.append([kind, R_db, err, err/NB]); out[f'{kind}|{R_db}'] = err/NB; print(kind, R_db, err/NB)
write_csv('q3b_nearfar.csv', rows, ['codes','interferer_excess_power_dB','errors','BER']); save_json('q3b_summary.json', out)
