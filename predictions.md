# Predictions written BEFORE running any simulation (fixed in this file, then code was run)

Q1  Immediate-send loses about (1-availability) of readings: ~10%, ~40%, ~70% lost.
    Store-and-forward (queue cap 60) delivers ~100% at 90% and 60%; at 30% a few readings may overflow/lose.
    Mean delay grows as availability falls (roughly the mean outage length: <1 min, ~2-3 min, ~5+ min).
Q2  FDMA usable = 76 channels, TDMA usable = 70 slots (edge guard + control). FDMA payload fraction 0.95,
    TDMA 0.73 (GSM slot: 114 data bits of 156.25). Voice: FDMA utilisation > TDMA by ~25%. Bursty (10% duty): both
    ~ 10x lower than voice; the waste is "held-but-silent". Mix: in between. Blocking at 65 Erlang: FDMA ~1-3%, TDMA ~5-8%.
    Synchronised arrivals (edge): blocking dominated by batch size > channels, utilisation very low.
Q3  (i) Walsh, synchronous, equal power: BER ~ noise floor Q(sqrt(2*10)) ~ 4e-6 -> ~0 errors in 20,000 bits for every K.
    (ii) Random codes: BER = Q(1/sqrt((K-1)/L + N0/2Eb)) rising with K/L; L=64 supports roughly K=8-10 for BER<=1e-2.
    (iii) Walsh + one 10x user: still ~0 errors while perfectly synchronous; BER explodes once chip-asynchrony appears.
    Processing gain = L (9.0, 12.0, 18.1 dB).
Q4  Orthogonality: inner product N for k=l, 0 for k!=l. Without CP, ICI/ISI errors; CP >= channel memory removes them.
    Sum throughput: channel-aware > interleaved >= contiguous. Fairness (Jain): interleaved/contiguous ~0.75-1.0 for equal mean SNR;
    channel-aware lowest when mean SNRs differ (strong users take most subcarriers). Benefit of channel-aware grows with number of users.
Q5  HPBW ~ 50.8/N degrees, side-lobe level ~ -13.3 dB for uniform weights. Minimum separation for SIR>=10 dB ~ inversely
    proportional to N: roughly 53, 27, 14, 7 degrees for N=2,4,8,16.
Q6  Cellular voice -> TDMA/FDMA hybrid; tactical -> CDMA; stadium downlink -> OFDMA; 5000 sensors -> FDMA/TDMA carrier with
    random access. Expect least robust: sensors (traffic pattern outside the five pure techniques).
Q7  Chosen limitation: dedicated channels idle under bursty traffic. Redesign (burst-level pool + statistical admission)
    should admit ~5-7x more sessions and lift useful utilisation from ~10% to >50%, at the cost of ~10x signalling per session
    and a small burst delay (<0.5 s).
