"""Original Himalayan-cafe soundtrack: tanpura drone, santoor arpeggios, bansuri melody in Raag Bhupali
(Sa Re Ga Pa Dha), soft tabla in keherwa taal (8 beats), warm pad. Copyright-free (synthesised here)."""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve, lfilter
SR = 48000

def make_himalayan(total, flute_start, tabla_start, break_start, end_start, seed=7, bpm=84):
    N = int(SR * total); rng = np.random.default_rng(seed)
    L = np.zeros(N); R = np.zeros(N)
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    lp = lambda x, f, o=2: sosfilt(butter(o, f, 'low', fs=SR, output='sos'), x)
    hp = lambda x, f, o=2: sosfilt(butter(o, f, 'high', fs=SR, output='sos'), x)
    bp = lambda x, a, b: sosfilt(butter(2, [a, b], 'band', fs=SR, output='sos'), x)
    def add(sig, t0, pan=0.0, g=1.0):
        i = int(t0 * SR)
        if i >= N or i < 0: return
        j = min(N, i + len(sig)); s = sig[:j - i] * g
        L[i:j] += s * np.sqrt((1 - pan) / 2); R[i:j] += s * np.sqrt((1 + pan) / 2)
    B = 60 / bpm; CYC = 8 * B
    # ---- tanpura (Pa Sa Sa Sa) ----
    def tanpura(f, dur=4.5):
        t = np.arange(int(dur * SR)) / SR; x = np.zeros_like(t); sw = t / dur
        for h in range(1, 20):
            env = np.exp(-t * (0.55 + 0.05 * h)) * (0.3 + 0.7 * np.exp(-((h - (3 + 11 * sw)) ** 2) / 8))
            x += env * np.sin(2 * np.pi * f * h * t * (1 + 0.0004 * h) + rng.uniform(0, 6)) / h ** 0.7
        return hp(x, 90) * np.minimum(1, t / 0.01) * 0.05
    t = 0.0; k = 0
    while t < total:
        add(tanpura([hz(45), hz(50), hz(50), hz(38)][k % 4]), t, pan=[-.3, .2, .3, -.1][k % 4]); t += B * 1.5; k += 1
    # ---- chords: D, Bm, Em7, Asus ----
    CH = [[62, 66, 69, 74], [59, 62, 66, 71], [64, 67, 71, 74], [57, 62, 64, 69]]
    CH[2] = [64, 71, 74, 76]                       # stay inside Bhupali (no Ma)
    BASS = [38, 35, 40, 33]
    def pad(notes, dur, amp):
        n = int(dur * SR); t = np.arange(n) / SR; x = np.zeros(n)
        for m in notes:
            for d in (-0.07, 0.07):
                x += 2 * ((hz(m + d) * t + rng.uniform()) % 1) - 1
        x = lp(x, 900, 2)
        return x * np.minimum(1, t / 1.0) * np.clip((dur - t) / 1.2, 0, 1) * amp / len(notes)
    def bassn(m, dur):
        n = int(dur * SR); t = np.arange(n) / SR
        return lp(np.sin(2 * np.pi * hz(m) * t) + 0.2 * np.sin(4 * np.pi * hz(m) * t), 400) * np.minimum(1, t / 0.02) * np.clip((dur - t) / 0.4, 0, 1) * 0.06
    c = 0; t = 0.0
    while t < end_start:
        ch = CH[c % 4]; d = min(CYC, end_start - t) + 1.2
        add(hp(pad([m - 12 for m in ch], d, 0.035), 140), t, -0.2); add(hp(pad(ch, d, 0.025), 140), t, 0.2)
        if t >= tabla_start - 0.01: add(bassn(BASS[c % 4], CYC * 0.95), t)
        c += 1; t += CYC
    # ---- santoor: bright hammered strings, arpeggios + tremolo ----
    def santoor(m, v=1.0, dur=1.6):
        f = hz(m); P = int(SR / f); n = int(dur * SR); exc = np.zeros(n)
        exc[:P] = hp(rng.standard_normal(P), 500)
        a = np.zeros(P + 2); a[0] = 1; a[P] = -0.499; a[P + 1] = -0.499
        y = lfilter([1], a, exc)
        y2 = np.zeros(n); dt = int(0.0035 * SR); y2[dt:] = y[:-dt]       # double string, slight detune feel
        return (y + 0.7 * y2) * 0.08 * v
    ARP = [0, 1, 2, 3, 2, 1, 2, 3]
    t = 0.0; i = 0
    while t < end_start:
        cyc = int(t / CYC); ch = CH[cyc % 4]; m = ch[ARP[i % 8]] + 12
        v = (0.9 if i % 2 == 0 else 0.6) * (0.7 if t < flute_start else 1.0) * (0.8 if break_start <= t else 1.0)
        add(santoor(m, v), t, pan=0.35 if i % 2 else 0.25)
        if i % 8 == 0: add(santoor(m, 0.4), t + B / 4, pan=0.3)          # tremolo flick
        t += B / 2; i += 1
    # ---- tabla (keherwa: dha ge na ti | na ka dhi na) ----
    def dayan(v=1.0, muted=False):
        n = int(0.6 * SR); t = np.arange(n) / SR; f = hz(62); x = np.zeros(n)
        for h, (a, dcy) in enumerate([(1, 7), (.55, 9), (.35, 13), (.2, 17), (.1, 22)], start=1):
            x += a * np.sin(2 * np.pi * f * h * t) * np.exp(-t * dcy * (4 if muted else 1))
        click = hp(rng.standard_normal(n), 3000) * np.exp(-t * 400) * 0.4
        return (x + click) * 0.13 * v
    def bayan(v=1.0, bend=True):
        n = int(0.7 * SR); t = np.arange(n) / SR
        f = 78 + (30 * (1 - np.exp(-t * 6)) if bend else 0)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.5) + lp(rng.standard_normal(n), 300) * np.exp(-t * 40) * 0.5
        return hp(x, 60) * 0.12 * v
    def ka(v=1.0):
        n = int(0.08 * SR); t = np.arange(n) / SR
        return bp(rng.standard_normal(n), 250, 1500) * np.exp(-t * 60) * 0.12 * v
    PAT = [('dha', 1.0), ('ge', .7), ('na', .8), ('ti', .6), ('na', .8), ('ka', .7), ('dhi', .9), ('na', .8)]
    t = tabla_start; b = 0
    while t < end_start - 0.05:
        s, v = PAT[b % 8]
        v *= 0.55 if break_start <= t else 1.0
        if s in ('dha', 'dhi', 'ge'): add(bayan(v, bend=(s != 'ge')), t, -0.1)
        if s in ('dha', 'na'): add(dayan(v), t, 0.1)
        if s == 'dhi': add(dayan(v * 0.8, muted=True), t, 0.1)
        if s == 'ti': add(dayan(v, muted=True), t, 0.1)
        if s == 'ka': add(ka(v), t, -0.05)
        if b % 8 in (3, 7): add(dayan(0.35 * v, muted=True), t + B / 2, 0.15)   # ghost "te"
        if b % 16 == 15: add(dayan(0.5 * v), t + B / 4, 0.1); add(dayan(0.6 * v, True), t + 3 * B / 4, 0.1)
        t += B; b += 1
    # ---- bansuri: Bhupali phrases with meend (glides) ----
    def flute(notes, v=1.0):
        tot = max(s + d for s, d, m in notes) * B + 1.2; n = int(tot * SR); t = np.arange(n) / SR
        freq = np.full(n, hz(notes[0][2])); amp = np.zeros(n)
        for s, d, m in notes:
            i0 = int(s * B * SR); i1 = int((s + d) * B * SR); g = int(0.09 * SR)
            prev = freq[i0 - 1] if i0 > 0 else hz(m)
            freq[i0:i0 + g] = np.geomspace(prev, hz(m), g); freq[i0 + g:] = hz(m)
            tt = np.arange(i1 - i0) / SR; ld = d * B
            amp[i0:i1] = np.maximum(amp[i0:i1], np.minimum(1, tt / 0.08) * np.clip((ld - tt) / 0.22, 0, 1) * (1 + 0.1 * np.sin(2 * np.pi * 0.6 * tt)))
        vib = 1 + 0.007 * np.sin(2 * np.pi * 5.0 * t) * np.clip((t % 1.6) / 0.5, 0, 1)
        ph = 2 * np.pi * np.cumsum(freq * vib) / SR
        tone = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.05 * np.sin(3 * ph)
        amp = lp(amp, 25, 1)
        return (tone + bp(rng.standard_normal(n), 1200, 6000) * 0.14) * amp * 0.14 * v
    Sa, Re, Ga, Pa, Dha, S2, R2, G2, P2 = 62, 64, 66, 69, 71, 74, 76, 78, 81
    A = [(0, 1, Pa), (1, .5, Dha), (1.5, .5, Pa), (2, 1, Ga), (3, 1, Re), (4, 2.5, Sa), (6.5, .5, Re), (7, 1, Ga)]
    Bp = [(0, 1.5, Pa), (1.5, .5, Dha), (2, 2, S2), (4, 1, R2), (5, .5, S2), (5.5, .5, Dha), (6, 2, Pa)]
    C = [(0, 1, S2), (1, 1, Dha), (2, 1, Pa), (3, 1, Ga), (4, 1.5, Re), (5.5, .5, Ga), (6, 2, Sa)]
    D = [(0, 1, R2), (1, 1, G2), (2, 2, P2), (4, 1, G2), (5, 1, R2), (6, 2, S2)]
    E = [(0, 1, Ga), (1, 1, Pa), (2, 1, Dha), (3, 1, S2), (4, 4, S2)]
    seq = [A, Bp, C, D, Bp, C]
    t = flute_start; i = 0
    while t < end_start - 1.0 and i < len(seq):
        add(flute(seq[i]), t, pan=0.05); t += CYC; i += 1
    add(flute(E, 1.05), end_start - 0.2, pan=0.05)
    # ---- ending: santoor run + bells ----
    for j, m in enumerate([86, 83, 81, 78, 76, 74, 71, 69]): add(santoor(m, 0.7, 2.5), end_start + j * 0.09, pan=-0.3 + 0.08 * j)
    def bell(m, v):
        n = int(4 * SR); t = np.arange(n) / SR; f = hz(m)
        return sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * r * 0.9) for r, a in ((1, 1), (2.76, .35), (5.4, .15))) * 0.06 * v
    add(bell(74, 1.0), end_start + 0.8, 0); add(bell(86, 0.6), end_start + 1.2, 0.3)
    add(pad([50, 57, 62, 66], total - end_start + 0.5, 0.06), end_start)
    add(bassn(38, total - end_start), end_start)
    # ---- reverb + master ----
    n = int(2.4 * SR); tt = np.arange(n) / SR
    irs = [lp(rng.standard_normal(n), 6000) * np.exp(-tt * 2.6) for _ in range(2)]
    wL = fftconvolve(L, irs[0])[:N]; wR = fftconvolve(R, irs[1])[:N]; wL /= np.abs(wL).max(); wR /= np.abs(wR).max()
    pk = max(np.abs(L).max(), np.abs(R).max())
    st = np.stack([L / pk + 0.3 * wL, R / pk + 0.3 * wR], 1); st = hp(st.T, 45).T
    st = st + 0.35 * hp(st.T, 2500).T            # air / presence for phone speakers
    st = np.tanh(st * 1.3) / np.tanh(1.3); st /= np.abs(st).max() / 0.9
    t = np.arange(N) / SR; st *= (np.minimum(1, t / 0.4) * np.clip((total - t) / 2.0, 0, 1))[:, None]
    return st
