"""Original melodic-house track for event promos (synthesised here, copyright-free).
Intro build with riser + snare roll -> drop (kick, claps, hats, rolling bass, supersaw pad, arp, lead)
-> breakdown with final impact. 120 BPM, A minor (Am - F - C - G)."""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
SR = 48000

def make_edm(total, drop, brk, seed=5, bpm=120):
    N = int(SR * total); rng = np.random.default_rng(seed)
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    lp = lambda x, f, o=2: sosfilt(butter(o, f, 'low', fs=SR, output='sos'), x)
    hp = lambda x, f, o=2: sosfilt(butter(o, f, 'high', fs=SR, output='sos'), x)
    bp = lambda x, a, b: sosfilt(butter(2, [a, b], 'band', fs=SR, output='sos'), x)
    bus = {k: np.zeros((N, 2)) for k in ('drums', 'bass', 'music', 'fx')}
    def add(b, sig, t0, pan=0.0, g=1.0):
        i = int(round(t0 * SR))
        if i >= N or i < 0: return
        j = min(N, i + len(sig)); s = sig[:j - i] * g
        bus[b][i:j, 0] += s * np.sqrt((1 - pan) / 2); bus[b][i:j, 1] += s * np.sqrt((1 + pan) / 2)
    B = 60 / bpm; BAR = 4 * B
    CH = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]; ROOT = [45, 41, 48, 43]
    chord_at = lambda t: int(max(0, t - drop) // BAR + (0 if t >= drop else 0)) % 4
    # --- instruments ---
    def kick(v=1.0):
        n = int(0.45 * SR); t = np.arange(n) / SR; f = 48 + 110 * np.exp(-t * 28)
        return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) + hp(rng.standard_normal(n), 4000) * np.exp(-t * 300) * 0.3) * 0.9 * v
    def clap(v=1.0):
        n = int(0.3 * SR); t = np.arange(n) / SR; x = np.zeros(n); nz = bp(rng.standard_normal(n), 900, 5000)
        for d in (0, 0.011, 0.022): x += nz * np.exp(-np.clip(t - d, 0, None) * 45) * (t >= d)
        return x * 0.35 * v
    def hat(v=1.0, open_=False):
        n = int((0.25 if open_ else 0.06) * SR); t = np.arange(n) / SR
        return hp(rng.standard_normal(n), 7000) * np.exp(-t * (14 if open_ else 70)) * 0.22 * v
    def pluck(m, v=1.0, dec=3.0, dur=0.6):
        n = int(dur * SR); t = np.arange(n) / SR; f = hz(m); x = np.zeros(n)
        for k in range(1, 14):
            if f * k > 16000: break
            x += np.sin(2 * np.pi * f * k * t) / k * np.exp(-t * (dec + 2.2 * k))
        return x * np.minimum(1, t / 0.002) * 0.12 * v
    def supersaw(notes, dur, v=1.0, cutoff=2500):
        n = int(dur * SR); t = np.arange(n) / SR; x = np.zeros(n)
        for m in notes:
            for d in (-0.12, -0.05, 0, 0.05, 0.12):
                x += 2 * ((hz(m + d) * t + rng.uniform()) % 1) - 1
        x = lp(x, cutoff, 2) / (5 * len(notes))
        return x * np.minimum(1, t / 0.25) * np.clip((dur - t) / 0.3, 0, 1) * 0.5 * v
    def bassnote(m, dur, v=1.0):
        n = int(dur * SR); t = np.arange(n) / SR; f = hz(m)
        x = 2 * ((f * t) % 1) - 1; x = lp(x, 520, 3) + 0.6 * np.sin(2 * np.pi * f * t)
        return x * np.minimum(1, t / 0.004) * np.exp(-t * 9) * 0.5 * v
    # --- intro: pad + filtered arp + riser + snare roll ---
    for b in range(int(np.ceil(drop / BAR))):
        t0 = b * BAR; c = CH[b % 4]
        add('music', supersaw([m + 12 for m in c], min(BAR, drop - t0) + 0.2, 0.7, 900 + 500 * b), t0, 0.0)
    i = 0; t = 0.0
    while t < drop:
        c = CH[int(t // BAR) % 4]; m = (c + [c[0] + 12])[[0, 1, 2, 3, 2, 1, 0, 2][i % 8]] + 12
        add('music', lp(pluck(m, 0.5 + 0.5 * t / drop), 1200 + 5000 * (t / drop) ** 2), t, 0.3 if i % 2 else -0.3)
        t += B / 2; i += 1
    rl = min(drop, 3.0); n = int(rl * SR); tt = np.arange(n) / SR
    noise = rng.standard_normal(n); rise = np.zeros(n)
    for k in range(12):                                   # band-pass sweep upward
        a, b2 = int(k * n / 12), int((k + 1) * n / 12); fc = 400 * 2 ** (k / 2.2)
        rise[a:b2] = bp(noise, fc, min(fc * 3, 20000))[a:b2]
    rise *= (tt / rl) ** 2 * 0.1
    sweep = np.sin(2 * np.pi * np.cumsum(200 * 2 ** (3 * tt / rl)) / SR) * (tt / rl) ** 3 * 0.05
    add('fx', rise + sweep, drop - rl)
    t = drop - 2 * B; s = 0
    while t < drop - 1e-6:                                # 8ths then 16ths into the drop
        step = B / 2 if t < drop - B else B / 4
        add('drums', clap(0.35 + 0.65 * (1 - (drop - t) / (2 * B))), t, 0.1); t += step
    # --- drop ---
    kicks = []
    t = drop; b = 0
    while t < brk - 1e-6:
        kicks.append(t); add('drums', kick(), t)
        if b % 2 == 1: add('drums', clap(), t, 0.05)
        add('drums', hat(0.9, open_=True), t + B / 2, 0.3)
        add('drums', hat(0.5), t + B / 4, -0.3); add('drums', hat(0.6), t + 3 * B / 4, -0.3)
        c = int((t - drop) // BAR) % 4
        for k in (1, 2, 3): add('bass', bassnote(ROOT[c], B / 4 * 0.9, 1.0 if k != 2 else 0.85), t + k * B / 4)
        if b % 4 == 0:
            add('music', supersaw([m + 12 for m in CH[c]], BAR + 0.1, 1.0, 3200), t, -0.15)
            add('music', supersaw([m + 24 for m in CH[c]], BAR + 0.1, 0.5, 4200), t, 0.15)
        t += B; b += 1
    i = 0; t = drop
    while t < brk - 1e-6:
        c = CH[int((t - drop) // BAR) % 4]; seq = c + [c[0] + 12, c[1] + 12]
        add('music', pluck(seq[[0, 2, 1, 3, 2, 4, 3, 2][i % 8]] + 12, 0.8), t, 0.35 if i % 2 else -0.35)
        t += B / 4; i += 1
    HOOK = [(0, 76, 1.0), (0.75, 74, .8), (1.5, 72, .8), (2, 69, 1.0), (3, 72, .8), (3.5, 74, .9),
            (4, 76, 1.0), (4.75, 79, .9), (5.5, 76, .8), (6, 74, 1.0), (7, 72, .8)]
    t = drop
    while t < brk - 8 * B + 0.01:
        for bb, m, v in HOOK: add('music', pluck(m + 12, v, 1.6, 1.2), t + bb * B, 0.0, 1.3)
        t += 8 * B
    # --- breakdown ---
    imp = int(2.5 * SR); ti = np.arange(imp) / SR
    add('fx', np.sin(2 * np.pi * np.cumsum(36 + 40 * np.exp(-ti * 6)) / SR) * np.exp(-ti * 1.6) * 0.8, brk)
    add('fx', hp(rng.standard_normal(imp), 3000) * np.exp(-ti * 2.2) * 0.12, brk)
    add('drums', kick(1.0), brk)
    end_len = total - brk
    for k in range(int(np.ceil(end_len / BAR))):
        c = CH[k % 4] if k < 3 else CH[0]
        add('music', supersaw([m + 12 for m in c], BAR + 0.4, 0.8, 1800), brk + k * BAR, 0.0)
    i = 0; t = brk
    while t < total - 1.5:
        c = CH[int((t - brk) // BAR) % 4]; seq = c + [c[0] + 12]
        add('music', pluck(seq[[0, 1, 2, 3, 2, 1][i % 6]] + 24, 0.45, 2.0, 1.0), t, 0.4 if i % 2 else -0.4)
        t += B / 2; i += 1
    for bb, m, v in HOOK[:6]: add('music', pluck(m + 12, v * 0.8, 1.2, 1.5), brk + BAR + bb * B, 0.0, 1.2)
    add('music', supersaw([57, 64, 69, 72, 76], 3.5, 0.9, 2200), total - 3.5)
    # --- sidechain, delay, reverb, master ---
    g = np.ones(N)
    for k in kicks:
        i0 = int(k * SR); n = min(N - i0, int(0.45 * SR)); tt = np.arange(n) / SR
        g[i0:i0 + n] = np.minimum(g[i0:i0 + n], 1 - 0.65 * np.exp(-tt / 0.11))
    mus = bus['music'] * g[:, None]; bas = bus['bass'] * g[:, None]
    d = int(0.75 * B * SR); dl = np.zeros_like(mus); dl[d:] += mus[:-d] * 0.3; dl[2 * d:] += mus[:-2 * d] * 0.15
    mus = mus + dl[:, ::-1]
    n = int(2.0 * SR); tt = np.arange(n) / SR
    irs = [lp(rng.standard_normal(n), 7000) * np.exp(-tt * 3.0) for _ in range(2)]
    wet = np.stack([fftconvolve(mus[:, 0], irs[0])[:N], fftconvolve(mus[:, 1], irs[1])[:N]], 1)
    wet /= np.abs(wet).max() + 1e-9
    mix = bus['drums'] * 0.75 + hp(bas.T, 45).T * 0.6 + mus / (np.abs(mus).max() + 1e-9) * 0.7 + wet * 0.2 + bus['fx']
    mix = mix + 0.25 * hp(mix.T, 3000).T             # presence for phone speakers
    mix = hp(mix.T, 30).T
    mix = np.tanh(mix * 1.6) / np.tanh(1.6)
    mix /= np.abs(mix).max() / 0.92
    t = np.arange(N) / SR; mix *= (np.minimum(1, t / 0.02) * np.clip((total - t) / 1.2, 0, 1))[:, None]
    return mix
