"""Original 140 BPM hypnotic psytech / groove track for the LOOP HOOLE reel.

Synthesised from scratch with numpy (no samples) -> copyright-free.
1 bar = 1.714 s.  Timeline:
  bar 0      tanpura drone (Karplus-Strong) + filtered "loop" arp
  bar 1      damaru-style tom roll, riser
  bar 2      DROP: kick + grooving rolling bass
  bar 4+     claps, rimshots, arp opens up
  bar 9      crash, full poster
  bar 11     outro: tanpura returns, final hit
"""
import sys
import wave
import numpy as np

SR = 44100
BPM = 140
BEAT = 60 / BPM
BAR = BEAT * 4
S16 = BEAT / 4
DROP = 2 * BAR
END_KICK = 11 * BAR
LENGTH = 12 * BAR
N = int(LENGTH * SR)
rng = np.random.default_rng(11)
ROOT = 37                     # C#2


def note(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def t_arr(d):
    return np.arange(int(d * SR)) / SR


def add(buf, x, at):
    i = int(round(at * SR))
    if 0 <= i < len(buf):
        x = x[: len(buf) - i]
        buf[i:i + len(x)] += x


def onepole_lp(x, cutoff):
    c = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), x.shape)
    a = 1 - np.exp(-2 * np.pi * c / SR)
    y = np.empty_like(x)
    z = 0.0
    for i in range(len(x)):
        z += a[i] * (x[i] - z)
        y[i] = z
    return y


def svf_lp(x, cutoff, q):
    c = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), x.shape)
    f = 2 * np.sin(np.pi * np.minimum(c, SR / 6) / SR)
    damp = 1.0 / q
    low = band = 0.0
    y = np.empty_like(x)
    for i in range(len(x)):
        high = x[i] - low - damp * band
        band += f[i] * high
        low += f[i] * band
        y[i] = low
    return y


def hp_noise(n, smooth=6):
    w = rng.standard_normal(n)
    return w - np.convolve(w, np.ones(smooth) / smooth, mode="same")


def saw(freq, t):
    ph = freq * t
    return 2 * (ph - np.floor(ph + 0.5))


# ---------------------------------------------------------------- instruments
def kick():
    t = t_arr(0.34)
    f = 50 + 150 * np.exp(-t / 0.032)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)
    click = hp_noise(len(t), 3) * np.exp(-t / 0.002) * 0.35
    return np.tanh(1.8 * (body + click)) * 0.95


def bass_note(midi, length=0.95, bright=1.0):
    t = t_arr(S16 * length)
    f = note(midi)
    raw = saw(f, t) * 0.6 + np.sin(2 * np.pi * f * t) * 0.7
    cut = 180 + 1700 * bright * np.exp(-t / 0.022)
    y = onepole_lp(onepole_lp(raw, cut), cut * 1.3)
    amp = np.exp(-t / 0.06) * np.minimum(1, t / 0.002)
    return np.tanh(2.0 * y * amp) * 0.8


def hat(closed=True):
    d = 0.03 if closed else 0.11
    t = t_arr(d + 0.02)
    return hp_noise(len(t), 3) * np.exp(-t / (d / 3)) * (0.17 if closed else 0.26)


def rim():
    t = t_arr(0.06)
    tone = np.sin(2 * np.pi * 1650 * t) * 0.5 + np.sin(2 * np.pi * 820 * t) * 0.4
    return (tone + hp_noise(len(t), 4) * 0.4) * np.exp(-t / 0.012) * 0.22


def clap():
    t = t_arr(0.25)
    env = np.exp(-t / 0.06)
    for off in (0.0, 0.01, 0.02):
        env = env + np.where(t >= off, np.exp(-(t - off) / 0.006), 0) * 0.6
    return hp_noise(len(t), 12) * env * 0.3


def tom(pitch=1.0, gain=1.0):
    """Damaru-ish pitched skin hit."""
    t = t_arr(0.18)
    f = (170 + 120 * np.exp(-t / 0.02)) * pitch
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.07)
    slap = hp_noise(len(t), 5) * np.exp(-t / 0.008) * 0.3
    return (body + slap) * 0.4 * gain


def tanpura(midi, dur=2.4):
    """Karplus-Strong pluck with a little 'jawari' buzz."""
    f = note(midi)
    p = int(SR / f)
    n = int(dur * SR)
    out = np.zeros(n)
    buf = rng.uniform(-1, 1, p)
    for i in range(n):
        v = buf[i % p]
        out[i] = v
        buf[i % p] = 0.4985 * (v + buf[(i + 1) % p])
    buzz = np.tanh(out * 3) * 0.3
    return onepole_lp(out + buzz, 3500) * 0.35


def pluck(midi, dur=0.35, bright=1.0):
    """FM bell-pluck for the hypnotic loop."""
    t = t_arr(dur)
    f = note(midi)
    mod = np.sin(2 * np.pi * f * 2 * t) * (2.5 * bright) * np.exp(-t / 0.05)
    return np.sin(2 * np.pi * f * t + mod) * np.exp(-t / 0.09) * 0.2


def crash(dur=2.5):
    t = t_arr(dur)
    return hp_noise(len(t), 2) * np.exp(-t / (dur / 4)) * 0.28


def boom(dur=2.2):
    t = t_arr(dur)
    f = 38 + 110 * np.exp(-t / 0.06)
    return np.tanh(1.8 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.65)) * 0.9


def reverb(x, seconds=1.8, mix=0.3):
    t = t_arr(seconds)
    ir = rng.standard_normal(len(t)) * np.exp(-t / (seconds / 6))
    ir /= np.sqrt(np.sum(ir ** 2))
    size = 1 << (len(x) + len(ir) - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    return x * (1 - mix) + y * mix * 1.5


# ---------------------------------------------------------------- arrangement
def build():
    drums, bass, music, fx = (np.zeros(N) for _ in range(4))
    hL, hR = np.zeros(N), np.zeros(N)
    tt = np.arange(N) / SR
    K, CH, OH, RM, CL = kick(), hat(True), hat(False), rim(), clap()
    swing = 0.014

    # --- kick, groove bass, hats, perc
    # per-bar 16th grid: 1 = bass note, 2 = bass note an octave up, 0 = rest (kick on 0,4,8,12)
    groove = [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 2, 0, 1, 1, 0]
    groove_b = [0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 2, 1, 0, 3, 1, 1]
    cache = {}
    nbar = 0
    t = DROP
    while t < END_KICK - 1e-6:
        g = groove_b if nbar % 4 == 3 else groove
        for s in range(16):
            ts = t + s * S16 + (swing if s % 2 else 0)
            if s % 4 == 0:
                add(drums, K, t + s * S16)
            elif g[s]:
                m = ROOT + {1: 0, 2: 12, 3: 3}[g[s]]
                key = (m, g[s])
                if key not in cache:
                    cache[key] = bass_note(m, bright=1.4 if g[s] > 1 else 1.0)
                add(bass, cache[key], ts)
            add(hL if s % 4 in (1, 3) else hR, CH * (0.6 if s % 2 else 0.3), ts)
            if s % 4 == 2:
                add(hL, OH * 0.5, ts)
                add(hR, OH * 0.45, ts + 0.01)
            if t >= 4 * BAR - 1e-6:
                if s in (3, 6, 11, 14):
                    add(hR if s in (3, 11) else hL, RM, ts)
                if s in (4, 12):
                    add(drums, CL, t + s * S16)
        t += BAR
        nbar += 1

    # --- tanpura: Pa Sa' Sa' Sa (G#3 C#4 C#4 C#3), intro + outro
    cycle = [56, 61, 61, 49]
    tan_cache = {m: tanpura(m) for m in set(cycle)}
    for i in range(8):
        add(music, tan_cache[cycle[i % 4]] * 1.2, i * BEAT)
    for i in range(int(BAR / BEAT) + 2):
        add(music, tan_cache[cycle[i % 4]] * 1.0, END_KICK + i * BEAT)
    for i in range(20, 44, 2):                   # quiet tanpura bed under the groove
        add(music, tan_cache[cycle[(i // 2) % 4]] * 0.35, DROP + i * BEAT)

    # --- the "loop": 3-note pluck cycling on 16ths (3 against 4 = hypnotic)
    loop = [61, 68, 76, 64, 68, 73]
    pl_cache = {}
    s = 0
    t = 0.0
    while t < END_KICK - 1e-6:
        if t >= BAR * 0.5:
            m = loop[s % 3] if t < 6 * BAR else loop[3 + s % 3]
            bright = 0.3 if t < DROP else (0.7 + 0.6 * (0.5 - 0.5 * np.cos(2 * np.pi * t / (4 * BAR))))
            key = (m, round(bright, 1))
            if key not in pl_cache:
                pl_cache[key] = pluck(m, bright=key[1])
            gain = 0.6 if t < DROP else 1.0
            add(music, pl_cache[key] * gain, t + (swing if s % 2 else 0))
        s += 1
        t = s * S16

    # --- psytech acid stab layer (bars 5-10), filtered resonant saw on the off-16ths
    a0, a1 = 5 * BAR, END_KICK
    L = int((a1 - a0) * SR)
    lt = np.arange(L) / SR
    freq = np.full(L, note(ROOT + 24))
    amp = np.zeros(L)
    step = int(S16 * SR)
    acid_pat = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    acid_nt = [0, 0, 7, 0, 0, 12, 0, 3, 0, 0, 7, 0, 15, 0, 0, 12]
    for k in range(L // step):
        if acid_pat[k % 16]:
            a, b = k * step, (k + 1) * step
            freq[a:b] = note(ROOT + 24 + acid_nt[k % 16])
            amp[a:b] = np.exp(-np.arange(b - a) / SR / 0.06)
    osc = saw(1, np.cumsum(freq) / SR)
    sweep = 0.5 - 0.5 * np.cos(2 * np.pi * lt / (2 * BAR))
    acid = np.tanh(1.5 * svf_lp(osc * amp, 400 + 2200 * sweep, 7.0)) * 0.16
    add(music, acid, a0)

    # --- intro drone + riser, damaru roll
    drone = (saw(note(ROOT + 12), tt) + saw(note(ROOT + 19) * 1.003, tt)) * 0.5
    d_env = np.clip(tt / 1.0, 0, 1) * np.clip((DROP + 0.1 - tt) / 0.3, 0, 1)
    music += onepole_lp(drone * d_env, 250 + 1500 * np.clip(tt / DROP, 0, 1) ** 2) * 0.35
    r_len = int(DROP * SR)
    r_t = np.arange(r_len) / SR
    add(fx, hp_noise(r_len, 2) * (r_t / DROP) ** 2.5 * 0.3, 0.0)
    hits = [BAR + i * BEAT / 2 for i in range(4)] + [BAR + 2 * BEAT + i * S16 for i in range(4)]
    hits += [BAR + 3 * BEAT + i * S16 / 2 for i in range(8)]
    for j, h in enumerate(hits):
        add(drums, tom(1.0 + 0.5 * j / len(hits), 0.6 + 0.8 * j / len(hits)), h)

    add(fx, boom(), DROP)
    add(fx, crash(), DROP)
    add(fx, crash(2.0) * 0.8, 9 * BAR)
    add(fx, boom(2.6), END_KICK)
    add(fx, crash(3.0) * 0.9, END_KICK)

    # --- mix
    music = reverb(music, 2.0, 0.35)
    fx = reverb(fx, 2.2, 0.25)
    dl = int(3 * S16 * SR)
    eL = np.concatenate([np.zeros(dl), music[:-dl]]) * 0.3
    eR = np.concatenate([np.zeros(2 * dl), music[:-2 * dl]]) * 0.22
    mono = drums + bass * 0.7 + fx * 1.1
    st = np.stack([mono + music + eL + hL + hR * 0.45,
                   mono + music + eR + hR + hL * 0.45], axis=1)
    st /= np.percentile(np.abs(st), 99.95)
    st = np.tanh(st * 0.9) * np.clip((LENGTH - tt) / 0.35, 0, 1)[:, None]
    return st / (np.max(np.abs(st)) / 0.95)


def write_wav(path, st):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "loophoole_track.wav"
    write_wav(out, build())
    print("wrote", out, f"{LENGTH:.2f}s")
