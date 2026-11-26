"""Original 150 BPM psytrance / hi-tech track for the Psychedelic Hi-Tech Ritual reel.

Everything is synthesised from scratch with numpy (no samples), so the audio is
copyright-free. Timeline (1 bar = 1.6 s at 150 BPM):
  0.0 - 1.6   intro: drone + noise riser + ticking hats
  1.6 - 3.2   build: accelerating snare roll, rising tone
  3.2         DROP: kick + rolling bass
  6.4 -       acid lead + claps join
 12.8         crash, full poster reveal
 19.2 - 20.8  outro: final hit, tail
"""
import sys
import wave
import numpy as np

SR = 44100
BPM = 150
BEAT = 60 / BPM          # 0.4 s
BAR = BEAT * 4           # 1.6 s
S16 = BEAT / 4           # 0.1 s
DROP = 2 * BAR           # 3.2 s
END_KICK = 12 * BAR      # 19.2 s
LENGTH = 13 * BAR        # 20.8 s
N = int(LENGTH * SR)
rng = np.random.default_rng(7)


def note(n):
    """MIDI note -> Hz."""
    return 440.0 * 2 ** ((n - 69) / 12)


def t_arr(dur):
    return np.arange(int(dur * SR)) / SR


def add(buf, x, at):
    i = int(round(at * SR))
    if i >= len(buf):
        return
    x = x[: len(buf) - i]
    buf[i:i + len(x)] += x


def onepole_lp(x, cutoff):
    """One-pole low-pass; cutoff may be a scalar or per-sample array."""
    c = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), x.shape)
    a = 1 - np.exp(-2 * np.pi * c / SR)
    y = np.empty_like(x)
    z = 0.0
    for i in range(len(x)):
        z += a[i] * (x[i] - z)
        y[i] = z
    return y


def svf_lp(x, cutoff, q):
    """Resonant state-variable low-pass (for the acid line)."""
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


def highpass_noise(n, smooth=6):
    w = rng.standard_normal(n)
    k = np.ones(smooth) / smooth
    return w - np.convolve(w, k, mode="same")


def saw(freq, t, detune=0.0):
    ph = (freq * (1 + detune)) * t
    return 2 * (ph - np.floor(ph + 0.5))


# ---------------------------------------------------------------- instruments
def kick():
    t = t_arr(0.32)
    f = 48 + 190 * np.exp(-t / 0.028)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.13)
    click = highpass_noise(len(t), 3) * np.exp(-t / 0.003) * 0.5
    return np.tanh(2.2 * (body + click)) * 0.95


def bass_note(midi):
    t = t_arr(S16 * 0.95)
    f = note(midi)
    raw = saw(f, t) * 0.7 + np.sin(2 * np.pi * f * t) * 0.6
    env_cut = 220 + 2600 * np.exp(-t / 0.018)
    y = onepole_lp(raw, env_cut)
    y = onepole_lp(y, env_cut * 1.2)
    amp = np.exp(-t / 0.055) * np.minimum(1, t / 0.002)
    return np.tanh(2.0 * y * amp) * 0.75


def hat(closed=True):
    d = 0.035 if closed else 0.12
    t = t_arr(d + 0.02)
    return highpass_noise(len(t), 3) * np.exp(-t / (d / 3)) * (0.18 if closed else 0.28)


def clap():
    t = t_arr(0.25)
    n = highpass_noise(len(t), 12)
    env = np.exp(-t / 0.06)
    for off in (0.0, 0.009, 0.018):            # triple-hit clap smear
        env = env + np.where(t >= off, np.exp(-(t - off) / 0.006), 0) * 0.6
    return n * env * 0.35


def snare(gain=1.0):
    t = t_arr(0.12)
    tone = np.sin(2 * np.pi * (190 + 60 * np.exp(-t / 0.01)) * t) * np.exp(-t / 0.03)
    noise = highpass_noise(len(t), 8) * np.exp(-t / 0.045)
    return (tone * 0.4 + noise * 0.6) * 0.45 * gain


def crash(dur=2.5):
    t = t_arr(dur)
    return highpass_noise(len(t), 2) * np.exp(-t / (dur / 4)) * 0.3


def boom(dur=2.0):
    t = t_arr(dur)
    f = 40 + 120 * np.exp(-t / 0.05)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.tanh(1.8 * np.sin(ph) * np.exp(-t / 0.6)) * 0.9


def zap(f0=3000, f1=120, dur=0.35):
    t = t_arr(dur)
    f = f1 + (f0 - f1) * np.exp(-t / (dur / 5))
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t / (dur / 2)) * 0.22


def reverb(x, seconds=1.6, mix=0.3):
    t = t_arr(seconds)
    ir = rng.standard_normal(len(t)) * np.exp(-t / (seconds / 6))
    ir /= np.sqrt(np.sum(ir ** 2))
    n = len(x) + len(ir)
    size = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    return x * (1 - mix) + y * mix * 1.5


# ---------------------------------------------------------------- arrangement
def build():
    drums = np.zeros(N)
    bass = np.zeros(N)
    music = np.zeros(N)       # pads / leads (gets reverb)
    fx = np.zeros(N)
    hatsL = np.zeros(N)
    hatsR = np.zeros(N)

    K = kick()
    CH, OH = hat(True), hat(False)
    CL = clap()

    # --- kick + rolling bass (F# psy groove)
    root = 42                                   # F#2
    bar_pattern = [0, 0, 0, 1]                  # last beat of each bar: variation
    bass_cache = {}
    t = DROP
    beat_i = 0
    while t < END_KICK - 1e-6:
        add(drums, K, t)
        var = bar_pattern[beat_i % 4]
        phrase = (beat_i // 16) % 2
        notes = [root, root, root] if not var else ([root + 1, root + 3, root + 12] if phrase else [root + 3, root + 7, root + 12])
        for k, m in enumerate(notes, start=1):
            if m not in bass_cache:
                bass_cache[m] = bass_note(m)
            add(bass, bass_cache[m], t + k * S16)
        # hats: closed 16ths (except on kick), open on the off-beat
        for k in range(4):
            v = 0.55 if k % 2 else 0.35
            add(hatsL if k in (1, 3) else hatsR, CH * v, t + k * S16)
        add(hatsL, OH * 0.5, t + 2 * S16)
        add(hatsR, OH * 0.5, t + 2 * S16 + 0.012)
        # clap on 2 & 4 once the lead comes in
        if t >= 4 * BAR - 1e-6 and beat_i % 2 == 1:
            add(drums, CL, t)
        t += BEAT
        beat_i += 1

    # --- intro: drone pad + riser + ticking hats
    tt = np.arange(N) / SR
    pad = sum(saw(note(m), tt, d) for m in (54, 61, 66) for d in (-0.004, 0.0, 0.005)) / 9
    pad_env = np.clip(tt / 1.2, 0, 1) * np.clip((DROP + 0.2 - tt) / 0.4, 0, 1)
    pad_env += 0.6 * np.clip((tt - END_KICK) / 0.1, 0, 1) * np.exp(-np.maximum(tt - END_KICK, 0) / 1.2)
    pad_cut = 300 + 2400 * np.clip(tt / DROP, 0, 1) ** 2
    music += onepole_lp(pad * pad_env, pad_cut) * 0.9

    r_len = int(DROP * SR)
    r_t = np.arange(r_len) / SR
    riser = highpass_noise(r_len, 2) * (r_t / DROP) ** 2.2 * 0.35
    rise_tone = np.sin(2 * np.pi * np.cumsum(200 + 1800 * (r_t / DROP) ** 3) / SR) * (r_t / DROP) ** 3 * 0.12
    add(fx, riser + rise_tone, 0.0)

    for i in range(8):                          # ticking hats through the intro bar
        add(hatsL if i % 2 else hatsR, OH * 0.35, i * BEAT / 2 + BEAT / 4)

    # --- snare roll (bar 2): 8ths -> 16ths -> 32nds
    hits = [BAR + i * BEAT / 2 for i in range(4)]
    hits += [BAR + 2 * BEAT + i * S16 for i in range(4)]
    hits += [BAR + 3 * BEAT + i * S16 / 2 for i in range(8)]
    for j, h in enumerate(hits):
        add(drums, snare(0.6 + 0.9 * j / len(hits)), h)

    # --- drop impact, crashes, zaps
    add(fx, boom(), DROP)
    add(fx, crash(), DROP)
    add(fx, crash(2.0) * 0.8, 8 * BAR)
    for b in range(3, 12):
        add(fx, zap(2600 + 400 * (b % 3), 150, 0.3), b * BAR + 3 * BEAT + 2 * S16)
    add(fx, boom(2.4), END_KICK)
    add(fx, crash(3.0), END_KICK)

    # --- acid hi-tech lead (bars 4-11)
    pat = [54, 66, 61, 54, 55, 66, 57, 54, 54, 62, 61, 66, 67, 54, 57, 61]
    acc = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1]
    lead_start, lead_end = 4 * BAR, END_KICK
    L = int((lead_end - lead_start) * SR)
    lt = np.arange(L) / SR
    freq = np.zeros(L)
    amp = np.zeros(L)
    accent = np.zeros(L)
    step = int(S16 * SR)
    for s in range(L // step):
        a, b = s * step, (s + 1) * step
        freq[a:b] = note(pat[s % 16])
        local = np.arange(b - a) / SR
        amp[a:b] = np.exp(-local / 0.07) * (1.0 if acc[s % 16] else 0.6)
        accent[a:b] = np.exp(-local / 0.03) * acc[s % 16]
    ph = np.cumsum(freq) / SR
    osc = 2 * (ph - np.floor(ph + 0.5))
    sweep = 0.5 - 0.5 * np.cos(2 * np.pi * lt / (4 * BAR))       # 4-bar filter sweep
    cut = 350 + 2800 * sweep + 2200 * accent
    lead = np.tanh(1.6 * svf_lp(osc * amp, cut, 6.0)) * 0.22
    lead *= np.clip(lt / 0.05, 0, 1) * np.clip((lead_end - lead_start - lt) / 0.05, 0, 1)
    add(music, lead, lead_start)

    # --- mix
    music = reverb(music, 1.8, 0.35)
    fx = reverb(fx, 2.2, 0.25)
    delay = int(3 * S16 * SR)                    # 3/16 ping-pong on the lead/music
    echoL = np.concatenate([np.zeros(delay), music[:-delay]]) * 0.3
    echoR = np.concatenate([np.zeros(2 * delay), music[:-2 * delay]]) * 0.2

    mono = drums * 1.0 + bass * 0.7 + fx * 1.1
    left = mono + music + echoL + hatsL + hatsR * 0.45
    right = mono + music + echoR + hatsR + hatsL * 0.45
    st = np.stack([left, right], axis=1)

    fade = np.clip((LENGTH - tt) / 0.35, 0, 1)[:, None]
    st /= np.percentile(np.abs(st), 99.95)      # gentle limiter instead of hard squash
    st = np.tanh(st * 0.9) * fade
    st /= np.max(np.abs(st)) / 0.95
    return st


def write_wav(path, st):
    pcm = (st * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "ritual_track.wav"
    write_wav(out, build())
    print("wrote", out, f"{LENGTH:.1f}s")
