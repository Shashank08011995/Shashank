"""Render a 1080x1920 Instagram Reel for LOOP HOOLE, beat-synced to make_music.py.

Usage: python3 make_reel.py POSTER.jpg TRACK.wav OUT.mp4 FONT_DIR [preview times...]
"""
import math
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

POSTER, TRACK, OUT, FONT_DIR = sys.argv[1:5]
W, H, FPS = 1080, 1920, 30
BPM = 140
BEAT = 60 / BPM
BAR = 4 * BEAT
DROP = 2 * BAR
END_KICK = 11 * BAR
LENGTH = 12 * BAR
CYAN = (70, 235, 255)
PINK = (255, 80, 165)
WHITE = (245, 245, 250)
CY = 900                                    # visual centre (clear of IG caption area)
rng = np.random.default_rng(5)

poster = Image.open(POSTER).convert("RGB")
F_BLACK = f"{FONT_DIR}/montserrat-black.ttf"
F_XB = f"{FONT_DIR}/montserrat-extrabold.ttf"
F_B = f"{FONT_DIR}/montserrat-bold.ttf"


# ---------------------------------------------------------------- helpers
def feathered(img, feather):
    w, h = img.size
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rectangle((feather, feather, w - feather, h - feather), fill=255)
    out = img.convert("RGBA")
    out.putalpha(m.filter(ImageFilter.GaussianBlur(feather / 2)))
    return out


def add_glow(img, radius=14, strength=2):
    g = img.filter(ImageFilter.GaussianBlur(radius))
    base = Image.new("RGBA", img.size, (0, 0, 0, 0))
    for _ in range(strength):
        base.alpha_composite(g)
    base.alpha_composite(img)
    return base


def text(s, font, size, fill, spacing=0.06, glow=True, gradient=None):
    """Letter-spaced text layer. fill = colour, or gradient=(c1, c2) left->right."""
    f = ImageFont.truetype(font, size)
    widths = [f.getlength(c) for c in s]
    track = size * spacing
    tw = int(sum(widths) + track * (len(s) - 1)) + 80
    asc, desc = f.getmetrics()
    th = asc + desc + 60
    mask = Image.new("L", (tw, th), 0)
    d = ImageDraw.Draw(mask)
    x = 40
    for c, w in zip(s, widths):
        d.text((x, 30), c, font=f, fill=255)
        x += w + track
    if gradient:
        g = np.linspace(0, 1, tw)[None, :, None]
        c1, c2 = np.array(gradient[0]), np.array(gradient[1])
        mid = np.array(WHITE)
        col = np.where(g < 0.5, c1 + (mid - c1) * (g * 2), mid + (c2 - mid) * ((g - 0.5) * 2))
        rgb = np.broadcast_to(col, (th, tw, 3)).astype(np.uint8)
        img = Image.fromarray(rgb).convert("RGBA")
    else:
        img = Image.new("RGBA", (tw, th), fill + (255,))
    img.putalpha(mask)
    return add_glow(img) if glow else img


def pill(label, size=40, color=PINK):
    t = text(label, F_B, size, WHITE, 0.12, glow=False)
    pad = 50
    img = Image.new("RGBA", (t.width + pad, t.height + 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((10, 10, img.width - 10, img.height - 10), radius=(img.height - 20) // 2,
                        outline=color + (255,), width=5, fill=(20, 8, 40, 170))
    img.alpha_composite(t, (pad // 2, 5))
    return add_glow(img, 10, 1)


def paste(frame, img, cx, cy, scale=1.0, alpha=1.0, angle=0.0):
    if img is None or alpha <= 0:
        return
    if scale != 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
    if angle:
        img = img.rotate(angle, resample=Image.BILINEAR)
    x0, y0 = int(cx - img.width / 2), int(cy - img.height / 2)
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + img.width, W), min(y0 + img.height, H)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    a = np.asarray(img).astype(np.float32)[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    al = a[..., 3:4] / 255.0 * min(alpha, 1.0)
    region = frame[fy0:fy1, fx0:fx1]
    region *= 1 - al
    region += a[..., :3] * al


def typed(img, p):
    p = max(0.0, min(1.0, p))
    if p <= 0:
        return None
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img.crop((0, 0, int(img.width * p), img.height)), (0, 0))
    return out


def geometry(t, rot, scale, alpha, draw_p=1.0):
    """Sacred-geometry layer (hexagram, rings, rotating polygons) in cyan/pink."""
    S = 1100
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    a = int(255 * alpha)

    def poly(n, r, ang, col, w):
        pts = [(c + r * math.cos(ang + 2 * math.pi * k / n), c + r * math.sin(ang + 2 * math.pi * k / n)) for k in range(n)]
        m = max(1, int(round(n * draw_p)))
        for k in range(m):
            d.line([pts[k], pts[(k + 1) % n]], fill=col + (a,), width=w)

    r0 = 420 * scale
    poly(3, r0, -math.pi / 2 + rot, PINK, 4)
    poly(3, r0, math.pi / 2 + rot, CYAN, 4)
    poly(6, r0 * 0.62, -rot * 1.7, CYAN, 3)
    poly(8, r0 * 1.12, rot * 0.6, PINK, 2)
    poly(12, r0 * 0.38, rot * 2.4, PINK, 2)
    for rr, col in ((r0, CYAN), (r0 * 0.5, PINK), (r0 * 1.25, CYAN)):
        sweep = 360 * draw_p
        d.arc((c - rr, c - rr, c + rr, c + rr), -90 + math.degrees(rot), -90 + math.degrees(rot) + sweep, fill=col + (a,), width=2)
    return add_glow(img, 8, 1)


# ---------------------------------------------------------------- audio analysis (for EQ bars)
with wave.open(TRACK) as w:
    sr = w.getframerate()
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2).mean(axis=1) / 32768
NBARS = 30
edges = np.geomspace(40, 12000, NBARS + 1)
freqs = np.fft.rfftfreq(2048, 1 / sr)
band_idx = [np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0] for i in range(NBARS)]
nframes = int(LENGTH * FPS)
EQ = np.zeros((nframes, NBARS))
win = np.hanning(2048)
for i in range(nframes):
    s = int(i / FPS * sr)
    seg = pcm[s:s + 2048]
    if len(seg) < 2048:
        seg = np.pad(seg, (0, 2048 - len(seg)))
    mag = np.abs(np.fft.rfft(seg * win))
    EQ[i] = [20 * np.log10(mag[b].mean() + 1e-6) if len(b) else -120 for b in band_idx]
EQ = np.clip((EQ + 10) / 50, 0, 1)
for i in range(1, nframes):                                 # fast attack, slow release
    EQ[i] = np.maximum(EQ[i], EQ[i - 1] * 0.82)


def draw_eq(frame, i, alpha=1.0, height=150):
    bw = W / NBARS
    for b in range(NBARS):
        h = int(14 + height * EQ[i, b])
        col = np.array(PINK if b % 2 == 0 else CYAN, np.float32)
        x0, x1 = int(b * bw + 5), int((b + 1) * bw - 5)
        region = frame[H - h:H, x0:x1]
        region *= 1 - alpha
        region += col * alpha


# ---------------------------------------------------------------- assets
print("preparing assets...")
BG = 1.35
bg_src = poster.resize((int(W * BG), int(H * BG)), Image.LANCZOS).filter(ImageFilter.GaussianBlur(30))
bg_big = np.asarray(bg_src).astype(np.float32) * np.array([0.38, 0.33, 0.48], np.float32)   # violet tint
reveal = np.asarray(poster.resize((W, H), Image.LANCZOS)).astype(np.float32)
reveal_img = Image.fromarray(reveal.astype(np.uint8)).convert("RGBA")

shiva = poster.crop((150, 375, 575, 875))
shiva = shiva.resize((int(shiva.width * 2.0), int(shiva.height * 2.0)), Image.LANCZOS)
sw, sh = shiva.size
_yy, _xx = np.mgrid[0:sh, 0:sw]
_r = np.hypot((_xx - sw / 2) / (sw * 0.5), (_yy - sh / 2) / (sh * 0.52))
_m = (np.clip((1 - _r) / 0.22, 0, 1) ** 1.5 * 255).astype(np.uint8)       # soft oval vignette
SHIVA = shiva.convert("RGBA")
SHIVA.putalpha(Image.fromarray(_m))
SHIVA = add_glow(SHIVA, 16, 1)

T = {
    "invite": text("INVITE ONLY EVENT", F_XB, 46, CYAN, 0.35),
    "loop": text("LOOP", F_BLACK, 250, None, 0.02, gradient=(CYAN, PINK)),
    "hoole": text("HOOLE", F_BLACK, 250, None, 0.02, gradient=(CYAN, PINK)),
    "28nov": text("28 NOV", F_BLACK, 210, PINK, 0.03),
    "neer": text("NEER, RISHIKESH", F_XB, 78, WHITE, 0.08),
    "lineup": text("LINE UP", F_XB, 44, CYAN, 0.5),
    "solzi": text("SOLZI", F_BLACK, 170, WHITE, 0.04),
    "yaksha": text("YAKSHA", F_BLACK, 170, WHITE, 0.04),
    "gtek": text("GTEK", F_BLACK, 170, WHITE, 0.04),
    "x": text("×", F_XB, 120, PINK, 0),
    "hypnotic": text("HYPNOTIC", F_BLACK, 130, CYAN, 0.04),
    "minimal": text("MINIMAL", F_BLACK, 130, PINK, 0.04),
    "psytech": text("PSYTECH", F_BLACK, 130, CYAN, 0.04),
    "groove": text("GROOVE", F_BLACK, 130, PINK, 0.04),
    "music": pill("MUSIC FIRST", 64),
    "exp": pill("EXPERIENCE DRIVEN", 56, CYAN),
    "limited": text("LIMITED CAPACITY", F_XB, 58, PINK, 0.3),
}
SURPRISE = "SURPRISE ACT"
SCRAMBLE = [text("".join(c if c == " " else rng.choice(list("ABCEFHKLNPRSTUXZ?")) for c in SURPRISE),
                 F_BLACK, 104, WHITE, 0.04) for _ in range(10)]
T["surprise"] = text(SURPRISE, F_BLACK, 104, PINK, 0.04)

yy, xx = np.mgrid[0:H, 0:W]
VIG = (1 - 0.5 * np.clip(np.hypot((xx - W / 2) / (W * 0.8), (yy - H / 2) / (H * 0.62)), 0, 1) ** 2)[..., None].astype(np.float32)
GRAIN = [rng.normal(0, 5, (H, W, 1)).astype(np.float32) for _ in range(6)]
STARS = np.zeros((H, W), np.float32)
for _ in range(260):
    STARS[rng.integers(0, H), rng.integers(0, W)] = rng.uniform(120, 255)
STARS = np.asarray(Image.fromarray(STARS.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32)[..., None] * 2.5
del yy, xx


# ---------------------------------------------------------------- timing
def kick_env(t):
    if t < DROP or t >= END_KICK:
        return 0.0
    return math.exp(-((t - DROP) % BEAT) / 0.09)


def roll_env(t):
    hits = [BAR + i * BEAT / 2 for i in range(4)] + [BAR + 2 * BEAT + i * BEAT / 4 for i in range(4)]
    hits += [BAR + 3 * BEAT + i * BEAT / 8 for i in range(8)]
    past = [h for h in hits if h <= t]
    return math.exp(-(t - past[-1]) / 0.04) if past else 0.0


def glitch(frame, amt, t):
    if amt <= 0.02:
        return frame
    s = int(3 + 28 * amt)
    out = frame.copy()
    out[..., 0] = np.roll(frame[..., 0], s, axis=1)
    out[..., 2] = np.roll(frame[..., 2], -s, axis=1)
    if amt > 0.4:
        r = np.random.default_rng(int(t * 1000))
        for _ in range(int(3 + 5 * amt)):
            y, h = r.integers(0, H - 80), r.integers(8, 70)
            out[y:y + h] = np.roll(out[y:y + h], int(r.integers(-110, 110) * amt), axis=1)
    return out


def render(i):
    t = i / FPS
    bar = int(t / BAR)
    u = t - bar * BAR                       # time since the bar (scene) started
    k = kick_env(t)
    punch = math.exp(-u / 0.08)
    shake = (rng.normal(0, 1, 2) * (8 * k + 16 * punch * (bar >= 2))).astype(int)
    cx, cy = W / 2 + shake[0], CY + shake[1]

    dx = int(120 + 100 * math.sin(t * 0.3))
    dy = int(220 + 160 * math.sin(t * 0.21 + 1))
    frame = bg_big[dy:dy + H, dx:dx + W] * (0.8 + 0.5 * k) + STARS * (0.6 + 0.4 * math.sin(t * 3 + 1))
    glitch_amt = 0.2 * k + 0.8 * punch * (bar >= 2)
    flash = 0.0
    eq_alpha = 0.0
    beat_in_bar = int(u / BEAT)

    if bar == 0:                                            # intro
        p = u / BAR
        frame *= 0.4 + 0.6 * p
        paste(frame, geometry(t, t * 0.4, 0.9 + 0.1 * p, 0.85, min(1, p * 1.4)), W / 2, cy)
        paste(frame, typed(T["invite"], (p - 0.2) / 0.6), W / 2, cy)
        glitch_amt = 0.0
    elif bar == 1:                                          # build
        p = u / BAR
        r = roll_env(t)
        paste(frame, geometry(t, t * 0.4 + 3 * p ** 3, 1.0 + 0.25 * p ** 2 + 0.05 * r, 0.7 + 0.3 * r), W / 2, cy)
        paste(frame, T["invite"], W / 2, cy - 260, 1.0, 0.8)
        paste(frame, T["loop"], cx, cy - 60, 0.9 + 0.1 * p, 1.0 if r > 0.2 or p < 0.5 else 0.4)
        if p > 0.5:
            paste(frame, T["hoole"], cx, cy + 160, 0.9 + 0.1 * p, 1.0 if r > 0.2 else 0.4)
        frame *= 0.85 + 0.45 * r
        glitch_amt = 0.1 + 0.5 * r * p
        if p > 0.9:
            flash = (p - 0.9) / 0.1 * 0.85
    elif bar < 9:                                           # content scenes
        eq_alpha = 0.85
        paste(frame, geometry(t, t * 0.35, 1.05 + 0.05 * k, 0.35 + 0.25 * k), W / 2, cy)
        z = (1 + 0.3 * punch) * (1 + 0.03 * k)
        if bar == 2:
            paste(frame, SHIVA, cx, cy, z)
        elif bar == 3:
            paste(frame, T["invite"], cx, cy - 300, 1.0, min(1, u / 0.2))
            paste(frame, T["loop"], cx, cy - 90, z)
            paste(frame, T["hoole"], cx, cy + 140, z)
        elif bar == 4:
            paste(frame, T["28nov"], cx, cy - 70, z)
            paste(frame, typed(T["neer"], (u - 0.2) / 0.4), cx, cy + 110)
        elif bar == 5:
            paste(frame, T["lineup"], W / 2, cy - 320, 1.0, min(1, u / 0.15))
            paste(frame, T["solzi"], cx - W * 1.1 * math.exp(-u / 0.06), cy - 150, 1 + 0.03 * k)
            paste(frame, T["x"], cx, cy + 10, 1 + 0.3 * math.exp(-max(0, u - BEAT) / 0.08), min(1, max(0, (u - BEAT) / 0.05)))
            paste(frame, T["yaksha"], cx + W * 1.1 * math.exp(-max(0, u - 2 * BEAT) / 0.06), cy + 170, 1 + 0.03 * k,
                  1.0 if u >= 2 * BEAT else 0)
        elif bar == 6:
            paste(frame, T["lineup"], W / 2, cy - 320, 1.0)
            paste(frame, T["gtek"], cx - W * 1.1 * math.exp(-u / 0.06), cy - 150, 1 + 0.03 * k)
            paste(frame, T["x"], cx, cy + 10, 1.0, min(1, max(0, (u - BEAT) / 0.05)))
            if u >= 2 * BEAT:
                img = T["surprise"] if u > 3 * BEAT else SCRAMBLE[i % len(SCRAMBLE)]
                paste(frame, img, cx, cy + 160, 1 + 0.03 * k + (0.2 * math.exp(-(u - 3 * BEAT) / 0.08) if u > 3 * BEAT else 0))
                if 2 * BEAT <= u < 3 * BEAT:
                    glitch_amt = max(glitch_amt, 0.3)
        elif bar == 7:
            words = ["hypnotic", "minimal", "psytech", "groove"]
            for j, wd in enumerate(words):
                if beat_in_bar >= j:
                    pj = math.exp(-(u - j * BEAT) / 0.07)
                    paste(frame, T[wd], cx, cy - 300 + j * 200, (1 + 0.35 * pj) * (1 + 0.02 * k))
            glitch_amt = 0.2 * k + 0.7 * math.exp(-(u % BEAT) / 0.06)
        elif bar == 8:
            paste(frame, T["music"], cx, cy - 170, z)
            if u > BEAT:
                paste(frame, T["exp"], cx, cy + 10, 1 + 0.25 * math.exp(-(u - BEAT) / 0.08))
            if u > 2 * BEAT:
                blink = 1.0 if int(u / (BEAT / 2)) % 2 == 0 else 0.35
                paste(frame, T["limited"], cx, cy + 220, 1.0, blink)
        flash = 0.55 * punch
    else:                                                   # full poster + outro
        uu = t - 9 * BAR
        outro = bar >= 11
        zoom = 1.0 + 0.04 * (uu / (3 * BAR)) + 0.015 * k + 0.1 * math.exp(-uu / 0.12)
        big = reveal_img.resize((int(W * zoom), int(H * zoom)), Image.BILINEAR)
        frame = np.zeros((H, W, 3), np.float32)
        paste(frame, big, W / 2 + shake[0], H / 2 + shake[1] + 30)    # slight drop so top text stays clear
        frame *= 0.95 + 0.25 * k
        paste(frame, geometry(t, t * 0.25, 1.0 + 0.04 * k, 0.18 + 0.2 * k), W / 2, 780)
        eq_alpha = 1.0
        flash = 0.75 * math.exp(-uu / 0.1)
        glitch_amt = 0.12 * k + 0.8 * math.exp(-uu / 0.08)
        if outro:
            uo = t - END_KICK
            flash = max(flash, 0.6 * math.exp(-uo / 0.12))
            glitch_amt = 0.7 * math.exp(-uo / 0.1)

    if eq_alpha:
        draw_eq(frame, i, eq_alpha, 70 if bar >= 9 else 130)
    if flash > 0:
        tint = np.array([235, 210, 255], np.float32)
        frame = frame * (1 - flash) + tint * flash
    frame = glitch(frame, glitch_amt, t)
    frame = frame * VIG + GRAIN[i % len(GRAIN)]
    if t > LENGTH - 0.3:
        frame *= max(0.0, (LENGTH - t) / 0.3)
    return np.clip(frame, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    only = [float(x) for x in sys.argv[5:]]
    if only:
        for tt in only:
            Image.fromarray(render(int(tt * FPS))).save(f"{OUT}_{tt:05.2f}.jpg", quality=85)
        sys.exit()
    n = int(LENGTH * FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", TRACK,
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-maxrate", "14M", "-bufsize", "28M",
           "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-9:TP=-1:LRA=7",
           "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
           "-shortest", "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(render(i).tobytes())
        if i % 90 == 0:
            print(f"frame {i}/{n}", flush=True)
    p.stdin.close()
    p.wait()
    print("done", OUT)
