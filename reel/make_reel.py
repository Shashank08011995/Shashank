"""Render a 1080x1920 Instagram Reel from the Psychedelic Hi-Tech Ritual poster.

Every cut, flash and zoom is locked to the 150 BPM grid of make_music.py.
Usage: python3 make_reel.py POSTER.jpg TRACK.wav OUT.mp4 FONT_DIR
"""
import math
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

POSTER, TRACK, OUT, FONT_DIR = sys.argv[1:5]
W, H, FPS = 1080, 1920, 30
BEAT = 0.4
BAR = 1.6
DROP = 3.2
END_KICK = 19.2
LENGTH = 20.8
GREEN = (170, 255, 40)
rng = np.random.default_rng(3)

poster = Image.open(POSTER).convert("RGB")
PW, PH = poster.size
FONT_BLACK = f"{FONT_DIR}/orbitron-black.ttf"
FONT_RAJ = f"{FONT_DIR}/rajdhani-bold.ttf"

# Regions of the poster (pixel boxes in the 1122x1402 original).
BOX = {
    "spiral":  (385, 40, 740, 360),
    "psy":     (145, 355, 990, 497),
    "ritual":  (135, 492, 995, 668),
    "date":    (285, 598, 845, 768),
    "yaksha":  (170, 770, 965, 888),
    "psyveda": (140, 880, 1005, 1010),
    "venue":   (215, 1055, 910, 1165),
    "contact": (315, 1175, 805, 1268),
}


def feathered(img, feather):
    """RGBA copy of img with soft edges so crops melt into the background."""
    w, h = img.size
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rectangle((feather, feather, w - feather, h - feather), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(feather / 2))
    out = img.convert("RGBA")
    out.putalpha(m)
    return out


def element(name, width, feather=26, glow=0.55):
    """Crop a poster region, upscale it and add a green-ish glow halo."""
    crop = poster.crop(BOX[name])
    s = width / crop.width
    crop = crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS)
    pad = 60
    canvas = Image.new("RGB", (crop.width + 2 * pad, crop.height + 2 * pad), (0, 0, 0))
    canvas.paste(crop, (pad, pad))
    arr = np.asarray(canvas).astype(np.float32)
    lum = arr.max(axis=2, keepdims=True)
    bright = np.where(lum > 140, arr, 0).astype(np.uint8)
    halo = np.asarray(Image.fromarray(bright).filter(ImageFilter.GaussianBlur(18))).astype(np.float32)
    arr = np.clip(arr + halo * glow, 0, 255).astype(np.uint8)
    return feathered(Image.fromarray(arr), feather + pad // 2)


def spiral_layer(size):
    crop = poster.crop(BOX["spiral"])
    side = min(crop.size)
    crop = crop.crop(((crop.width - side) // 2, crop.height - side, (crop.width + side) // 2, crop.height))
    crop = crop.resize((size, size), Image.LANCZOS)
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    alpha = (np.clip(1 - r, 0, 1) ** 0.8 * 255).astype(np.uint8)
    out = crop.convert("RGBA")
    out.putalpha(Image.fromarray(alpha))
    return out


def text_layer(text, font, size, fill, spacing=0.18, glow=True):
    f = ImageFont.truetype(font, size)
    chars = list(text)
    widths = [f.getlength(c) for c in chars]
    track = size * spacing
    tw = int(sum(widths) + track * (len(chars) - 1)) + 80
    th = int(size * 1.6) + 40
    img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = 40
    for c, w in zip(chars, widths):
        d.text((x, 20), c, font=f, fill=fill)
        x += w + track
    if glow:
        g = img.filter(ImageFilter.GaussianBlur(12))
        base = Image.new("RGBA", img.size, (0, 0, 0, 0))
        base.alpha_composite(g)
        base.alpha_composite(g)
        base.alpha_composite(img)
        img = base
    return img


# ---------------------------------------------------------------- precomputed assets
print("preparing assets...")
# Blurred, darkened poster as moving background (bigger than frame so it can drift).
BGW, BGH = 1400, 2400
bg_src = poster.resize((int(PW * BGH / PH * 1.0), BGH), Image.LANCZOS)
bg_src = bg_src.crop(((bg_src.width - BGW) // 2, 0, (bg_src.width + BGW) // 2, BGH)) if bg_src.width > BGW else bg_src.resize((BGW, BGH))
bg_big = np.asarray(bg_src.filter(ImageFilter.GaussianBlur(28))).astype(np.float32) * 0.42

# Sharp poster for the reveal: fits 1000 px wide.
REVEAL_W = 1000
reveal = poster.resize((REVEAL_W, int(PH * REVEAL_W / PW)), Image.LANCZOS)
reveal = feathered(reveal, 6)

ELEM = {
    "psy": element("psy", 1000),
    "ritual": element("ritual", 1040, glow=0.8),
    "date": element("date", 960),
    "yaksha": element("yaksha", 1000),
    "psyveda": element("psyveda", 1040, glow=0.8),
    "venue": element("venue", 980),
    "contact": element("contact", 940, glow=0.7),
}
SPIRAL = spiral_layer(900)
SPIRAL_SMALL = spiral_layer(620)

T_SAT = text_layer("THIS SATURDAY", FONT_BLACK, 74, GREEN + (255,))
T_LINEUP = text_layer("LINE UP", FONT_RAJ, 64, (235, 235, 225, 255), 0.5)
T_WHEN = text_layer("MARK  THE  DATE", FONT_RAJ, 52, (235, 235, 225, 255), 0.25)
T_READY = text_layer("ARE YOU READY?", FONT_RAJ, 70, (235, 235, 225, 255), 0.35)

yy, xx = np.mgrid[0:H, 0:W]
VIGNETTE = (1 - 0.55 * np.clip(np.hypot((xx - W / 2) / (W * 0.75), (yy - H / 2) / (H * 0.62)), 0, 1) ** 2)[..., None].astype(np.float32)
SCAN = (1 - 0.10 * ((yy // 3) % 2))[..., None].astype(np.float32)
POST = VIGNETTE * SCAN
GRAIN = [rng.normal(0, 7, (H, W, 1)).astype(np.float32) for _ in range(6)]
del yy, xx

# ---------------------------------------------------------------- timeline
# (start, end, element, entrance, label)
SCENES = [
    (3.2, 4.8, "ritual", "slam", None),
    (4.8, 6.4, "date", "slam", None),
    (6.4, 8.0, "yaksha", "left", "lineup"),
    (8.0, 9.6, "psyveda", "right", "lineup"),
    (9.6, 11.2, "venue", "slam", None),
    (11.2, 12.8, "contact", "slam", None),
]
CUTS = [0.0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6, 11.2, 12.8, 14.4, 16.0, 17.6, 19.2]


def kick_env(t):
    if t < DROP or t >= END_KICK:
        return 0.0
    return math.exp(-((t - DROP) % BEAT) / 0.09)


def cut_env(t, tau=0.07):
    last = max(c for c in CUTS if c <= t + 1e-9)
    return math.exp(-(t - last) / tau), t - last


def snare_env(t):
    hits = [BAR + i * BEAT / 2 for i in range(4)]
    hits += [BAR + 2 * BEAT + i * 0.1 for i in range(4)]
    hits += [BAR + 3 * BEAT + i * 0.05 for i in range(8)]
    past = [h for h in hits if h <= t]
    return math.exp(-(t - past[-1]) / 0.04) if past else 0.0


def paste(frame, img, cx, cy, scale=1.0, alpha=1.0, angle=0.0):
    if scale != 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
    if angle:
        img = img.rotate(angle, resample=Image.BILINEAR)
    x0, y0 = int(cx - img.width / 2), int(cy - img.height / 2)
    x1, y1 = x0 + img.width, y0 + img.height
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    a = np.asarray(img).astype(np.float32)[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    rgb, al = a[..., :3], a[..., 3:4] / 255.0 * alpha
    region = frame[fy0:fy1, fx0:fx1]
    region *= 1 - al
    region += rgb * al


def typed(img, progress):
    """Reveal a text layer left-to-right (glitchy type-on)."""
    w = int(img.width * max(0.0, min(1.0, progress)))
    if w <= 0:
        return None
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img.crop((0, 0, w, img.height)), (0, 0))
    return out


def background(t, k):
    dx = int(150 + 120 * math.sin(t * 0.35))
    dy = int(200 + 180 * math.sin(t * 0.23 + 1))
    bg = bg_big[dy:dy + H, dx:dx + W].copy()
    return bg * (0.75 + 0.6 * k)


def glitch(frame, amount, t):
    """RGB split + horizontal slice displacement."""
    if amount <= 0.02:
        return frame
    s = int(4 + 34 * amount)
    out = frame.copy()
    out[..., 0] = np.roll(frame[..., 0], s, axis=1)
    out[..., 2] = np.roll(frame[..., 2], -s, axis=1)
    if amount > 0.35:
        r = np.random.default_rng(int(t * 1000))
        for _ in range(int(3 + 6 * amount)):
            y = r.integers(0, H - 80)
            h = r.integers(10, 90)
            out[y:y + h] = np.roll(out[y:y + h], int(r.integers(-120, 120) * amount), axis=1)
    return out


def render(i):
    t = i / FPS
    k = kick_env(t)
    ce, since_cut = cut_env(t)
    frame = background(t, k)
    shake = (rng.normal(0, 1, 2) * (14 * k + 22 * ce)).astype(int)
    cx, cy = W / 2 + shake[0], 900 + shake[1]
    glitch_amt = 0.25 * k + 0.9 * ce
    flash = 0.0

    if t < BAR:                                            # intro: spiral wakes up
        p = t / BAR
        frame *= 0.35 + 0.65 * p
        hat = math.exp(-((t - BEAT / 4) % (BEAT / 2)) / 0.05) if t >= BEAT / 4 else 0
        paste(frame, SPIRAL, cx, cy - 40, 0.7 + 0.35 * p, 0.55 + 0.4 * p + 0.15 * hat, angle=-t * 70)
        txt = typed(T_SAT, (t - 0.35) / 0.8)
        if txt is not None and not (0.9 < t < 0.95):
            paste(frame, txt, W / 2, 1420)
        glitch_amt = 0.3 * hat
    elif t < DROP:                                         # build: snare roll, everything accelerates
        p = (t - BAR) / BAR
        s = snare_env(t)
        paste(frame, SPIRAL, W / 2, cy, 1.05 + 0.8 * p ** 2, 0.45, angle=-t * (70 + 400 * p ** 2))
        flick = 1.0 if (s > 0.25 or p < 0.85) else 0.55
        paste(frame, ELEM["psy"], cx, cy, 0.85 + 0.25 * p ** 2 + 0.04 * s, flick)
        txt = typed(T_READY, (p - 0.15) / 0.5)
        if txt is not None:
            paste(frame, txt, W / 2, 1300, 1.0, 0.9)
        frame *= 0.9 + 0.5 * s
        glitch_amt = 0.15 + 0.55 * s * p + 0.5 * max(0, p - 0.85) / 0.15
        if p > 0.9:
            flash = (p - 0.9) / 0.1 * 0.8
    elif t < 12.8:                                         # crop scenes
        sc = next(s for s in SCENES if s[0] <= t < s[1])
        start, end, name, mode, label = sc
        u = t - start
        paste(frame, SPIRAL, W / 2, 900, 1.5 + 0.05 * k, 0.22 + 0.12 * k, angle=-t * 45)
        punch = math.exp(-u / 0.09)
        el = ELEM[name]
        if mode == "slam":
            paste(frame, el, cx, cy, (1 + 0.35 * punch) * (1 + 0.04 * k))
        else:
            sign = -1 if mode == "left" else 1
            off = sign * W * 1.1 * math.exp(-u / 0.06)
            paste(frame, el, cx + off, cy, 1 + 0.04 * k)
            if u < 0.2:                                    # motion streak
                paste(frame, el, cx + off * 1.6, cy, 1.0, 0.35)
        if label == "lineup":
            paste(frame, T_LINEUP, W / 2, 680, 1.0, min(1, u / 0.15))
        if name == "date":
            paste(frame, T_WHEN, W / 2, 1150, 1.0, min(1, max(0, (u - 0.3) / 0.2)))
        flash = 0.75 * punch
    else:                                                  # full poster reveal + outro
        u = t - 12.8
        out = t >= END_KICK
        paste(frame, SPIRAL, W / 2, 900, 1.9, 0.18 + 0.1 * k, angle=-t * 30)
        zoom = 0.94 + 0.05 * (u / 8.0) + 0.018 * k + 0.12 * math.exp(-u / 0.12)
        paste(frame, reveal, cx, 930 + shake[1], zoom)
        pulse = 0.5 + 0.5 * math.sin(t * math.pi / BEAT)
        paste(frame, T_SAT, W / 2, 160, 0.95, 0.75 + 0.25 * (k if not out else pulse))
        flash = 0.8 * math.exp(-u / 0.1)
        if out:
            uo = t - END_KICK
            flash = max(flash, 0.7 * math.exp(-uo / 0.12))
            glitch_amt = 0.8 * math.exp(-uo / 0.1)
        else:
            glitch_amt = 0.15 * k + (0.9 * ce if since_cut < 0.12 else 0)

    if flash > 0:
        tint = np.array([200, 255, 170], np.float32)
        frame = frame * (1 - flash) + tint * flash
    frame = glitch(frame, glitch_amt, t)
    frame = frame * POST + GRAIN[i % len(GRAIN)]
    if t > LENGTH - 0.3:
        frame *= (LENGTH - t) / 0.3
    return np.clip(frame, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    only = [float(x) for x in sys.argv[5:]]                # optional: preview frames at given times
    if only:
        for tt in only:
            Image.fromarray(render(int(tt * FPS))).save(f"{OUT}_{tt:05.2f}.jpg", quality=85)
        sys.exit()
    n = int(LENGTH * FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", TRACK,
           "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-profile:v", "high",
           "-af", "loudnorm=I=-9:TP=-1:LRA=7",
           "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
           "-shortest", "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(render(i).tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{n}", flush=True)
    p.stdin.close()
    p.wait()
    print("done", OUT)
