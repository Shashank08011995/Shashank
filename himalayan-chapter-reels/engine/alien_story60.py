"""60-second event story: "Signal from Tapovan" (alien DJs are the heroes from frame one).
Cockpit close-up -> signal decoded -> hyperspace -> arrival over the Himalayas -> tractor-beam landing on the
cafe rooftop -> drop with lasers, close-ups of each DJ -> date/time -> punchline + booking card.
Everything is drawn in code. Music: music_edm (pre-groove, drop, breakdown) + sci-fi SFX + alien chatter.
usage: python engine/alien_story60.py episodes/event_fullmoon_2411_aliens60.json
"""
import json, math, os, re, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from music_edm import make_edm, SR

C = json.load(open(sys.argv[1])); base = os.path.dirname(os.path.abspath(sys.argv[1]))
P = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(base, x))
W, H, FPS = 1080, 1920, 30
TT = C['times']   # named scene starts
T_COCK, T_EXT, T_HYPER, T_ARR, T_LAND, DROP, T_CUA, T_CUB, T_WIDE, T_INFO, BRK, TOTAL = (TT[k] for k in
    ('cockpit', 'exterior', 'hyper', 'arrive', 'land', 'drop', 'close_a', 'close_b', 'wide', 'info', 'end', 'total'))
BPM = C.get('bpm', 120); BEAT = 60 / BPM
SILVER=(236,240,255); VIOLET=(176,120,255); CYAN=(110,230,255); GOLD=(226,186,104); CREAM=(247,240,222)
WHITE=(236,238,246); PINK=(255,90,200); LIME=(150,255,170); INK=(16,16,28)
FD = P(C['assets']['fonts']); INTER = '/usr/share/fonts/opentype/inter/'; MONO = '/usr/share/fonts/truetype/jetbrains-mono/'
def F(n, s):
    if n.startswith('Inter'): return ImageFont.truetype(INTER + n, s)
    if n.startswith('JetBrains'): return ImageFont.truetype(MONO + n, s)
    return ImageFont.truetype(os.path.join(FD, n), s)
cx = W // 2
ease = lambda p: 1 - (1 - min(max(p, 0), 1)) ** 3
def smooth(p): p = min(max(p, 0), 1); return p * p * (3 - 2 * p)
def env(t, a, b, fi=0.35, fo=0.3): return float(np.clip(min((t - a) / fi, (b - t) / fo), 0, 1))
rng = np.random.default_rng(42)
KICKS = np.arange(DROP, BRK - 1e-6, BEAT)
def kick_env(t, tau=0.12):
    if t < DROP or t >= BRK: return 0.0
    k = KICKS[KICKS <= t + 1e-9]; return math.exp(-(t - k[-1]) / tau) if len(k) else 0.0
def vgrad(top, bot, h, w):
    y = np.linspace(0, 1, h)[:, None, None]
    return (np.array(top, np.float32) * (1 - y) + np.array(bot, np.float32) * y) * np.ones((1, w, 1), np.float32)
def fill_grad(mask, top, bot):
    g = Image.fromarray(vgrad(top, bot, mask.size[1], mask.size[0]).astype(np.uint8)).convert('RGBA'); g.putalpha(mask); return g
def glow(img, r, k=1.8):
    g = img.filter(ImageFilter.GaussianBlur(r)); g.putalpha(g.split()[3].point(lambda v: min(255, int(v * k)))); return g

# ============================ characters ============================
def rot_ellipse(cxe, cye, rx, ry, ang, n=60):
    a = math.radians(ang); return [(cxe + rx * math.cos(t) * math.cos(a) - ry * math.sin(t) * math.sin(a),
                                    cye + rx * math.cos(t) * math.sin(a) + ry * math.sin(t) * math.cos(a)) for t in np.linspace(0, 2 * math.pi, n)]
OY = 120
def alien_2x(kind, blink=False, talk=False):
    S2W, S2H, oy = 600, 900, OY
    im = Image.new('RGBA', (S2W, S2H), (0, 0, 0, 0))
    if kind == 'A': skin_t, skin_b, rim, body_t, body_b, led, lid = (140, 240, 210), (30, 130, 125), (210, 255, 240), (34, 36, 52), (12, 12, 22), CYAN, (90, 200, 180)
    else:           skin_t, skin_b, rim, body_t, body_b, led, lid = (190, 160, 255), (80, 52, 170), (235, 220, 255), (70, 40, 110), (24, 14, 44), PINK, (150, 120, 235)
    bm = Image.new('L', (S2W, S2H), 0); bd = ImageDraw.Draw(bm)
    bd.polygon([(40, S2H), (70, 700), (190, 640), (410, 640), (530, 700), (560, S2H)], fill=255)
    bd.ellipse([60, 640, 220, 790], fill=255); bd.ellipse([380, 640, 540, 790], fill=255)
    im.alpha_composite(fill_grad(bm, body_t, body_b)); d = ImageDraw.Draw(im)
    if kind == 'A':
        d.ellipse([175, 585, 425, 700], fill=(20, 22, 34)); d.line([(270, 680), (262, 790)], fill=CYAN, width=6); d.line([(330, 680), (338, 790)], fill=CYAN, width=6)
    else:
        d.polygon([(200, 640), (300, 720), (400, 640), (380, 610), (220, 610)], fill=(40, 22, 70)); d.line([(300, 720), (300, S2H)], fill=(130, 100, 200), width=5)
        for x in range(90, 520, 22): d.line([(x, 860), (x, S2H)], fill=(50, 30, 80), width=3)
    nm = Image.new('L', (S2W, S2H), 0); ImageDraw.Draw(nm).rounded_rectangle([255, 600, 345, 640], 30, fill=255)
    im.alpha_composite(fill_grad(nm, skin_b, (20, 90, 90) if kind == 'A' else (60, 40, 130)))
    hm = Image.new('L', (S2W, S2H), 0); hd = ImageDraw.Draw(hm)
    if kind == 'A':
        hd.ellipse([105, 20 + oy, 495, 420 + oy], fill=255); hd.polygon([(130, 290 + oy), (470, 290 + oy), (360, 520 + oy), (240, 520 + oy)], fill=255); hd.ellipse([228, 455 + oy, 372, 545 + oy], fill=255)
    else:
        hd.ellipse([95, 10 + oy, 505, 430 + oy], fill=255); hd.polygon([(115, 300 + oy), (485, 300 + oy), (380, 515 + oy), (220, 515 + oy)], fill=255); hd.ellipse([215, 445 + oy, 385, 540 + oy], fill=255)
    hm = hm.filter(ImageFilter.GaussianBlur(1.2)); head = fill_grad(hm, skin_t, skin_b)
    rl = ImageChops.subtract(hm, hm.transform(hm.size, Image.AFFINE, (1, 0, 10, 0, 1, 12))).filter(ImageFilter.GaussianBlur(4))
    rimL = Image.new('RGBA', (S2W, S2H), rim + (0,)); rimL.putalpha(rl.point(lambda v: int(v * 0.8))); head.alpha_composite(rimL)
    sh = Image.new('RGBA', (S2W, S2H), (0, 0, 0, 0)); ImageDraw.Draw(sh).ellipse([150, 380 + oy, 450, 560 + oy], fill=(0, 0, 0, 60))   # soft chin shade
    head.alpha_composite(sh.filter(ImageFilter.GaussianBlur(30)))
    tex = np.asarray(Image.fromarray((np.random.default_rng(7 if kind == 'A' else 9).random((S2H // 6, S2W // 6)) * 255).astype(np.uint8)).resize((S2W, S2H), Image.BICUBIC)).astype(np.float32) / 255
    fine = np.random.default_rng(3).normal(0, 1, (S2H, S2W)).astype(np.float32)
    ha = np.asarray(head).astype(np.float32); m_ = (np.asarray(hm).astype(np.float32) / 255)[..., None]
    ha[..., :3] = np.clip(ha[..., :3] * (1 + (0.16 * (tex - 0.5) + 0.03 * fine)[..., None] * m_), 0, 255); head = Image.fromarray(ha.astype(np.uint8), 'RGBA')
    sp = Image.new('RGBA', (S2W, S2H), (0, 0, 0, 0)); dsp = ImageDraw.Draw(sp)
    dsp.ellipse([170, 50 + oy, 330, 150 + oy], fill=(255, 255, 255, 120)); dsp.ellipse([380, 380 + oy, 430, 430 + oy], fill=(255, 255, 255, 60))
    sp = sp.filter(ImageFilter.GaussianBlur(26)); sp.putalpha(ImageChops.multiply(sp.split()[3], hm)); head.alpha_composite(sp)
    im.alpha_composite(head); d = ImageDraw.Draw(im)
    if kind == 'A':
        for ex, ang in ((215, 28), (385, -28)):
            if blink:
                d.polygon(rot_ellipse(ex, 315 + oy, 82, 44, ang), fill=lid); d.line([(ex - 70, 318 + oy), (ex + 70, 318 + oy)], fill=(20, 70, 70), width=8)
            else:
                d.polygon(rot_ellipse(ex, 315 + oy, 82, 44, ang), fill=(8, 10, 18)); d.polygon(rot_ellipse(ex - 18, 300 + oy, 16, 9, ang), fill=(235, 250, 255))
                d.arc([ex - 60, 285 + oy, ex + 50, 345 + oy], 20, 70, fill=(60, 200, 230), width=5)
        if talk: d.ellipse([272, 410 + oy, 328, 462 + oy], fill=(20, 40, 40)); d.ellipse([285, 438 + oy, 315, 458 + oy], fill=(220, 90, 120))
        else: d.arc([268, 420 + oy, 332, 452 + oy], 20, 160, fill=(20, 80, 80), width=7)
    else:
        for ex, ey, rx, ry, ang in ((300, 290, 48, 56, 0), (192, 335, 46, 34, 22), (408, 335, 46, 34, -22)):
            if blink:
                d.polygon(rot_ellipse(ex, ey + oy, rx, ry, ang), fill=lid); d.line([(ex - rx, ey + oy + 4), (ex + rx, ey + oy + 4)], fill=(50, 20, 90), width=7)
            else:
                d.polygon(rot_ellipse(ex, ey + oy, rx, ry, ang), fill=(10, 6, 20)); d.polygon(rot_ellipse(ex, ey + oy, rx * 0.45, ry * 0.45, ang), fill=(255, 190, 60))
                d.polygon(rot_ellipse(ex, ey + oy, rx * 0.18, ry * 0.3, ang), fill=(20, 8, 10))
                d.ellipse([ex - rx * 0.5 - 6, ey + oy - ry * 0.5 - 6, ex - rx * 0.5 + 6, ey + oy - ry * 0.5 + 6], fill=(255, 245, 230))
        if talk: d.ellipse([255, 405 + oy, 345, 490 + oy], fill=(30, 10, 50)); d.rectangle([265, 412 + oy, 335, 425 + oy], fill=(240, 235, 255))
        else: d.chord([250, 400 + oy, 350, 470 + oy], 0, 180, fill=(30, 10, 50)); d.rectangle([262, 435 + oy, 338, 446 + oy], fill=(240, 235, 255))
        bm2 = Image.new('L', (S2W, S2H), 0); ImageDraw.Draw(bm2).chord([88, 0 + oy, 512, 360 + oy], 180, 360, fill=255); ImageDraw.Draw(bm2).rounded_rectangle([85, 150 + oy, 515, 215 + oy], 25, fill=255)
        im.alpha_composite(fill_grad(bm2, (40, 40, 56), (16, 16, 26))); d = ImageDraw.Draw(im)
        for x in range(105, 500, 24): d.line([(x, 158 + oy), (x, 208 + oy)], fill=(58, 58, 78), width=5)
        d.ellipse([280, -10 + oy, 320, 30 + oy], fill=(60, 60, 80))
    d.arc([88, -6 + oy, 512, 470 + oy], 195, 345, fill=(20, 20, 28), width=30)
    for x0 in (62, 470):
        d.rounded_rectangle([x0, 230 + oy, x0 + 68, 380 + oy], 30, fill=(22, 22, 30)); d.rounded_rectangle([x0 + 10, 245 + oy, x0 + 58, 365 + oy], 22, outline=led, width=6)
    if kind == 'A':
        d.line([(230, 60 + oy), (190, -60 + oy), (165, -95 + oy)], fill=(60, 170, 150), width=9, joint='curve')
        d.line([(370, 60 + oy), (410, -60 + oy), (435, -95 + oy)], fill=(60, 170, 150), width=9, joint='curve')
    return im
BASE = {(k, b, tk): alien_2x(k, b, tk) for k in 'AB' for b in (False, True) for tk in (False, True)}
_cache = {}
def sprite(kind, scale, blink=False, talk=False):
    key = (kind, round(scale, 3), blink, talk)
    if key not in _cache:
        b = BASE[(kind, blink, talk)]; _cache[key] = b.resize((int(b.width * scale), int(b.height * scale)), Image.LANCZOS)
    return _cache[key]
ANT = [(165, -95 + OY), (435, -95 + OY)]
def is_blink(t, ph): return ((t + ph) % 3.1) < 0.12
def place_alien(fr, kind, x, top, scale, t, talk=False, tilt=0.0, alpha=1.0, glow_ant=0.0, ph=0.0):
    sp = sprite(kind, scale, is_blink(t, ph), talk and int(t * 9) % 2 == 0)
    if tilt: sp = sp.rotate(tilt, resample=Image.BICUBIC, center=(sp.width / 2, sp.height * 0.6))
    if alpha < 1:
        sp = sp.copy(); a = sp.split()[3]; m = Image.new('L', sp.size, 0); ImageDraw.Draw(m).rectangle([0, 0, sp.width, int(sp.height * alpha)], fill=255); sp.putalpha(ImageChops.multiply(a, m))
    x0 = int(x - sp.width / 2); fr.alpha_composite(sp, (x0, int(top)))
    if kind == 'A' and glow_ant >= 0:
        g = Image.new('RGBA', (W, H), (0, 0, 0, 0)); dg = ImageDraw.Draw(g)
        for (ax, ay) in ANT:
            r = (14 + 8 * glow_ant) * scale / 0.7; px, py = x0 + ax * scale, top + ay * scale
            dg.ellipse([px - r, py - r, px + r, py + r], fill=LIME + (int(255 * min(1, alpha * 2)),))
        fr.alpha_composite(glow(g, 12, 1.5)); fr.alpha_composite(g)
    return x0

def ufo_sprite():
    w, h = 760, 420; im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    dm = Image.new('L', (w, h), 0); ImageDraw.Draw(dm).chord([230, 20, 530, 300], 180, 360, fill=255)
    dome = fill_grad(dm, (170, 240, 255), (40, 110, 160)); dome.putalpha(dm.point(lambda v: int(v * 0.85))); im.alpha_composite(dome)
    for ex0, col in ((310, (60, 170, 140)), (410, (130, 100, 220))):
        d.ellipse([ex0 - 30, 75, ex0 + 30, 145], fill=col); d.polygon(rot_ellipse(ex0 - 11, 112, 11, 7, 25), fill=(5, 8, 12)); d.polygon(rot_ellipse(ex0 + 11, 112, 11, 7, -25), fill=(5, 8, 12))
    d.arc([250, 40, 420, 200], 200, 250, fill=(240, 255, 255), width=8)
    sm = Image.new('L', (w, h), 0); ImageDraw.Draw(sm).ellipse([20, 130, 740, 300], fill=255); im.alpha_composite(fill_grad(sm, (215, 220, 236), (70, 74, 96)))
    d = ImageDraw.Draw(im); d.ellipse([20, 200, 740, 300], outline=(40, 42, 60), width=6)
    d.ellipse([230, 250, 530, 330], fill=(40, 44, 66)); d.ellipse([270, 270, 490, 320], fill=(120, 255, 220))
    return im.resize((w // 2, h // 2), Image.LANCZOS)
UFO = ufo_sprite(); UFO_LIGHTS = [(20 + 720 * (0.5 + 0.48 * math.cos(a))) / 2 for a in np.linspace(0.15, math.pi - 0.15, 9)]
def draw_ufo(fr, x, y, s, t, tilt=0.0, beat=0.0):
    u = UFO if s == 1.0 else UFO.resize((max(2, int(UFO.width * s)), max(2, int(UFO.height * s))), Image.BICUBIC)
    if tilt: u = u.rotate(tilt, resample=Image.BICUBIC, expand=True)
    g = Image.new('RGBA', u.size, (0, 0, 0, 0)); ImageDraw.Draw(g).ellipse([u.width * 0.3, u.height * 0.55, u.width * 0.7, u.height * 0.85], fill=(120, 255, 220, 200))
    fr.alpha_composite(glow(g, max(2, 18 * s), 2.5), (int(x - u.width / 2), int(y - u.height / 2))); fr.alpha_composite(u, (int(x - u.width / 2), int(y - u.height / 2)))
    if not tilt:
        d = ImageDraw.Draw(fr); cols = [CYAN, PINK, LIME, GOLD]
        for i, lx in enumerate(UFO_LIGHTS):
            on = (int(t * 8) + i) % 3 == 0 or beat > 0.5; c = cols[i % 4] if on else (60, 60, 80)
            px = x - u.width / 2 + lx * s; py = y - u.height / 2 + 128 * s; d.ellipse([px - 7 * s, py - 7 * s, px + 7 * s, py + 7 * s], fill=c)

# ============================ scenery ============================
def ridge(w, base_y, amp, rough, seed, n=1025):
    r = np.random.default_rng(seed); y = np.zeros(n); y[0], y[-1] = r.uniform(-1, 1, 2) * amp; step = n - 1; a = amp
    while step > 1:
        half = step // 2
        for i in range(half, n - 1, step): y[i] = (y[i - half] + y[i + half]) / 2 + r.uniform(-a, a)
        step = half; a *= rough
    return np.interp(np.arange(w), np.linspace(0, w, n), y) + base_y
def mountain(w, base_y, amp, rough, seed, top, bot, rim=None, snow=False):
    y = ridge(w, base_y, amp, rough, seed); img = Image.fromarray(vgrad(top, bot, H, w).astype(np.uint8)).convert('RGBA')
    m = Image.new('L', (w, H), 0); ImageDraw.Draw(m).polygon([(0, H)] + list(zip(range(w), y)) + [(w, H)], fill=255)
    if snow:
        sn = Image.new('L', (w, H), 0); ds = ImageDraw.Draw(sn)
        for x in range(0, w, 2):
            dep = max(0, (base_y - 120 - y[x]) * 0.55)
            if dep > 4: ds.line([(x, y[x]), (x, y[x] + dep)], fill=int(min(210, 40 + dep * 2)), width=2)
        img = Image.composite(Image.new('RGBA', (w, H), (196, 206, 238, 255)), img, sn.filter(ImageFilter.GaussianBlur(3)))
    img.putalpha(m)
    if rim:
        rl = Image.new('RGBA', (w, H), (0, 0, 0, 0)); ImageDraw.Draw(rl).line(list(zip(range(w), y)), fill=rim, width=3); img.alpha_composite(rl.filter(ImageFilter.GaussianBlur(1.5)))
    return img, y
BACK, _ = mountain(W + 120, 960, 380, 0.5, 3, (52, 60, 104), (24, 26, 56), rim=(170, 180, 230, 200), snow=True)
MID, _ = mountain(W + 80, 1120, 120, 0.55, 8, (26, 30, 62), (14, 16, 36), rim=(90, 96, 160, 150))
NEAR, ny = mountain(W + 40, 1250, 60, 0.6, 13, (12, 14, 30), (6, 7, 16)); d = ImageDraw.Draw(NEAR)
for x in range(-10, W + 40, 18):
    hh = rng.uniform(60, 140); bx = x + rng.uniform(-6, 6); by = ny[min(max(int(bx), 0), W + 39)] + 8
    for k in range(5):
        wd = hh * (0.36 - 0.06 * k); yy = by - hh * k / 5; d.polygon([(bx - wd, yy), (bx + wd, yy), (bx, yy - hh * 0.38)], fill=(6, 7, 16, 255))
SKY = Image.fromarray(vgrad((10, 12, 40), (34, 24, 66), H, W).astype(np.uint8)).convert('RGBA')
NST = 360; SX, SY = rng.uniform(0, W, NST), rng.uniform(0, H, NST); SRAD = rng.uniform(0.6, 2.2, NST); SPH = rng.uniform(0, 6.3, NST)
def stars(fr, t, ymax=1200, box=None):
    d = ImageDraw.Draw(fr)
    for i in range(NST):
        if SY[i] > ymax: continue
        if box and not (box[0] < SX[i] < box[2] and box[1] < SY[i] < box[3]): continue
        b = 0.5 + 0.5 * math.sin(t * 1.5 + SPH[i]); c = int(90 + 165 * b); r = SRAD[i]
        d.ellipse([SX[i] - r, SY[i] - r, SX[i] + r, SY[i] + r], fill=(c, c, min(255, c + 25)))
def moon_img(r, seed=1):
    rr = np.random.default_rng(seed); s = 2 * r; yy, xx = np.mgrid[0:s, 0:s]; dd = np.sqrt((xx - r) ** 2 + (yy - r) ** 2) / r
    n = np.asarray(Image.fromarray((rr.random((max(2, s // 14),) * 2) * 255).astype(np.uint8)).resize((s, s), Image.BICUBIC)).astype(np.float32) / 255
    n2 = np.asarray(Image.fromarray((rr.random((max(2, s // 45),) * 2) * 255).astype(np.uint8)).resize((s, s), Image.BICUBIC)).astype(np.float32) / 255
    sh = 0.8 + 0.1 * n + 0.14 * n2 - 0.16 * dd ** 3
    rgb = np.stack([sh * 220, sh * 222, sh * 234], -1).clip(0, 255); a = np.clip((1 - dd) * r / 1.5, 0, 1) * 255
    im = Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), 'RGBA')
    g = Image.new('RGBA', (int(s * 2.2), int(s * 2.2)), (0, 0, 0, 0)); ImageDraw.Draw(g).ellipse([s * 0.45, s * 0.45, s * 1.75, s * 1.75], fill=(170, 185, 255, 150))
    g = g.filter(ImageFilter.GaussianBlur(s * 0.22)); g.alpha_composite(im, (int(s * 0.6), int(s * 0.6))); return g
MOON_BIG, MOON_ROOF, MOON_END, MOON_WIN = moon_img(420), moon_img(140, 2), moon_img(215, 3), moon_img(120, 4)
def nebula():
    n = np.zeros((H, W, 3), np.float32); yy, xx = np.mgrid[0:H, 0:W]
    for col, cxn, cyn, sc in (((120, 60, 200), 300, 700, 380), ((40, 160, 200), 800, 400, 320), ((200, 60, 150), 650, 1100, 300)):
        n += np.array(col, np.float32)[None, None] * np.exp(-((xx - cxn) ** 2 + (yy - cyn) ** 2) / sc ** 2)[..., None] * 0.5
    tex = np.asarray(Image.fromarray((rng.random((60, 34)) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
    n *= (0.5 + 0.8 * tex)[..., None]; return Image.fromarray(np.clip(vgrad((4, 4, 14), (10, 6, 22), H, W) + n, 0, 255).astype(np.uint8)).convert('RGBA')
NEB = nebula()
def string_lights():
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im); bulbs = []
    for (x0, y0, x1, y1, sag) in ((-20, 560, 560, 600, 70), (520, 600, 1100, 560, 70)):
        pts = [(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u + sag * 4 * u * (1 - u)) for u in np.linspace(0, 1, 60)]; d.line(pts, fill=(20, 18, 24), width=3); bulbs += pts[2::5]
    g = Image.new('RGBA', (W, H), (0, 0, 0, 0)); dg = ImageDraw.Draw(g)
    for (x, y) in bulbs: dg.ellipse([x - 12, y - 2, x + 12, y + 22], fill=(255, 190, 110, 200))
    im.alpha_composite(glow(g, 14, 1.6)); d = ImageDraw.Draw(im)
    for (x, y) in bulbs: d.ellipse([x - 5, y + 4, x + 5, y + 16], fill=(255, 226, 170))
    return im
LIGHTS = string_lights()
logo = Image.open(P(C['assets']['logo']))
def booth():
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).rectangle([150, 1390, 930, 1720], fill=255); im.alpha_composite(fill_grad(m, (30, 30, 46), (8, 8, 16)))
    d.rounded_rectangle([130, 1350, 950, 1396], 12, fill=(46, 48, 66)); d.line([(140, 1352), (940, 1352)], fill=(120, 125, 160), width=2)
    lh = 260; lw = int(logo.width * lh / logo.height); im.alpha_composite(logo.resize((lw, lh), Image.LANCZOS), (cx - lw // 2, 1420)); return im
BOOTH = booth()
def draw_decks(fr, t, on):
    d = ImageDraw.Draw(fr)
    for x in (300, 780):
        d.ellipse([x - 120, 1336, x + 120, 1376], fill=(14, 14, 20), outline=(90, 95, 130), width=2)
        a = (t * 4) % (2 * math.pi); px, py = x + 85 * math.cos(a), 1356 + 14 * math.sin(a); d.ellipse([px - 5, py - 3, px + 5, py + 3], fill=CYAN if on else (90, 90, 110))
    d.rectangle([480, 1340, 600, 1374], fill=(24, 24, 34))
    for i in range(6): d.rectangle([490 + i * 18, 1346, 498 + i * 18, 1368], fill=PINK if on and (int(t * 8) + i) % 2 else (60, 60, 80))
PEOPLE = [dict(x=-20 + i * 72 + rng.uniform(-18, 18), h=rng.uniform(140, 190), ph=rng.uniform(0, 1), arms=rng.random() < 0.7, amp=rng.uniform(16, 32), point=rng.random() < 0.4) for i in range(16)]
def draw_crowd(fr, t, party, look_up=False):
    d = ImageDraw.Draw(fr); k = kick_env(t, 0.18)
    for p in PEOPLE:
        bounce = p['amp'] * k if party else 4 * math.sin(2 * math.pi * (t * 0.5 + p['ph'])); top = 1920 - p['h'] - bounce; x = p['x']; col = (4, 4, 10)
        oc = (70, 56, 140) if party else (40, 36, 80)
        d.rounded_rectangle([x - 36, top + 56, x + 36, 1960], 28, fill=col, outline=oc, width=3); d.ellipse([x - 25, top, x + 25, top + 54], fill=col, outline=oc, width=3)
        if (p['arms'] and party) or (look_up and p['point']):
            sides = (-1, 1) if party else (1,)
            for s in sides:
                ex, ey = x + s * 50, top - 40 - bounce * 0.3
                d.line([(x + s * 28, top + 74), (ex, ey)], fill=col, width=17); d.ellipse([ex - 10, ey - 10, ex + 10, ey + 10], fill=col)
LCOL1 = [VIOLET, CYAN, PINK, LIME, CYAN, VIOLET]; LCOL2 = [CYAN, LIME, CYAN, VIOLET, LIME, CYAN]
def lasers(t, inten, pal, oy=1360):
    L = Image.new('RGB', (W, H), 0); d = ImageDraw.Draw(L); ph = 2 * math.pi * (t - DROP) / (8 * BEAT)
    for i in range(6):
        a = math.radians(-90 + (i - 2.5) * 17 + 24 * math.sin(ph + i * 0.7)); ex, ey = cx + 2600 * math.cos(a), oy + 2600 * math.sin(a); pa = a + math.pi / 2
        col = tuple(int(c * inten) for c in pal[i]); d.polygon([(cx, oy), (ex + 16 * math.cos(pa), ey + 16 * math.sin(pa)), (ex - 16 * math.cos(pa), ey - 16 * math.sin(pa))], fill=col)
    g = L.filter(ImageFilter.GaussianBlur(14)); return ImageChops.add(ImageChops.multiply(L, Image.new('RGB', (W, H), (150, 150, 150))), ImageChops.add(g, g))
def screen(fr, L): return Image.fromarray(np.asarray(ImageChops.screen(fr.convert('RGB'), L))).convert('RGBA')

# ============================ text ============================
def tw(d, txt, font, track): return sum(d.textlength(c, font=font) for c in txt) + track * (len(txt) - 1)
def spaced(d, x, y, txt, font, fill, track):
    x = x - tw(d, txt, font, track) / 2
    for c in txt: d.text((x, y), c, font=font, fill=fill); x += d.textlength(c, font=font) + track
def glow_layer(items):
    out = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    for it in items:
        im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im); f = F(it['font'], it['size']); col = tuple(it.get('color', WHITE)); x = it.get('x', cx)
        if it.get('track') is not None: spaced(d, x, it['y'], it['text'], f, col, it['track'])
        else: d.text((x, it['y']), it['text'], font=f, fill=col, anchor='mt')
        a = im.split()[3]
        if it.get('glow'):
            g = Image.new('RGBA', (W, H), tuple(it['glow']) + (0,)); g.putalpha(a.filter(ImageFilter.GaussianBlur(it.get('glow_r', 18))).point(lambda v: min(255, int(v * 2.2)))); out.alpha_composite(g)
        sh = Image.new('RGBA', (W, H), (0, 0, 0, 0)); sh.putalpha(a.filter(ImageFilter.GaussianBlur(8)).point(lambda v: int(v * 0.85))); out.alpha_composite(sh, (0, 4)); out.alpha_composite(im)
    return out
def put(fr, L, a, dy=0):
    if a <= 0: return
    if a < 1: L = L.copy(); L.putalpha(L.split()[3].point(lambda v: int(v * a)))
    fr.alpha_composite(L, (0, int(dy)))
def glitch(fr, L, t, t0, dur=0.35):
    p = (t - t0) / dur
    if p < 0: return
    if p >= 1: fr.alpha_composite(L); return
    r = np.random.default_rng(int(t * 1000)); amt = (1 - p) ** 2; bb = L.getbbox() or (0, 0, W, H); y0, y1 = bb[1] - 10, bb[3] + 10
    out = Image.new('RGBA', (W, H), (0, 0, 0, 0)); hs = max(1, (y1 - y0) // 8)
    for i in range(8): ys = y0 + i * hs; out.alpha_composite(L.crop((0, ys, W, ys + hs)), (int(r.uniform(-110, 110) * amt), ys))
    a = out.split()[3]
    for col, off in ((CYAN, -int(16 * amt)), (PINK, int(16 * amt))):
        c = Image.new('RGBA', (W, H), col + (0,)); c.putalpha(a.point(lambda v: int(v * 0.6 * amt))); fr.alpha_composite(c, (off, 0))
    out.putalpha(a.point(lambda v: int(v * min(1, p * 2.5)))); fr.alpha_composite(out)
def typewriter(fr, lines, t, x=110, y0=250, size=38, gap=62, color=CYAN, cps=26):
    g = Image.new('RGBA', (W, H), (0, 0, 0, 0)); dg = ImageDraw.Draw(g); f = F('JetBrainsMono-Bold.ttf', size)
    for i, (t0, txt) in enumerate(lines):
        if t < t0: continue
        n = min(len(txt), int((t - t0) * cps)); s = txt[:n]
        if n < len(txt) or (i == len(lines) - 1 and int(t * 2.5) % 2 == 0): s += '▌'
        dg.text((x, y0 + i * gap), s, font=f, fill=color + (255,))
    fr.alpha_composite(glow(g, 8, 1.4)); fr.alpha_composite(g)
BUB = {}
def bubble(fr, text, x, y, tail, t, t0, t1, who='A'):
    """comic speech bubble centred at (x, y) with a tail pointing at `tail`; pops in, fades out."""
    if not (t0 <= t < t1): return
    key = (text, x, y, tail)
    if key not in BUB:
        f = F('Inter-ExtraBold.otf', 50); tmp = ImageDraw.Draw(Image.new('RGBA', (4, 4))); w = int(tmp.textlength(text, font=f)) + 70; h = 104
        S = 2; im = Image.new('RGBA', (W * S, H * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        bx0, by0 = (x - w / 2) * S, (y - h / 2) * S; bx1, by1 = (x + w / 2) * S, (y + h / 2) * S
        ring = (CYAN if who == 'A' else PINK if who == 'B' else GOLD)
        d.polygon([((x - 22) * S, (y + h / 2 - 6) * S), ((x + 22) * S, (y + h / 2 - 6) * S), (tail[0] * S, tail[1] * S)], fill=(226, 230, 238, 255))
        d.rounded_rectangle([bx0, by0, bx1, by1], 46 * S, fill=(226, 230, 238, 255), outline=ring + (255,), width=6 * S)
        d.polygon([((x - 18) * S, (y + h / 2 - 12) * S), ((x + 18) * S, (y + h / 2 - 12) * S), (tail[0] * S, tail[1] * S - 8 * S)], fill=(226, 230, 238, 255))
        d.text((x * S, y * S + 2 * S), text, font=F('Inter-ExtraBold.otf', 50 * S), fill=INK + (255,), anchor='mm')
        im = im.resize((W, H), Image.LANCZOS); bb = im.getbbox(); BUB[key] = (im.crop(bb), bb)
    im, bb = BUB[key]; p = (t - t0) / 0.22; s = 0.6 + 0.4 * ease(p) + 0.06 * math.sin(min(1, p) * math.pi)
    a = min(1, (t1 - t) / 0.2)
    ims = im.resize((max(2, int(im.width * s)), max(2, int(im.height * s))), Image.BICUBIC)
    if a < 1: ims.putalpha(ims.split()[3].point(lambda v: int(v * a)))
    cxb, cyb = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2; fr.alpha_composite(ims, (int(cxb - ims.width / 2), int(cyb - ims.height / 2)))
def hud_frame(fr, a, box=(60, 170, 1020, 1000)):
    if a <= 0: return
    L = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(L); c = CYAN + (int(200 * a),); s = 60; x0, y0, x1, y1 = box
    for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(x, y), (x + sx * s, y)], fill=c, width=4); d.line([(x, y), (x, y + sy * s)], fill=c, width=4)
    fr.alpha_composite(L)
def brackets(fr, x0, y0, x1, y1, a, col=CYAN):
    if a <= 0: return
    d = ImageDraw.Draw(fr); c = col + (int(255 * a),); s = 34
    for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(x, y), (x + sx * s, y)], fill=c, width=4); d.line([(x, y), (x, y + sy * s)], fill=c, width=4)

T_t1 = glow_layer([{'text': C['title1'], 'font': 'cinzel-700.ttf', 'size': 136, 'y': 175, 'color': SILVER, 'track': 6, 'glow': (150, 170, 255), 'glow_r': 26}])
T_t2 = glow_layer([{'text': C['title2'], 'font': 'Inter-ExtraBold.otf', 'size': 84, 'y': 340, 'color': WHITE, 'track': 24, 'glow': VIOLET, 'glow_r': 22}])
AX, BX = 300, 780
def name_tag(name, x, col):
    return glow_layer([{'text': '◉ IDENTIFIED', 'font': 'JetBrainsMono-Bold.ttf', 'size': 30, 'y': 690, 'x': x, 'color': CYAN, 'track': 2},
                       {'text': name, 'font': 'Inter-ExtraBold.otf', 'size': 60, 'y': 730, 'x': x, 'color': WHITE, 'track': 3, 'glow': col, 'glow_r': 20}])
T_n1, T_n2 = name_tag(C['artists'][0], AX, VIOLET), name_tag(C['artists'][1], BX, PINK)
def big_name(name, col, role):
    return glow_layer([{'text': role, 'font': 'JetBrainsMono-Bold.ttf', 'size': 34, 'y': 205, 'color': CYAN, 'track': 3},
                       {'text': name, 'font': 'Inter-ExtraBold.otf', 'size': 130, 'y': 250, 'color': WHITE, 'track': 4, 'glow': col, 'glow_r': 30}])
T_bigA, T_bigB = big_name(C['artists'][0], VIOLET, '◉ ON THE DECKS'), big_name(C['artists'][1], PINK, '◉ ON THE DECKS')
STORY = glow_layer([{'text': C['story'][0], 'font': 'corm-500i.ttf', 'size': 80, 'y': 860, 'color': CREAM},
                    {'text': C['story'][1], 'font': 'cinzel-700.ttf', 'size': 60, 'y': 965, 'color': GOLD, 'track': 4},
                    {'text': C['story'][2], 'font': 'corm-500i.ttf', 'size': 80, 'y': 1050, 'color': CREAM}])
T_moon = glow_layer([{'text': C['moon_line'], 'font': 'Inter-ExtraBold.otf', 'size': 66, 'y': 760, 'color': WHITE, 'track': 3, 'glow': LIME, 'glow_r': 18}])
pm = Image.new('L', (W, H), 0); ImageDraw.Draw(pm).rounded_rectangle([60, 440, W - 60, 830], 50, fill=200)
INFO_PLATE = Image.new('RGBA', (W, H), (8, 6, 22, 0)); INFO_PLATE.putalpha(pm.filter(ImageFilter.GaussianBlur(40)))
T_date = glow_layer([{'text': C['date_line'], 'font': 'Inter-ExtraBold.otf', 'size': 80, 'y': 480, 'color': WHITE, 'track': 4, 'glow': VIOLET, 'glow_r': 20}])
T_time = glow_layer([{'text': C['time_line'], 'font': 'Inter-ExtraBold.otf', 'size': 100, 'y': 590, 'color': WHITE, 'track': 5, 'glow': CYAN, 'glow_r': 26}])
T_seats = glow_layer([{'text': C['seats_line'], 'font': 'cinzel-700.ttf', 'size': 42, 'y': 740, 'color': GOLD, 'track': 5}])
E_hook1 = glow_layer([{'text': C['end_hook'][0], 'font': 'Inter-ExtraBold.otf', 'size': 64, 'y': 175, 'color': WHITE, 'track': 3, 'glow': LIME, 'glow_r': 16}])
E_hook2 = glow_layer([{'text': C['end_hook'][1], 'font': 'cinzel-700.ttf', 'size': 96, 'y': 262, 'color': GOLD, 'track': 6, 'glow': (255, 200, 120), 'glow_r': 18}])
E_title = glow_layer([{'text': C['title1'] + ' ' + C['title2'], 'font': 'cinzel-700.ttf', 'size': 64, 'y': 850, 'color': SILVER, 'track': 4, 'glow': (150, 170, 255), 'glow_r': 16}])
E_art = glow_layer([{'text': C['artists'][0] + '  &  ' + C['artists'][1], 'font': 'Inter-ExtraBold.otf', 'size': 60, 'y': 940, 'color': WHITE, 'track': 3, 'glow': VIOLET, 'glow_r': 18}])
E_when = glow_layer([{'text': C['end_when'], 'font': 'Inter-Bold.otf', 'size': 50, 'y': 1028, 'color': CYAN, 'track': 3}])
E_book = glow_layer([{'text': C['seats_line'], 'font': 'cinzel-700.ttf', 'size': 38, 'y': 1130, 'color': GOLD, 'track': 5},
                     {'text': C['booking_label'], 'font': 'Inter-SemiBold.otf', 'size': 44, 'y': 1192, 'color': SILVER},
                     {'text': C['phone'], 'font': 'Inter-ExtraBold.otf', 'size': 80, 'y': 1250, 'color': WHITE, 'track': 2, 'glow': (120, 140, 255), 'glow_r': 14}])
E_tag = glow_layer([{'text': C['place'], 'font': 'cinzel-700.ttf', 'size': 36, 'y': 1375, 'color': GOLD, 'track': 7},
                    {'text': C['tagline'], 'font': 'corm-500i.ttf', 'size': 62, 'y': 1432, 'color': CREAM}])
RULE = Image.new('RGBA', (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(RULE)
dr.line([(cx - 240, 1104), (cx - 14, 1104)], fill=GOLD, width=2); dr.line([(cx + 14, 1104), (cx + 240, 1104)], fill=GOLD, width=2); dr.polygon([(cx, 1097), (cx + 7, 1104), (cx, 1111), (cx - 7, 1104)], fill=GOLD)
LOGO_E = logo.resize((int(logo.width * 400 / logo.height), 400), Image.LANCZOS)
pm2 = Image.new('L', (W, H), 0); ImageDraw.Draw(pm2).rounded_rectangle([40, 810, W - 40, 1510], 40, fill=170)
E_PLATE = Image.new('RGBA', (W, H), (6, 6, 18, 0)); E_PLATE.putalpha(pm2.filter(ImageFilter.GaussianBlur(40)))

# cockpit interior (static parts)
def cockpit_static():
    im = Image.fromarray(vgrad((26, 28, 44), (8, 8, 16), H, W).astype(np.uint8)).convert('RGBA'); d = ImageDraw.Draw(im)
    for y in range(80, 1100, 120): d.line([(0, y), (40, y)], fill=(50, 54, 80), width=3); d.line([(W - 40, y), (W, y)], fill=(50, 54, 80), width=3)
    win = Image.new('L', (W, H), 0); ImageDraw.Draw(win).rounded_rectangle([60, 150, W - 60, 1010], 90, fill=255)
    frame = Image.new('RGBA', (W, H), (0, 0, 0, 0)); ImageDraw.Draw(frame).rounded_rectangle([48, 138, W - 48, 1022], 100, outline=(80, 86, 120), width=22)
    for a in np.linspace(0, 2 * math.pi, 28, endpoint=False):
        px, py = cx + 505 * math.cos(a), 580 + 450 * math.sin(a)
    console = Image.new('RGBA', (W, H), (0, 0, 0, 0)); cm = Image.new('L', (W, H), 0)
    ImageDraw.Draw(cm).polygon([(0, 1500), (W, 1500), (W, H), (0, H)], fill=255); ImageDraw.Draw(cm).ellipse([-200, 1420, W + 200, 1600], fill=255)
    console = fill_grad(cm, (46, 50, 74), (14, 14, 24))
    return im, win, frame, console
CK_BG, CK_WIN, CK_FRAME, CK_CONSOLE = cockpit_static()
BTN = [(rng.uniform(80, 1000), rng.uniform(1560, 1820), rng.choice([0, 1, 2, 3]), rng.uniform(0, 6)) for _ in range(46)]
def cockpit(t):
    fr = CK_BG.copy()
    space = NEB.copy(); stars(space, t * 0.5, H)
    mx = 900 - 60 * (t - T_COCK); space.alpha_composite(MOON_WIN, (int(mx - MOON_WIN.width / 2), int(520 - MOON_WIN.height / 2)))
    fr = Image.composite(space, fr, CK_WIN); fr.alpha_composite(CK_FRAME)
    hud_frame(fr, env(t, 0.2, T_EXT, 0.3, 0.3), (110, 200, 970, 960))
    typewriter(fr, [(T_COCK + 0.4 + 0.8 * i, s) for i, s in enumerate(C['hud'])], t)
    if t > T_COCK + 0.3:                                       # signal waveform
        L = Image.new('RGBA', (W, H), (0, 0, 0, 0)); dl = ImageDraw.Draw(L)
        pts = [(140 + i * 8, 560 + 34 * math.sin(i * 0.33 + t * 14) * (0.3 + 0.7 * abs(math.sin(t * 2.4 + i * 0.05)))) for i in range(100)]
        dl.line(pts, fill=CYAN + (220,), width=3); fr.alpha_composite(glow(L, 6, 1.5)); fr.alpha_composite(L)
    # the two pilots, big, from behind the console
    lean = 3 * math.sin(t * 1.3)
    talkA = any(a <= t < b for a, b, txt, who in C['cockpit_lines'] if who == 'A')
    talkB = any(a <= t < b for a, b, txt, who in C['cockpit_lines'] if who == 'B')
    place_alien(fr, 'A', 290, 700, 0.86, t, talk=talkA, tilt=lean + (6 if talkB else 0), glow_ant=0.3 + 0.3 * math.sin(t * 4), ph=0.0)
    place_alien(fr, 'B', 800, 712, 0.86, t, talk=talkB, tilt=-lean - (6 if talkA else 0), ph=1.4)
    fr.alpha_composite(CK_CONSOLE); d = ImageDraw.Draw(fr); cols = [CYAN, PINK, LIME, GOLD]
    for (bx, by, ci, ph) in BTN:
        on = math.sin(t * 5 + ph) > 0.2; c = cols[ci] if on else (50, 54, 70); d.rounded_rectangle([bx - 14, by - 8, bx + 14, by + 8], 5, fill=c)
    for a, b, txt, who in C['cockpit_lines']:
        bubble(fr, txt, 330 if who == 'A' else 750, 600, (300 if who == 'A' else 780, 700), t, a, b, who)
    return fr

def rooftop(t, party, ufo_xy=None, ufo_s=1.0, tilt=0.0, aliens=0.0, beam=0.0, look_up=False, talkA=False, moon_pulse=False):
    fr = SKY.copy(); stars(fr, t); k = kick_env(t)
    ms = 1.0 + (0.22 * k if moon_pulse else 0)
    mo = MOON_ROOF if ms == 1.0 else MOON_ROOF.resize((int(MOON_ROOF.width * ms), int(MOON_ROOF.height * ms)), Image.BICUBIC)
    fr.alpha_composite(mo, (850 - mo.width // 2, 500 - mo.height // 2))
    fr.alpha_composite(BACK, (-int(60 + 2 * t % 60), 0)); fr.alpha_composite(MID, (-int(40 + 3 * t % 40), 0)); fr.alpha_composite(NEAR, (-int(20 + 4 * t % 20), 0))
    if party: fr = screen(fr, lasers(t, (0.6 + 0.4 * k) * (1 - smooth((t - BRK + 0.3) / 0.3)), LCOL1 if t < T_INFO else LCOL2))
    fr.alpha_composite(LIGHTS)
    if ufo_xy: draw_ufo(fr, ufo_xy[0], ufo_xy[1], ufo_s, t, tilt, k)
    if beam > 0:
        ux, uy = ufo_xy; B_ = Image.new('RGBA', (W, H), (0, 0, 0, 0)); dbm = ImageDraw.Draw(B_)
        dbm.polygon([(ux - 40, uy + 30), (ux + 40, uy + 30), (960, 1350), (120, 1350)], fill=(120, 255, 220, int(90 * beam)))
        for i in range(12):
            yy = uy + 30 + ((t * 300 + i * 60) % (1350 - uy - 30)); wfr = (yy - uy - 30) / (1350 - uy - 30); hw = 40 + 380 * wfr
            dbm.line([(cx - hw, yy), (cx + hw, yy)], fill=(200, 255, 240, int(120 * beam)), width=3)
        fr.alpha_composite(glow(B_, 10, 1.2)); fr.alpha_composite(B_)
    if aliens > 0:
        for key, x, ph in (('A', AX, 0.0), ('B', BX, 1.2)):
            bob = (14 * k if party else 3 * math.sin(t * 3 + ph)) * (1 if key == 'A' else 1.2)
            x0 = place_alien(fr, key, x, 830 + bob, 0.7, t, talk=(talkA and key == 'A'), tilt=(4 * k if party and key == 'A' else 0), alpha=aliens, glow_ant=k, ph=ph)
            if party and key == 'B' and DROP + 1.5 < t < T_INFO + 2:
                up = smooth((t - DROP - 1.5) / 0.4) * (1 - smooth((t - T_INFO - 1.5) / 0.4)); d = ImageDraw.Draw(fr)
                sx_, sy_ = x0 + 350, 830 + bob + 500; ex, ey = sx_ + 70, sy_ - 320 * up - 12 * k
                d.line([(sx_, sy_), (ex, ey)], fill=(110, 80, 200), width=46); d.ellipse([ex - 34, ey - 40, ex + 34, ey + 30], fill=(170, 140, 250))
    fr.alpha_composite(BOOTH)
    if party:
        L = Image.new('RGBA', (W, H), (0, 0, 0, 0)); ImageDraw.Draw(L).line([(155, 1402), (925, 1402)], fill=(CYAN if t < T_INFO else LIME) + (255,), width=6)
        fr.alpha_composite(glow(L, 10 + 8 * k, 2.0)); fr.alpha_composite(L)
    draw_decks(fr, t, party); draw_crowd(fr, t, party, look_up)
    return fr

def closeup(t, key):
    """one DJ, big, at the decks with lasers behind"""
    k = kick_env(t); fr = SKY.copy(); stars(fr, t)
    fr.alpha_composite(BACK, (-60, 120)); fr = screen(fr, lasers(t, 0.65 + 0.35 * k, LCOL1, oy=1500)); fr.alpha_composite(LIGHTS, (0, -200))
    bob = 18 * k; sc = 1.28
    x0 = place_alien(fr, key, cx, 560 + bob, sc, t, tilt=(5 * k if key == 'A' else -3 * k), glow_ant=k, ph=0.5)
    d = ImageDraw.Draw(fr)
    if key == 'B':                                           # both hands up
        for s in (-1, 1):
            sx_, sy_ = cx + s * 240, 1340 + bob; ex, ey = cx + s * 400, 760 - 30 * k
            d.line([(sx_, sy_), (ex, ey)], fill=(110, 80, 200), width=70); d.ellipse([ex - 50, ey - 58, ex + 50, ey + 44], fill=(170, 140, 250))
    dk = Image.new('RGBA', (W, H), (0, 0, 0, 0)); dd = ImageDraw.Draw(dk)
    dd.rounded_rectangle([-40, 1470, W + 40, H + 40], 40, fill=(30, 32, 48)); dd.line([(0, 1474), (W, 1474)], fill=(CYAN if key == 'A' else PINK), width=6)
    fr.alpha_composite(glow(dk.crop((0, 1460, W, 1490)).resize((W, 30)), 6, 1.5) if False else dk)
    plat = (cx, 1640); dd = ImageDraw.Draw(fr)
    dd.ellipse([plat[0] - 360, plat[1] - 110, plat[0] + 360, plat[1] + 110], fill=(12, 12, 18), outline=(100, 105, 140), width=4)
    for r in range(80, 350, 22): dd.ellipse([plat[0] - r, plat[1] - r * 0.3, plat[0] + r, plat[1] + r * 0.3], outline=(28, 28, 38), width=2)
    lab = logo.resize((int(logo.width * 120 / logo.height), 120), Image.LANCZOS).rotate(-(t * 200) % 360, resample=Image.BICUBIC)
    lab = lab.resize((lab.width, int(lab.height * 0.32)), Image.BICUBIC); fr.alpha_composite(lab, (plat[0] - lab.width // 2, plat[1] - lab.height // 2))
    if key == 'A':                                           # scratching hand
        hx = plat[0] + 170 + 70 * math.sin(t * 2 * math.pi * 2); hy = plat[1] - 40
        dd.line([(cx + 230, 1390 + bob), (hx, hy)], fill=(30, 110, 110), width=64); dd.ellipse([hx - 56, hy - 40, hx + 56, hy + 40], fill=(110, 220, 190))
        for f_ in range(3): dd.ellipse([hx - 60 + f_ * 30, hy + 20, hx - 36 + f_ * 30, hy + 52], fill=(110, 220, 190))
    return fr

def hyper(t):
    p = (t - T_HYPER) / (T_ARR - T_HYPER); fr = Image.new('RGBA', (W, H), (4, 4, 14, 255)); d = ImageDraw.Draw(fr)
    for i in range(NST):
        dx, dy = SX[i] - cx, SY[i] - 960; L = 1 + 600 * p ** 2; n_ = math.hypot(dx, dy) + 1e-6
        x0, y0 = cx + dx * (1 + p), 960 + dy * (1 + p); d.line([(x0, y0), (x0 + dx / n_ * L, y0 + dy / n_ * L)], fill=(200, 220, 255), width=2)
    draw_ufo(fr, cx, 960, max(0.05, 1.2 * (1 - p) ** 2), t)
    return fr

# ============================ audio ============================
m = make_edm(TOTAL, DROP, BRK, seed=C.get('seed', 5), bpm=BPM, pre=T_ARR); N = len(m); sfx = np.zeros(N)
lpf = lambda x, f: sosfilt(butter(2, f, 'low', fs=SR, output='sos'), x)
bpf = lambda x, a, b: sosfilt(butter(2, [a, b], 'band', fs=SR, output='sos'), x)
def addsfx(sig, t0, g=1.0):
    i = int(t0 * SR); j = min(N, i + len(sig))
    if 0 <= i < N: sfx[i:j] += sig[:j - i] * g
def bleep(f, dur=0.07):
    n = int(dur * SR); tt = np.arange(n) / SR; return np.sin(2 * np.pi * f * tt) * np.minimum(1, tt / 0.004) * np.clip((dur - tt) / 0.01, 0, 1) * 0.18
def chatter(dur, base_f, seed):
    """alien gibberish: short pitched syllables with vibrato through a vocal-ish band-pass"""
    r = np.random.default_rng(seed); out = np.zeros(int(dur * SR)); t0 = 0.0
    while t0 < dur - 0.08:
        sl = r.uniform(0.06, 0.13); n = int(sl * SR); tt = np.arange(n) / SR; f0 = base_f * r.uniform(0.8, 1.5)
        f = f0 * (1 + 0.25 * np.sin(2 * np.pi * r.uniform(10, 25) * tt)) * (1 + r.uniform(-0.3, 0.3) * tt / sl)
        syl = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.sin(np.pi * tt / sl) ** 2
        i = int(t0 * SR); out[i:i + n] += syl[:len(out) - i]; t0 += sl + r.uniform(0.01, 0.05)
    return bpf(out, 400, 3500) * 0.12
for i in range(len(C['hud'])): addsfx(bleep(1500 + 300 * (i % 2)), T_COCK + 0.4 + 0.8 * i); addsfx(bleep(2200), T_COCK + 0.49 + 0.8 * i)
for j, (a, b, txt, who) in enumerate(C['cockpit_lines'] + C['later_lines']):
    addsfx(chatter(min(1.2, b - a - 0.2), 520 if who == 'A' else (330 if who == 'B' else 700), j), a + 0.05)
for i in range(2): addsfx(bleep(1800 + 400 * i), T_EXT + 0.4 + 0.15 * i)
n = int((T_ARR - T_HYPER) * SR); tt = np.arange(n) / SR                                     # hyperspace whoosh
addsfx(bpf(np.random.default_rng(1).standard_normal(n), 300, 6000) * (tt / tt[-1]) ** 2 * 0.3 + np.sin(2 * np.pi * np.cumsum(80 * 2 ** (3 * tt / tt[-1])) / SR) * (tt / tt[-1]) ** 2 * 0.1, T_HYPER)
n = int(1.2 * SR); tt = np.arange(n) / SR; addsfx(np.sin(2 * np.pi * np.cumsum(50 + 60 * np.exp(-tt * 5)) / SR) * np.exp(-tt * 3) * 0.35, T_ARR)
n = int((T_LAND - T_ARR) * SR); tt = np.arange(n) / SR; f = 95 - 25 * tt / tt[-1]
addsfx((np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.4 * np.sin(4 * np.pi * np.cumsum(f) / SR)) * (0.5 + 0.5 * np.sin(2 * np.pi * 7 * tt)) * np.minimum(1, tt / 0.5) * 0.08, T_ARR)
n = int((DROP - T_LAND - 1.0) * SR); tt = np.arange(n) / SR; fb = 220 * 2 ** (2 * tt / tt[-1])
addsfx((np.sin(2 * np.pi * np.cumsum(fb) / SR) * (0.6 + 0.4 * np.sin(2 * np.pi * 14 * tt)) * 0.07 + bpf(np.random.default_rng(2).standard_normal(n), 3000, 9000) * 0.04) * np.minimum(1, tt / 0.3), T_LAND + 0.3)
for i, f0 in enumerate((1568, 2093, 2637, 3136)): addsfx(bleep(f0, 0.18) * 0.8, TOTAL - 3.0 + i * 0.12)
mix = m * 0.9 + np.stack([sfx, sfx], 1)
aw = os.path.splitext(P(C['out']))[0] + '_audio.wav'; os.makedirs(os.path.dirname(aw), exist_ok=True)
mix /= np.abs(mix).max() / 0.95; wavfile.write(aw, SR, (mix * 32767).astype(np.int16))
st = subprocess.run(['ffmpeg', '-hide_banner', '-i', aw, '-af', 'ebur128=peak=true', '-f', 'null', '-'], capture_output=True, text=True).stderr
I = float(re.findall(r'I:\s+(-?[\d.]+) LUFS', st)[-1]); pk = float(re.findall(r'Peak:\s+(-?[\d.]+) dBFS', st)[-1])
gn = min(-14 - I, -1.0 - pk); wavfile.write(aw, SR, (np.clip(mix * 10 ** (gn / 20), -1, 1) * 32767).astype(np.int16))

# ============================ cinematic post ============================
yy_, xx_ = np.mgrid[0:H, 0:W]; rr_ = np.sqrt(((xx_ - W / 2) / W) ** 2 + ((yy_ - H / 2) / H) ** 2)
VIG = np.clip(1.08 - 0.75 * rr_ ** 2, 0, 1).astype(np.float32)[..., None]
GRAIN = [np.random.default_rng(i).normal(0, 1, (H // 2, W // 2)).astype(np.float32) for i in range(8)]
PART = dict(x=rng.uniform(0, W, 140), y=rng.uniform(0, H, 140), vx=rng.uniform(-12, 12, 140), vy=rng.uniform(-60, -15, 140), r=rng.uniform(1.5, 4.5, 140), ph=rng.uniform(0, 6.3, 140))
def particles(fr, t, amt):
    if amt <= 0: return
    L = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(L); cols = [CYAN, PINK, (255, 230, 180), LIME]
    for i in range(140):
        x = (PART['x'][i] + PART['vx'][i] * t) % W; y = (PART['y'][i] + PART['vy'][i] * t) % H; r = PART['r'][i]
        a = int(255 * amt * (0.4 + 0.6 * abs(math.sin(t * 2 + PART['ph'][i]))))
        d.ellipse([x - r, y - r, x + r, y + r], fill=cols[i % 4] + (a,))
    fr.alpha_composite(glow(L, 5, 2.0)); fr.alpha_composite(L)
def post(fr, t, shake=0.0):
    a = np.asarray(fr.convert('RGB')).astype(np.float32) / 255
    sm = fr.convert('RGB').resize((W // 4, H // 4), Image.BILINEAR); sa = np.asarray(sm).astype(np.float32) / 255
    lum = sa.max(2, keepdims=True); sat = lum - sa.min(2, keepdims=True)
    br = np.clip((lum - 0.9) / 0.1, 0, 1) ** 2 * sa * (0.35 + 0.65 * np.clip(sat * 2.5, 0, 1))   # coloured lights bloom, white text doesn't
    bi = Image.fromarray((br * 255).astype(np.uint8))
    b1 = np.asarray(bi.filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255
    b2 = np.asarray(bi.filter(ImageFilter.GaussianBlur(22)).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255
    an = np.asarray(bi.resize((12, H // 4), Image.BILINEAR).resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2)).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255
    a = a + 0.4 * b1 + 0.35 * b2 + 0.6 * an.mean(2, keepdims=True) * np.array([0.45, 0.75, 1.0], np.float32)
    lum = a.mean(2, keepdims=True)                                            # teal shadows / warm highlights
    a = a + (1 - np.clip(lum * 1.6, 0, 1)) * np.array([-0.02, 0.012, 0.03], np.float32) + np.clip(lum - 0.6, 0, 1) * np.array([0.05, 0.02, -0.03], np.float32)
    a = a * VIG
    a[..., 0] = np.roll(a[..., 0], 2, axis=1); a[..., 2] = np.roll(a[..., 2], -2, axis=1)   # chromatic aberration
    g = GRAIN[int(t * FPS) % 8]; g = np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None]; a = a + g * 0.022
    out = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    if shake > 0.01:
        r_ = np.random.default_rng(int(t * 1000)); dx, dy = r_.uniform(-1, 1, 2) * 14 * shake
        z = 1.02; big = out.resize((int(W * z), int(H * z)), Image.BILINEAR)
        out = big.crop((int((big.width - W) / 2 + dx), int((big.height - H) / 2 + dy), int((big.width - W) / 2 + dx) + W, int((big.height - H) / 2 + dy) + H))
    return out

# ============================ render ============================
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-i', aw, '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-maxrate', '12M', '-bufsize', '24M', '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', P(C['out'])], stdin=subprocess.PIPE)
LL = C['later_lines']
for n in range(int(round(TOTAL * FPS))):
    t = n / FPS; k = kick_env(t)
    if t < T_EXT: fr = cockpit(t)
    elif t < T_HYPER:                                                    # ship flies past the moon, course set
        fr = NEB.copy(); stars(fr, t, 1920); fr.alpha_composite(MOON_BIG, (int(860 + 30 * t - MOON_BIG.width / 2), int(1450 - MOON_BIG.height / 2)))
        p = (t - T_EXT) / (T_HYPER - T_EXT); ux, uy = -150 + 820 * smooth(p), 900 + 60 * math.sin(t * 2)
        draw_ufo(fr, ux, uy, 1.4, t); hud_frame(fr, 1, (60, 170, 1020, 1700))
        typewriter(fr, [(T_EXT + 0.3 + 0.9 * i, s) for i, s in enumerate(C['decode'])], t, y0=240)
        for a, b, txt, who in LL:
            if who == 'ship': bubble(fr, txt, max(330, min(750, ux)), uy - 210, (ux, uy - 70), t, a, b, 'B')
    elif t < T_ARR: fr = hyper(t)
    elif t < T_LAND:                                                     # arrival over the Himalayas
        p = (t - T_ARR) / (T_LAND - T_ARR); e = smooth(p)
        ux, uy = 980 - (980 - cx) * e, 120 + (640 - 120) * e - 120 * math.sin(math.pi * e) * 0.6
        fr = rooftop(t, False, (ux, uy), 0.3 + 0.7 * e, tilt=18 * (1 - e), look_up=p > 0.55)
        put(fr, STORY, env(t, T_ARR + 0.4, T_LAND - 1.6, 0.5, 0.4), dy=int(20 * (1 - ease((t - T_ARR - 0.4) / 0.6))))
        for a, b, txt, who in LL:
            if who == 'crowd': bubble(fr, txt, 330, 1230, (70, 1745), t, a, b, 'crowd')
    elif t < DROP:                                                       # landing
        p = (t - T_LAND) / (DROP - T_LAND)
        talkA = any(a <= t < b for a, b, txt, who in LL if who == 'A')
        fr = rooftop(t, False, (cx, 640 + 6 * math.sin(t * 3)), 1.0, beam=min(1, p * 3) * (1 - smooth((p - 0.8) / 0.2)), aliens=smooth((p - 0.2) / 0.45), look_up=True, talkA=talkA)
        typewriter(fr, [(T_LAND + 0.1, C['landing'])], t, x=90, y0=230)
        for a, b, txt, who in LL:
            if who == 'A': bubble(fr, txt, 420, 700, (320, 840), t, a, b, 'A')
    elif t < T_CUA or (T_WIDE <= t < BRK):                               # wide party / moon / info
        e = smooth((t - DROP) / 1.0); ux, uy = cx + (210 - cx) * e, 640 + (470 - 640) * e + 8 * math.sin(t * 2)
        fr = rooftop(t, True, (ux, uy), 1.0 - 0.35 * e, aliens=1.0, moon_pulse=(T_WIDE <= t < T_INFO))
        put(fr, T_t1, 1 if t > DROP + 0.4 else env(t, DROP, BRK, 0.4), dy=int(-30 * (1 - ease((t - DROP) / 0.5)))); glitch(fr, T_t2, t, DROP + 0.1)
        if k > 0.05 and t > DROP + 0.5: put(fr, T_t2, 0.35 * k)
        if t < T_CUA:
            for x, L, t0, col in ((AX, T_n1, DROP + 0.6, VIOLET), (BX, T_n2, DROP + 1.4, PINK)):
                a = env(t, t0, T_CUA, 0.2, 0.3); brackets(fr, x - 205, 840, x + 205, 1320, a, col)
                if t < T_CUA - 0.3: glitch(fr, L, t, t0)
                else: put(fr, L, a)
        elif t < T_INFO: glitch(fr, T_moon, t, T_WIDE + 0.2)
        else:
            put(fr, INFO_PLATE, env(t, T_INFO, BRK, 0.3, 0.3))
            if t < BRK - 0.3: glitch(fr, T_date, t, T_INFO + 0.05); glitch(fr, T_time, t, T_INFO + 0.5)
            else: put(fr, T_date, 1); put(fr, T_time, 1)
            put(fr, T_seats, env(t, T_INFO + 1.2, BRK, 0.4, 0.3))
    elif t < T_CUB:
        fr = closeup(t, 'A'); glitch(fr, T_bigA, t, T_CUA + 0.05)
    elif t < T_WIDE:
        fr = closeup(t, 'B'); glitch(fr, T_bigB, t, T_CUB + 0.05)
    else:                                                                # punchline + booking
        lt = t - BRK; fr = SKY.copy(); stars(fr, t, 1920)
        fr.alpha_composite(MOON_END, (cx - MOON_END.width // 2, 610 - MOON_END.height // 2))
        a = env(lt, 0.5, 99, 0.8); lg = LOGO_E.copy(); lg.putalpha(lg.split()[3].point(lambda v: int(v * a))); fr.alpha_composite(lg, (cx - lg.width // 2, 610 - lg.height // 2))
        glitch(fr, E_hook1, t, BRK + 0.15); put(fr, E_hook2, env(lt, 0.9, 99, 0.3), dy=int(30 * (1 - ease((lt - 0.9) / 0.35))))
        put(fr, E_PLATE, env(lt, 1.0, 99, 0.6)); put(fr, E_title, env(lt, 1.4, 99, 0.4)); put(fr, E_art, env(lt, 1.7, 99, 0.4))
        put(fr, E_when, env(lt, 2.0, 99, 0.4)); put(fr, RULE, env(lt, 2.2, 99, 0.4)); put(fr, E_book, env(lt, 2.5, 99, 0.5)); put(fr, E_tag, env(lt, 3.2, 99, 0.6))
        for key, x, ph in (('A', 150, 0.0), ('B', 930, 1.3)):
            up = smooth((lt - 3.6) / 0.6)
            if up > 0: place_alien(fr, key, x, 1920 - 400 * up, 0.47, t, tilt=8 * math.sin(lt * 4 + ph), glow_ant=-1, ph=ph)
        if TOTAL - t < 3.2:
            p = 1 - (TOTAL - t) / 3.2; draw_ufo(fr, -150 + 1400 * smooth(p), 120 + 30 * math.sin(p * 6), 0.45, t)
    fl = 0.10 * k if DROP <= t < BRK else 0.0
    for tt_, s in ((T_ARR, 0.85), (DROP, 0.8), (T_CUA, 0.35), (T_CUB, 0.35), (T_WIDE, 0.35), (BRK, 0.6)):
        if 0 <= t - tt_ < 0.3: fl = max(fl, s * math.exp(-(t - tt_) / 0.08))
    if DROP <= t < BRK: particles(fr, t, 0.8)
    elif T_LAND <= t < DROP: particles(fr, t, 0.35 * (t - T_LAND) / (DROP - T_LAND))
    if fl > 0.01: fr.alpha_composite(Image.new('RGBA', (W, H), (230, 230, 255, int(255 * fl))))
    sh_ = (0.9 * kick_env(t, 0.1) if DROP <= t < BRK else 0) + (max(0, 1 - (t - T_ARR) / 0.6) if T_ARR <= t < T_ARR + 0.6 else 0) + (max(0, 1 - (t - DROP) / 0.8) * 1.5 if DROP <= t < DROP + 0.8 else 0)
    fr = post(fr, t, sh_).convert('RGBA')
    if TOTAL - t < 0.35: fr = Image.blend(fr.convert('RGB'), Image.new('RGB', (W, H), 0), 1 - (TOTAL - t) / 0.35).convert('RGBA')
    enc.stdin.write(fr.convert('RGB').tobytes())
enc.stdin.close(); enc.wait(); print(f'done {P(C["out"])} {TOTAL:.1f}s (audio {I:.1f} LUFS -> {gn:+.1f} dB)')
