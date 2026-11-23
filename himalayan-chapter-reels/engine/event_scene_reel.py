"""Event promo built entirely from code-drawn visuals: "The Moon Drops the Beat".
Moonrise over Himalayan ranges -> on the drop the moon becomes a spinning vinyl with the cafe logo
as its label, lasers sweep over the mountains, a rooftop crowd jumps on the kick, names glitch in,
then a booking card under the moon. Music: music_edm.make_edm (drop + breakdown synced).
usage: python engine/event_scene_reel.py episodes/event_fullmoon_2411_v2.json
"""
import json, math, os, re, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from music_edm import make_edm, SR

C = json.load(open(sys.argv[1])); base = os.path.dirname(os.path.abspath(sys.argv[1]))
P = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(base, x))
W, H, FPS = 1080, 1920, 30
TOTAL, DROP, BRK = C['total'], C['drop'], C['break']
BPM = C.get('bpm', 120); BEAT = 60 / BPM
SILVER=(236,240,255); VIOLET=(176,120,255); CYAN=(120,225,255); GOLD=(226,186,104); CREAM=(247,240,222); WHITE=(255,255,255); GREEN=(120,255,190)
FD = P(C['assets']['fonts']); INTER = '/usr/share/fonts/opentype/inter/'
F = lambda n, s: ImageFont.truetype(os.path.join(INTER, n) if n.startswith('Inter') else os.path.join(FD, n), s)
cx = W // 2
ease = lambda p: 1 - (1 - min(max(p, 0), 1)) ** 3
def smooth(p): p = min(max(p, 0), 1); return p * p * (3 - 2 * p)
def env(t, a, b, fi=0.35, fo=0.3): return float(np.clip(min((t - a) / fi, (b - t) / fo), 0, 1))
rng = np.random.default_rng(21)
KICKS = np.arange(DROP, BRK - 1e-6, BEAT)
def kick_env(t, tau=0.12):
    if t < DROP or t >= BRK: return 0.0
    k = KICKS[KICKS <= t + 1e-9]; return math.exp(-(t - k[-1]) / tau) if len(k) else 0.0

# ---------------- static layers ----------------
def vgrad(top, bot, h=H, w=W):
    y = np.linspace(0, 1, h)[:, None, None]
    return (np.array(top, np.float32) * (1 - y) + np.array(bot, np.float32) * y) * np.ones((1, w, 1), np.float32)
SKY = Image.fromarray(vgrad((10, 12, 40), (30, 22, 62)).astype(np.uint8))
def ridge(w, base_y, amp, rough, seed, n=1025):
    r = np.random.default_rng(seed); y = np.zeros(n); y[0], y[-1] = r.uniform(-1, 1, 2) * amp
    step = n - 1; a = amp
    while step > 1:
        half = step // 2
        for i in range(half, n - 1, step): y[i] = (y[i - half] + y[i + half]) / 2 + r.uniform(-a, a)
        step = half; a *= rough
    xs = np.linspace(0, w, n); return np.interp(np.arange(w), xs, y) + base_y
def mountain(w, base_y, amp, rough, seed, top_col, bot_col, rim=None, snow=False, valley=0.0):
    y = ridge(w, base_y, amp, rough, seed)
    if valley: y += valley * np.exp(-((np.arange(w) - w / 2) / (w * 0.18)) ** 2)   # dip in the middle for the moon
    img = Image.fromarray(vgrad(top_col, bot_col, H, w).astype(np.uint8)).convert('RGBA')
    m = Image.new('L', (w, H), 0); ImageDraw.Draw(m).polygon([(0, H)] + list(zip(range(w), y)) + [(w, H)], fill=255)
    if snow:
        sn = Image.new('L', (w, H), 0); ds = ImageDraw.Draw(sn)
        for x in range(0, w, 2):
            depth = max(0, (base_y - 150 - y[x]) * 0.55)
            if depth > 4: ds.line([(x, y[x]), (x, y[x] + depth)], fill=int(min(200, 40 + depth * 2)), width=2)
        sn = sn.filter(ImageFilter.GaussianBlur(3))
        img = Image.composite(Image.new('RGBA', (w, H), (196, 206, 238, 255)), img, sn)
    img.putalpha(m)
    if rim:
        rl = Image.new('RGBA', (w, H), (0, 0, 0, 0)); ImageDraw.Draw(rl).line(list(zip(range(w), y)), fill=rim, width=3)
        img.alpha_composite(rl.filter(ImageFilter.GaussianBlur(1.5)))
    return img, y
BACK, _ = mountain(W + 120, 1080, 430, 0.5, 3, (52, 60, 104), (24, 26, 56), rim=(170, 180, 230, 200), snow=True, valley=170)
MID, _ = mountain(W + 80, 1300, 140, 0.55, 8, (26, 30, 62), (14, 16, 36), rim=(90, 96, 160, 150))
NEAR, ny = mountain(W + 40, 1510, 70, 0.6, 13, (12, 14, 30), (6, 7, 16))
d = ImageDraw.Draw(NEAR)                                     # pine silhouettes along the near ridge
for x in range(-10, W + 40, 18):
    hh = rng.uniform(60, 150); bx = x + rng.uniform(-6, 6); by = ny[min(max(int(bx), 0), W + 39)] + 8
    for k in range(5):
        wdt = hh * (0.36 - 0.06 * k); yy = by - hh * k / 5
        d.polygon([(bx - wdt, yy), (bx + wdt, yy), (bx, yy - hh * 0.38)], fill=(6, 7, 16, 255))
def fog_tex():
    n = rng.random((40, 30)); im = Image.fromarray((n * 255).astype(np.uint8)).resize((W * 2, 600), Image.BICUBIC).filter(ImageFilter.GaussianBlur(30))
    a = np.asarray(im).astype(np.float32) / 255; a = np.clip((a - 0.35) * 1.6, 0, 1) * 70
    yy = np.linspace(0, 1, 600)[:, None]; a *= np.sin(np.pi * yy)
    out = np.dstack([np.full_like(a, 150), np.full_like(a, 160), np.full_like(a, 220), a]).astype(np.uint8)
    return Image.fromarray(out, 'RGBA')
FOG = fog_tex()
NST = 380; SX, SY = rng.uniform(0, W, NST), rng.uniform(0, 1250, NST); SRAD = rng.uniform(0.6, 2.3, NST); SPH = rng.uniform(0, 6.3, NST); SFQ = rng.uniform(0.6, 2.2, NST)

# moon + vinyl
MR = 255
def moon_img(r):
    s = 2 * r; yy, xx = np.mgrid[0:s, 0:s]; dd = np.sqrt((xx - r) ** 2 + (yy - r) ** 2) / r
    n = np.asarray(Image.fromarray((rng.random((s // 14, s // 14)) * 255).astype(np.uint8)).resize((s, s), Image.BICUBIC)).astype(np.float32) / 255
    n2 = np.asarray(Image.fromarray((rng.random((s // 45, s // 45)) * 255).astype(np.uint8)).resize((s, s), Image.BICUBIC)).astype(np.float32) / 255
    sh = 0.8 + 0.1 * n + 0.14 * n2 - 0.16 * dd ** 3
    rgb = np.stack([sh * 238, sh * 240, sh * 252], -1).clip(0, 255); a = np.clip((1 - dd) * r / 1.5, 0, 1) * 255
    return Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), 'RGBA')
MOON = moon_img(MR)
def vinyl_img(r):
    s = 2 * r; yy, xx = np.mgrid[0:s, 0:s]; dd = np.sqrt((xx - r) ** 2 + (yy - r) ** 2) / r; ang = np.arctan2(yy - r, xx - r)
    grooves = 0.5 + 0.5 * np.sin(dd * r * 1.3)
    sheen = 0.5 + 0.5 * np.cos(2 * ang)                      # light streak across the grooves
    v = 18 + 14 * grooves * (dd > 0.36) + 70 * sheen ** 6 * (dd > 0.36)
    rgb = np.stack([v * 0.95, v * 0.97, v * 1.15], -1).clip(0, 255)
    rim = (dd > 0.97) * 60; rgb += rim[..., None]
    a = np.clip((1 - dd) * r / 1.5, 0, 1) * 255
    im = Image.fromarray(np.dstack([rgb.clip(0, 255), a]).astype(np.uint8), 'RGBA')
    logo = Image.open(P(C['assets']['logo'])); lh = int(r * 0.78); lw = int(logo.width * lh / logo.height)
    label = Image.new('RGBA', (s, s), (0, 0, 0, 0)); ImageDraw.Draw(label).ellipse([r - r * 0.4, r - r * 0.4, r + r * 0.4, r + r * 0.4], fill=(244, 236, 214, 255))
    im.alpha_composite(label); im.alpha_composite(logo.resize((lw, lh), Image.LANCZOS), (r - lw // 2, r - lh // 2))
    ImageDraw.Draw(im).ellipse([r - 7, r - 7, r + 7, r + 7], fill=(10, 10, 16, 255))
    return im
VINYL = vinyl_img(MR)

# crowd on the rooftop
PEOPLE = []
for i in range(17):
    x = -20 + i * 70 + rng.uniform(-18, 18); h = rng.uniform(170, 240)
    PEOPLE.append(dict(x=x, h=h, ph=rng.uniform(0, 1), arms=rng.random() < 0.65, amp=rng.uniform(14, 30)))
def draw_crowd(im, t):
    d = ImageDraw.Draw(im); k = kick_env(t, 0.18)
    base = 1920
    for p in PEOPLE:
        bounce = p['amp'] * (k if t >= DROP and t < BRK else 0.25 * (0.5 + 0.5 * math.sin(2 * math.pi * (t / (2 * BEAT) + p['ph']))))
        top = base - p['h'] - bounce; x = p['x']
        d.rounded_rectangle([x - 34, top + 52, x + 34, base + 40], 26, fill=(3, 3, 8))
        d.ellipse([x - 23, top, x + 23, top + 50], fill=(3, 3, 8))
        if p['arms'] and t >= DROP - 1.0:
            up = min(1, (t - DROP + 1.0)) * (1 if t < BRK + 2 else max(0, 1 - (t - BRK - 2) / 1.5))
            for s in (-1, 1):
                ex, ey = x + s * (40 + 10 * up), top + 60 - 130 * up - bounce * 0.5
                d.line([(x + s * 28, top + 70), (ex, ey)], fill=(3, 3, 8), width=16)
                d.ellipse([ex - 9, ey - 9, ex + 9, ey + 9], fill=(3, 3, 8))
    d.rectangle([0, 1880, W, H], fill=(3, 3, 8))

# lasers
LCOL = [VIOLET, CYAN, (255, 90, 200), GREEN, CYAN, VIOLET]
def draw_lasers(t, inten):
    if inten <= 0: return None
    L = Image.new('RGB', (W, H), 0); d = ImageDraw.Draw(L); ox, oy = cx, 1700
    nb = 6; ph = 2 * math.pi * (t - DROP) / (8 * BEAT)
    pal = LCOL if t < C['date_at'] else [CYAN, GREEN, CYAN, VIOLET, GREEN, CYAN]
    for i in range(nb):
        a = math.radians(-90 + (i - (nb - 1) / 2) * 15 + 26 * math.sin(ph + i * 0.7))
        ex, ey = ox + 2600 * math.cos(a), oy + 2600 * math.sin(a); pa = a + math.pi / 2
        wd = 16; col = tuple(int(c * inten) for c in pal[i % len(pal)])
        d.polygon([(ox, oy), (ex + wd * math.cos(pa), ey + wd * math.sin(pa)), (ex - wd * math.cos(pa), ey - wd * math.sin(pa))], fill=col)
    core = L.copy(); glow = L.filter(ImageFilter.GaussianBlur(14))
    return ImageChops.add(ImageChops.multiply(core, Image.new('RGB', (W, H), (150, 150, 150))), ImageChops.add(glow, glow))

# ---------------- text ----------------
def tw(d, txt, font, track): return sum(d.textlength(c, font=font) for c in txt) + track * (len(txt) - 1)
def spaced(d, x, y, txt, font, fill, track):
    x = x - tw(d, txt, font, track) / 2
    for c in txt: d.text((x, y), c, font=font, fill=fill); x += d.textlength(c, font=font) + track
def glow_layer(items):
    out = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    for it in items:
        im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im); f = F(it['font'], it['size'])
        col = tuple(it.get('color', WHITE))
        if it.get('track'): spaced(d, cx, it['y'], it['text'], f, col, it['track'])
        else: d.text((cx, it['y']), it['text'], font=f, fill=col, anchor='mt')
        a = im.split()[3]
        if it.get('glow'):
            g = Image.new('RGBA', (W, H), tuple(it['glow']) + (0,))
            g.putalpha(a.filter(ImageFilter.GaussianBlur(it.get('glow_r', 18))).point(lambda v: min(255, int(v * 2.2)))); out.alpha_composite(g)
        sh = Image.new('RGBA', (W, H), (0, 0, 0, 0)); sh.putalpha(a.filter(ImageFilter.GaussianBlur(8)).point(lambda v: int(v * 0.8)))
        out.alpha_composite(sh, (0, 4)); out.alpha_composite(im)
    return out
def put(fr, L, a, dy=0):
    if a <= 0: return
    if a < 1: L = L.copy(); L.putalpha(L.split()[3].point(lambda v: int(v * a)))
    fr.alpha_composite(L, (0, int(dy)))
def glitch(fr, L, t, t0, dur=0.35):
    """glitch-in: sliced horizontal offsets + RGB split that settle"""
    p = (t - t0) / dur
    if p < 0: return
    if p >= 1: fr.alpha_composite(L); return
    r = np.random.default_rng(int(t * 1000)); amt = (1 - p) ** 2
    bb = L.getbbox() or (0, 0, W, H); y0, y1 = bb[1] - 10, bb[3] + 10
    out = Image.new('RGBA', (W, H), (0, 0, 0, 0)); n = 9; hstep = max(1, (y1 - y0) // n)
    for i in range(n):
        ys = y0 + i * hstep; strip = L.crop((0, ys, W, ys + hstep)); dx = int(r.uniform(-120, 120) * amt)
        out.alpha_composite(strip, (dx, ys))
    a = out.split()[3]
    for col, off in ((CYAN, -int(16 * amt)), ((255, 60, 180), int(16 * amt))):
        c = Image.new('RGBA', (W, H), col + (0,)); c.putalpha(a.point(lambda v: int(v * 0.6 * amt))); fr.alpha_composite(c, (off, 0))
    out.putalpha(a.point(lambda v: int(v * min(1, p * 2.5)))); fr.alpha_composite(out)

T_pres = glow_layer([{'text': C['presenter'], 'font': 'cinzel-700.ttf', 'size': 42, 'y': 300, 'color': SILVER, 'track': 8, 'glow': (120, 140, 255), 'glow_r': 14},
                     {'text': 'presents', 'font': 'corm-500i.ttf', 'size': 60, 'y': 365, 'color': GOLD}])
T_t1 = glow_layer([{'text': C['title1'], 'font': 'cinzel-700.ttf', 'size': 140, 'y': 215, 'color': SILVER, 'track': 6, 'glow': (150, 170, 255), 'glow_r': 28}])
T_t2 = glow_layer([{'text': C['title2'], 'font': 'Inter-ExtraBold.otf', 'size': 86, 'y': 385, 'color': WHITE, 'track': 24, 'glow': VIOLET, 'glow_r': 22}])
T_live = glow_layer([{'text': 'LIVE ON THE DECKS', 'font': 'cinzel-700.ttf', 'size': 40, 'y': 1030, 'color': GOLD, 'track': 8}])
T_a1 = glow_layer([{'text': C['artists'][0], 'font': 'Inter-ExtraBold.otf', 'size': 118, 'y': 1095, 'color': WHITE, 'track': 6, 'glow': VIOLET, 'glow_r': 26}])
T_amp = glow_layer([{'text': '&', 'font': 'corm-500i.ttf', 'size': 90, 'y': 1225, 'color': GOLD}])
T_a2 = glow_layer([{'text': C['artists'][1], 'font': 'Inter-ExtraBold.otf', 'size': 118, 'y': 1315, 'color': WHITE, 'track': 6, 'glow': CYAN, 'glow_r': 26}])
T_date = glow_layer([{'text': C['date_line'], 'font': 'Inter-ExtraBold.otf', 'size': 80, 'y': 1060, 'color': WHITE, 'track': 4, 'glow': VIOLET, 'glow_r': 20}])
T_time = glow_layer([{'text': C['time_line'], 'font': 'Inter-ExtraBold.otf', 'size': 100, 'y': 1170, 'color': WHITE, 'track': 5, 'glow': CYAN, 'glow_r': 26}])
T_seats = glow_layer([{'text': C['seats_line'], 'font': 'cinzel-700.ttf', 'size': 42, 'y': 1320, 'color': GOLD, 'track': 5}])
E_title = glow_layer([{'text': C['title1'] + ' ' + C['title2'], 'font': 'cinzel-700.ttf', 'size': 66, 'y': 830, 'color': SILVER, 'track': 4, 'glow': (150, 170, 255), 'glow_r': 16}])
E_art = glow_layer([{'text': C['artists'][0] + '  &  ' + C['artists'][1], 'font': 'Inter-ExtraBold.otf', 'size': 62, 'y': 925, 'color': WHITE, 'track': 3, 'glow': VIOLET, 'glow_r': 18}])
E_when = glow_layer([{'text': C['end_when'], 'font': 'Inter-Bold.otf', 'size': 52, 'y': 1015, 'color': CYAN, 'track': 3}])
E_book = glow_layer([{'text': C['seats_line'], 'font': 'cinzel-700.ttf', 'size': 38, 'y': 1118, 'color': GOLD, 'track': 5},
                     {'text': C['booking_label'], 'font': 'Inter-SemiBold.otf', 'size': 44, 'y': 1182, 'color': SILVER},
                     {'text': C['phone'], 'font': 'Inter-ExtraBold.otf', 'size': 80, 'y': 1240, 'color': WHITE, 'track': 2, 'glow': (120, 140, 255), 'glow_r': 14}])
E_tag = glow_layer([{'text': C['place'], 'font': 'cinzel-700.ttf', 'size': 36, 'y': 1365, 'color': GOLD, 'track': 7},
                    {'text': C['tagline'], 'font': 'corm-500i.ttf', 'size': 62, 'y': 1425, 'color': CREAM}])
RULE = Image.new('RGBA', (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(RULE)
dr.line([(cx - 240, 1092), (cx - 14, 1092)], fill=GOLD, width=2); dr.line([(cx + 14, 1092), (cx + 240, 1092)], fill=GOLD, width=2)
dr.polygon([(cx, 1085), (cx + 7, 1092), (cx, 1099), (cx - 7, 1092)], fill=GOLD)
# dark plate behind the end-card text so it reads over the mountains
PLATE = Image.new('RGBA', (W, H), (0, 0, 0, 0)); pm = Image.new('L', (W, H), 0)
ImageDraw.Draw(pm).rounded_rectangle([40, 790, W - 40, 1500], 40, fill=170); PLATE = Image.new('RGBA', (W, H), (6, 6, 18, 0)); PLATE.putalpha(pm.filter(ImageFilter.GaussianBlur(40)))
pm2 = Image.new('L', (W, H), 0); ImageDraw.Draw(pm2).rounded_rectangle([60, 1010, W - 60, 1400], 50, fill=185)
INFO_PLATE = Image.new('RGBA', (W, H), (8, 6, 22, 0)); INFO_PLATE.putalpha(pm2.filter(ImageFilter.GaussianBlur(45)))
logo = Image.open(P(C['assets']['logo'])); LOGO_H = 470; LOGO = logo.resize((int(logo.width * LOGO_H / logo.height), LOGO_H), Image.LANCZOS)

# ---------------- music ----------------
m = make_edm(TOTAL, DROP, BRK, seed=C.get('seed', 5), bpm=BPM)
aw = os.path.splitext(P(C['out']))[0] + '_audio.wav'; os.makedirs(os.path.dirname(aw), exist_ok=True)
wavfile.write(aw, SR, (m * 32767).astype(np.int16))
st = subprocess.run(['ffmpeg', '-hide_banner', '-i', aw, '-af', 'ebur128=peak=true', '-f', 'null', '-'], capture_output=True, text=True).stderr
I = float(re.findall(r'I:\s+(-?[\d.]+) LUFS', st)[-1]); pk = float(re.findall(r'Peak:\s+(-?[\d.]+) dBFS', st)[-1])
gn = min(-14 - I, -1.0 - pk); wavfile.write(aw, SR, (np.clip(m * 10 ** (gn / 20), -1, 1) * 32767).astype(np.int16))

# ---------------- render ----------------
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-i', aw, '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-maxrate', '12M', '-bufsize', '24M', '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', P(C['out'])], stdin=subprocess.PIPE)
LU, DA = C['lineup_at'], C['date_at']
for n in range(int(round(TOTAL * FPS))):
    t = n / FPS; k = kick_env(t)
    fr = SKY.copy().convert('RGBA'); d = ImageDraw.Draw(fr)
    for i in range(NST):                                        # stars
        b = 0.5 + 0.5 * math.sin(t * SFQ[i] + SPH[i]); c = int(90 + 165 * b); r = SRAD[i]
        d.ellipse([SX[i] - r, SY[i] - r, SX[i] + r, SY[i] + r], fill=(c, c, min(255, c + 25)))
    if 1.2 < t < 2.2:                                           # shooting star
        p = (t - 1.2) / 1.0; x0, y0 = 120 + 700 * p, 180 + 260 * p
        d.line([(x0 - 140 * (1 - p * 0.5), y0 - 52), (x0, y0)], fill=(230, 235, 255), width=3)
    # moon rises over the ranges, then becomes a vinyl on the drop, back to moon for the end card
    my = 1380 - 650 * smooth(t / (DROP - 0.3)) if t < DROP else 730
    if t >= BRK: my = 730 - 185 * smooth((t - BRK) / 1.2)
    scale = 1.0 + (0.06 * k if DROP <= t < BRK else 0)
    gl = Image.new('L', (W, H), 0); gr = MR * (1.35 + 0.25 * k)
    ImageDraw.Draw(gl).ellipse([cx - gr, my - gr, cx + gr, my + gr], fill=int(150 + 60 * k))
    glo = Image.new('RGBA', (W, H), (180, 190, 255, 0)); glo.putalpha(gl.filter(ImageFilter.GaussianBlur(90))); fr.alpha_composite(glo)
    vin = smooth((t - DROP + 0.05) / 0.3) * (1 - smooth((t - BRK) / 0.8)) if t >= DROP - 0.05 else 0
    rad = int(MR * scale)
    mo = MOON if rad == MR else MOON.resize((2 * rad, 2 * rad), Image.BICUBIC)
    if vin < 1: fr.alpha_composite(mo, (cx - rad, int(my) - rad))
    if vin > 0:
        vi = VINYL.rotate(-(t - DROP) * 200, resample=Image.BICUBIC)
        if rad != MR: vi = vi.resize((2 * rad, 2 * rad), Image.BICUBIC)
        if vin < 1: vi.putalpha(vi.split()[3].point(lambda v: int(v * vin)))
        fr.alpha_composite(vi, (cx - rad, int(my) - rad))
    # mountains with slow parallax
    fr.alpha_composite(BACK, (-int(60 + 8 * t), 0))
    fx = int((t * 25) % W); fog = FOG.crop((fx, 0, fx + W, 600)); fr.alpha_composite(fog, (0, 1050))
    fr.alpha_composite(MID, (-int(40 + 14 * t), 0))
    fr.alpha_composite(NEAR, (-int(20 + 20 * t), 0))
    # lasers over the mountains (drop only), fading into the end card
    li = (0.55 + 0.45 * k) * (1 - smooth((t - BRK) / 0.6)) if t >= DROP else (0.25 * smooth((t - DROP + 0.8) / 0.8) if t > DROP - 0.8 else 0)
    Ls = draw_lasers(t, li)
    if Ls is not None: fr = Image.fromarray(np.asarray(ImageChops.screen(fr.convert('RGB'), Ls))).convert('RGBA')
    draw_crowd(fr, t)
    if LU - 0.2 <= t < BRK: put(fr, INFO_PLATE, env(t, LU - 0.2, BRK, 0.4, 0.4))
    # ---- text ----
    if t < DROP:
        put(fr, T_pres, env(t, 0.3, DROP - 0.6, 0.5, 0.4), dy=int(20 * (1 - ease((t - 0.3) / 0.6))))
        put(fr, T_t1, env(t, 2.2, 99, 0.6) * 0.9, dy=0) if t > 2.2 and False else None
    if t >= DROP - 0.6 and t < BRK:
        put(fr, T_t1, env(t, DROP - 0.6, BRK, 0.5), dy=int(-30 * (1 - ease((t - DROP + 0.6) / 0.5))))
        glitch(fr, T_t2, t, DROP)
        if k > 0.05 and t >= DROP + 0.35: put(fr, T_t2, 0.4 * k)
    if LU <= t < DA:
        a = env(t, LU, DA, 0.3, 0.25)
        put(fr, T_live, a)
        if t < DA - 0.25: glitch(fr, T_a1, t, LU + 0.1); glitch(fr, T_a2, t, LU + 1.0)
        else: put(fr, T_a1, a); put(fr, T_a2, a)
        put(fr, T_amp, env(t, LU + 0.7, DA, 0.3, 0.25))
    if DA <= t < BRK:
        a = env(t, DA, BRK, 0.3, 0.3)
        if t < BRK - 0.3: glitch(fr, T_date, t, DA + 0.05); glitch(fr, T_time, t, DA + 0.55)
        else: put(fr, T_date, a); put(fr, T_time, a)
        put(fr, T_seats, env(t, DA + 1.4, BRK, 0.4, 0.3))
    if t >= BRK:
        lt = t - BRK
        a = env(lt, 0.4, 99, 0.8); lg = LOGO.copy(); lg.putalpha(lg.split()[3].point(lambda v: int(v * a)))
        fr.alpha_composite(lg, (cx - LOGO.width // 2, int(my) - LOGO_H // 2))
        put(fr, PLATE, env(lt, 0.6, 99, 0.8))
        put(fr, E_title, env(lt, 0.9, 99, 0.4), dy=int(20 * (1 - ease((lt - 0.9) / 0.4))))
        put(fr, E_art, env(lt, 1.3, 99, 0.4), dy=int(20 * (1 - ease((lt - 1.3) / 0.4))))
        put(fr, E_when, env(lt, 1.7, 99, 0.4)); put(fr, RULE, env(lt, 2.0, 99, 0.4))
        put(fr, E_book, env(lt, 2.3, 99, 0.5)); put(fr, E_tag, env(lt, 3.1, 99, 0.6))
    # beat strobe + drop / break flashes
    fl = 0.0
    if DROP <= t < BRK: fl = 0.10 * k
    for tt, s in ((DROP, 0.75), (BRK, 0.55)):
        if 0 <= t - tt < 0.3: fl = max(fl, s * math.exp(-(t - tt) / 0.08))
    if fl > 0.01: fr.alpha_composite(Image.new('RGBA', (W, H), (230, 225, 255, int(255 * fl))))
    if TOTAL - t < 0.4: fr = Image.blend(fr.convert('RGB'), Image.new('RGB', (W, H), 0), 1 - (TOTAL - t) / 0.4).convert('RGBA')
    enc.stdin.write(fr.convert('RGB').tobytes())
enc.stdin.close(); enc.wait(); print(f'done {P(C["out"])} {TOTAL:.1f}s (audio {I:.1f} LUFS -> {gn:+.1f} dB)')
