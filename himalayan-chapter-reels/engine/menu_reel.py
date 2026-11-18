"""Menu launch reel: the camera glides over illustrated menu pages, pages flip like a book,
a gold highlighter draws around featured dishes, and brand tags show dish + price.
usage: python engine/menu_reel.py episodes/menu_launch.json
"""
import json, math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from music import make_music, SR

cfg = json.load(open(sys.argv[1])); base = os.path.dirname(os.path.abspath(sys.argv[1]))
P = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(base, x))
W, H, FPS = 1080, 1920, 30
PW, PH = 1055, 1491                       # menu page size (page coordinates)
CREAM=(247,240,222); LGOLD=(240,212,150); GOLD=(170,122,48); GREEN=(22,56,38); BOX=(16,40,28)
FD = P(cfg['assets']['fonts']); F = lambda n, s: ImageFont.truetype(os.path.join(FD, n), s)
cx0 = W // 2
ease = lambda p: 1 - (1 - min(max(p, 0), 1)) ** 3
def smooth(p): p = min(max(p, 0), 1); return p * p * (3 - 2 * p)
def env(t, a, b, fi=0.35, fo=0.25): return float(np.clip(min((t - a) / fi, (b - t) / fo), 0, 1))
rng = np.random.default_rng(3)

# ---------- backdrop: deep green linen ----------
def backdrop():
    b = np.ones((H, W, 3), np.float32) * np.array([20, 44, 32], np.float32) / 255
    n = rng.normal(0, 1, (H, W)).astype(np.float32)
    lin = np.asarray(Image.fromarray(((n * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(1))).astype(np.float32) / 255
    b *= (0.92 + 0.12 * lin)[..., None]
    yy, xx = np.mgrid[0:H, 0:W]; r = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    b *= (1.15 - 0.9 * r ** 2)[..., None]
    return Image.fromarray((b.clip(0, 1) * 255).astype(np.uint8))
BACK = backdrop()

# ---------- pages: 2x upscale + shadow ----------
SC = 2; PAD = 90
pages = {}
def page(name):
    if name in pages: return pages[name]
    im = Image.open(P(cfg['pages'][name])).convert('RGB').resize((PW * SC, PH * SC), Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.6, percent=60, threshold=2))
    full = Image.new('RGBA', (PW * SC + 2 * PAD, PH * SC + 2 * PAD), (0, 0, 0, 0))
    m = Image.new('L', full.size, 0); ImageDraw.Draw(m).rectangle([PAD + 10, PAD + 24, PAD + PW * SC + 10, PAD + PH * SC + 24], fill=150)
    full.putalpha(m.filter(ImageFilter.GaussianBlur(30)))
    sh = Image.new('RGBA', full.size, (0, 0, 0, 255)); sh.putalpha(full.split()[3])
    out = Image.new('RGBA', full.size, (0, 0, 0, 0)); out.alpha_composite(sh); out.paste(im, (PAD, PAD))
    pages[name] = out
    if len(pages) > 3: pages.pop(next(iter(pages)))
    return out

def clamp_cam(cx, cy, vw):
    vh = vw * H / W
    if vw < PW: cx = min(max(cx, vw / 2), PW - vw / 2)
    else: cx = PW / 2
    if vh < PH: cy = min(max(cy, vh / 2), PH - vh / 2)
    else: cy = PH / 2
    return cx, cy, vw
def to_frame(cam, x, y):
    cx, cy, vw = cam; s = W / vw
    return (x - cx) * s + W / 2, (y - cy) * s + H / 2
def render_page(name, cam):
    cx, cy, vw = clamp_cam(*cam); s = W / vw                      # frame px per page px
    pg = page(name); k = s / SC                                    # frame px per world px
    # frame position of padded page image
    ox = W / 2 - (cx * SC + PAD) * k; oy = H / 2 - (cy * SC + PAD) * k
    fx0, fy0 = max(0, int(math.floor(ox))), max(0, int(math.floor(oy)))
    fx1, fy1 = min(W, int(math.ceil(ox + pg.width * k))), min(H, int(math.ceil(oy + pg.height * k)))
    fr = BACK.copy().convert('RGBA')
    if fx1 > fx0 and fy1 > fy0:
        box = (max(0.0, (fx0 - ox) / k), max(0.0, (fy0 - oy) / k), min(pg.width, (fx1 - ox) / k), min(pg.height, (fy1 - oy) / k))
        patch = pg.resize((fx1 - fx0, fy1 - fy0), Image.BICUBIC, box=box)
        fr.alpha_composite(patch, (fx0, fy0))
    return fr, (cx, cy, vw)

# ---------- overlays ----------
_RUPEE = {}
def gf(c, font):
    """brand fonts lack the rupee sign: borrow it from Inter at a matching size"""
    if c != '\u20b9': return font
    k = font.size
    if k not in _RUPEE: _RUPEE[k] = ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-SemiBold.otf', int(k * 0.92))
    return _RUPEE[k]
def tw(d, txt, font, track): return sum(d.textlength(c, font=gf(c, font)) for c in txt) + track * (len(txt) - 1)
def spaced(d, x, y, txt, font, fill, track):
    x = x - tw(d, txt, font, track) / 2
    for c in txt:
        f = gf(c, font); d.text((x, y + (font.size - f.size) * 0.55 if f is not font else y), c, font=f, fill=fill); x += d.textlength(c, font=f) + track
TAGS = {}
def tag_img(tg):
    key = json.dumps(tg)
    if key in TAGS: return TAGS[key]
    f1, f2, f3 = F('cinzel-700.ttf', 60), F('corm-500i.ttf', 56), F('cinzel-700.ttf', 48)
    tmp = ImageDraw.Draw(Image.new('RGBA', (10, 10)))
    w = max(tw(tmp, tg['name'], f1, 4), tmp.textlength(tg.get('desc', ''), font=f2), tw(tmp, tg.get('price', ''), f3, 4)) + 110
    w = min(max(w, 600), 1010); h = 104 + (76 if tg.get('desc') else 0) + (72 if tg.get('price') else 0)
    im = Image.new('RGBA', (int(w) + 40, int(h) + 40), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    sh = Image.new('L', im.size, 0); ImageDraw.Draw(sh).rounded_rectangle([20, 26, w + 20, h + 26], 28, fill=140)
    shi = Image.new('RGBA', im.size, (0, 0, 0, 255)); shi.putalpha(sh.filter(ImageFilter.GaussianBlur(12))); im.alpha_composite(shi)
    d.rounded_rectangle([20, 20, w + 20, h + 20], 28, fill=BOX + (238,), outline=GOLD + (255,), width=3)
    c = (w + 40) / 2; y = 48
    spaced(d, c, y, tg['name'], f1, CREAM, 4); y += 84
    if tg.get('desc'): d.text((c, y + 26), tg['desc'], font=f2, fill=CREAM, anchor='mm'); y += 76
    if tg.get('price'): spaced(d, c, y, tg['price'], f3, LGOLD, 4)
    TAGS[key] = im; return im
def draw_highlights(fr, cam, hls, lt):
    if not hls: return fr
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    dim = Image.new('L', (W, H), 0); dd = ImageDraw.Draw(dim)
    amt = 0
    for i, hl in enumerate(hls):
        t0 = hl.get('t', 0.4 + 0.45 * i); p = (lt - t0) / 0.6
        if p <= 0: continue
        x0, y0 = to_frame(cam, hl['r'][0] - 10, hl['r'][1] - 8); x1, y1 = to_frame(cam, hl['r'][2] + 10, hl['r'][3] + 8)
        amt = max(amt, min(1, p))
        dd.rounded_rectangle([x0, y0, x1, y1], 14, fill=255)
        # stroke draws on around the perimeter
        e = ease(p); per = 2 * ((x1 - x0) + (y1 - y0)); L = per * e
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]; acc = 0
        for a, b in zip(pts, pts[1:]):
            seg = math.dist(a, b)
            if acc >= L: break
            f = min(1, (L - acc) / seg); d.line([a, (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)], fill=GOLD + (255,), width=6)
            acc += seg
        d.ellipse([x0 - 9, y0 - 9 + (y1 - y0) / 2, x0 + 9, y0 + 9 + (y1 - y0) / 2], fill=GOLD + (int(255 * min(1, p)),))
    if amt > 0:
        m = dim.filter(ImageFilter.GaussianBlur(6)); inv = Image.eval(m, lambda v: 255 - v)
        dark = Image.new('RGBA', (W, H), (60, 42, 20, 0)); dark.putalpha(inv.point(lambda v: int(v * 0.16 * amt)))
        fr.alpha_composite(dark)
    fr.alpha_composite(ov); return fr
def put_tag(fr, tg, lt, dur, pos):
    a = env(lt, tg.get('t', 0.55), dur - 0.15, 0.35, 0.2)
    if a <= 0: return fr
    im = tag_img(tg).copy(); im.putalpha(im.split()[3].point(lambda v: int(v * a)))
    y = (1400 if pos == 'bottom' else 230) + int(30 * (1 - ease((lt - tg.get('t', 0.55)) / 0.4)))
    fr.alpha_composite(im, (int(cx0 - im.width / 2), y)); return fr
def text_layer(lines):
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for l in lines:
        f = F(l['font'], l['size'])
        if l.get('track'): spaced(d, cx0, l['y'], l['text'], f, tuple(l.get('color', CREAM)), l['track'])
        else: d.text((cx0, l['y']), l['text'], font=f, fill=tuple(l.get('color', CREAM)), anchor='mm')
    sh = Image.new('RGBA', (W, H), (0, 0, 0, 0)); sh.putalpha(im.split()[3].filter(ImageFilter.GaussianBlur(12)))
    o = Image.new('RGBA', (W, H), (0, 0, 0, 0)); o.alpha_composite(sh, (0, 5)); o.alpha_composite(im); return o

# ---------- beats ----------
beats = cfg['beats']; t = 0
for b in beats: b['t0'] = t; t += b['dur']; b['t1'] = t
TOTAL = t
# carry camera between beats on the same page (glide)
for i, b in enumerate(beats):
    if b['type'] == 'page' and b.get('glide') and i > 0: b['cam0'] = beats[i - 1]['cam1']

def hook_frames(b):
    vf = (f"fps={FPS},scale=1080:1920,zoompan=z='1+0.0012*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1920:fps={FPS},"
          "curves=all='0/0 0.6/0.56 1/0.93',eq=saturation=0.97,colorbalance=rm=0.03:bm=-0.03,noise=alls=5:allf=t,vignette=PI/6")
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', str(b.get('in', 0)), '-t', str(b['dur'] + 0.6), '-i', P(b['src']), '-vf', vf,
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    last = None
    while True:
        r = p.stdout.read(W * H * 3)
        if len(r) < W * H * 3: break
        last = Image.frombytes('RGB', (W, H), r); yield last
    while True: yield last
HOOK = {}
def render_beat(b, t):
    lt = t - b['t0']
    if b['type'] == 'hook':
        if 'gen' not in b: b['gen'] = hook_frames(b); b['L'] = [(text_layer(g['lines']), g['t']) for g in b['text']]
        fr = next(b['gen']).copy()
        fr = Image.blend(fr, Image.new('RGB', (W, H), (0, 0, 0)), b.get('dim', 0.45)).convert('RGBA')
        for L, t0 in b['L']:
            a = env(lt, t0, 99, 0.4)
            if a > 0:
                Lc = L.copy(); Lc.putalpha(L.split()[3].point(lambda v: int(v * a)))
                fr.alpha_composite(Lc, (0, int(24 * (1 - ease((lt - t0) / 0.4)))))
        return fr
    if b['type'] == 'page':
        hold = b.get('hold', FLIP if b.get('trans', 'flip') == 'flip' and not b.get('glide') else 0)
        p = smooth((lt - hold) / max(0.1, b['dur'] - hold - 0.6))
        cam = tuple(a + (c - a) * p for a, c in zip(b['cam0'], b['cam1']))
        fr, cam = render_page(b['page'], cam)
        fr = draw_highlights(fr, cam, b.get('hl', []), lt)
        if b.get('tag'):
            pos = b.get('tag_pos', 'bottom'); fr = put_tag(fr, b['tag'], lt, b['dur'] + (b.get('tag_carry', 0)), pos)
        if b.get('note'):
            if 'NL' not in b: b['NL'] = text_layer(b['note'])
            a = env(lt, b.get('note_t', 0.3), b.get('note_out', 99), 0.4, 0.3)
            if a > 0:
                Lc = b['NL'].copy(); Lc.putalpha(b['NL'].split()[3].point(lambda v: int(v * a))); fr.alpha_composite(Lc)
        return fr
    if b['type'] == 'end':
        if 'paper' not in b:
            pp = np.ones((H, W, 3), np.float32) * np.array([243, 234, 212], np.float32) / 255
            pp += (rng.normal(0, 1, (H, W)).astype(np.float32) * 0.012)[..., None]
            yy, xx = np.mgrid[0:H, 0:W]; r = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
            b['paper'] = Image.fromarray(((pp * (1.04 - 0.32 * r ** 2)[..., None]).clip(0, 1) * 255).astype(np.uint8)).convert('RGBA')
            logo = Image.open(P(cfg['assets']['logo'])); lh = 600; lw = int(logo.width * lh / logo.height)
            b['logo'] = logo.resize((lw, lh), Image.LANCZOS)
            im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
            d.text((cx0, 880), b['line1'], font=F('corm-500i.ttf', 80), fill=GREEN, anchor='mm')
            spaced(d, cx0, 950, b['line2'], F('cinzel-700.ttf', 76), GREEN, 5)
            y = 1075; d.line([(cx0 - 230, y), (cx0 - 16, y)], fill=GOLD, width=3); d.line([(cx0 + 16, y), (cx0 + 230, y)], fill=GOLD, width=3)
            d.polygon([(cx0, y - 8), (cx0 + 8, y), (cx0, y + 8), (cx0 - 8, y)], fill=GOLD)
            off = 0
            if b.get('sub'): spaced(d, cx0, 1108, b['sub'], F('cinzel-700.ttf', 36), GOLD, 7); off = 70
            for i, l in enumerate(b['info']): d.text((cx0, 1140 + off + 64 * i), l, font=F('corm-600.ttf', 48), fill=GREEN, anchor='mm')
            spaced(d, cx0, 1150 + off + 64 * len(b['info']), b['place'], F('cinzel-700.ttf', 30), GOLD, 5)
            b['txt'] = im
        fr = b['paper'].copy()
        a = env(lt, 0.25, 99, 0.6); lg = b['logo'].copy(); lg.putalpha(lg.split()[3].point(lambda v: int(v * a)))
        fr.alpha_composite(lg, (cx0 - lg.width // 2, 170 + int(30 * (1 - ease((lt - 0.25) / 0.6)))))
        a = env(lt, 1.0, 99, 0.6); tx = b['txt'].copy(); tx.putalpha(tx.split()[3].point(lambda v: int(v * a))); fr.alpha_composite(tx)
        rem = TOTAL - t
        if rem < 0.6: fr = Image.blend(fr.convert('RGB'), Image.new('RGB', (W, H), 0), 1 - rem / 0.6).convert('RGBA')
        return fr

# ---------- transitions ----------
def coeffs(dst, src):
    A = []; B = []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B += [u, v]
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()
def page_flip(a_img, b_img, p):
    """old frame turns away around the left spine, revealing the new one"""
    th = ease(p) * math.pi / 2; w = W * math.cos(th); lift = 0.06 * H * math.sin(th)
    out = b_img.convert('RGB')
    if w > 2:
        # shadow cast on the new page next to the turning edge
        sh = Image.new('L', (W, H), 0); dsh = ImageDraw.Draw(sh)
        for i in range(160):
            dsh.line([(w + i, 0), (w + i, H)], fill=int(150 * math.sin(th) * (1 - i / 160) ** 2))
        out = Image.composite(Image.new('RGB', (W, H), 0), out, sh)
        src = [(0, 0), (W, 0), (W, H), (0, H)]; dst = [(0, 0), (w, -lift), (w, H + lift), (0, H)]
        A = a_img.convert('RGBA').transform((W, H), Image.PERSPECTIVE, coeffs(dst, src), Image.BICUBIC)
        shade = Image.new('RGBA', (W, H), (0, 0, 0, 0)); shade.putalpha(A.split()[3].point(lambda v: int(v * 0.45 * math.sin(th))))
        A.alpha_composite(shade)
        out = out.convert('RGBA'); out.alpha_composite(A)
    return out
XF = 0.5; FLIP = 0.9

# ---------- audio ----------
end_t = [b['t0'] for b in beats if b['type'] == 'end'][0]
if cfg.get('music') == 'himalayan':
    from music_himalayan import make_himalayan
    mk = cfg['music_marks']
    m = make_himalayan(TOTAL, beats[mk['flute']]['t0'], beats[mk['tabla']]['t0'], beats[mk['break']]['t0'], end_t, seed=cfg.get('seed', 7))
else:
    m = make_music(TOTAL, beats[1]['t0'], end_t, (), (), None, seed=cfg.get('seed', 31), flute_at=cfg.get('flute_at', []))
r = os.path.splitext(P(cfg['out']))[0] + '_audio.wav'; os.makedirs(os.path.dirname(r), exist_ok=True)
wavfile.write(r, SR, (m * 32767).astype(np.int16))
import re
st = subprocess.run(['ffmpeg', '-hide_banner', '-i', r, '-af', 'ebur128=peak=true', '-f', 'null', '-'], capture_output=True, text=True).stderr
I = float(re.findall(r'I:\s+(-?[\d.]+) LUFS', st)[-1]); pk = float(re.findall(r'Peak:\s+(-?[\d.]+) dBFS', st)[-1])
g = min(-14 - I, -1.5 - pk); wavfile.write(r, SR, (np.clip(m * 10 ** (g / 20), -1, 1) * 32767).astype(np.int16))

# ---------- render ----------
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-i', r, '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-maxrate', '10M', '-bufsize', '20M', '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', P(cfg['out'])], stdin=subprocess.PIPE)
for n in range(int(round(TOTAL * FPS))):
    t = n / FPS
    k = max(i for i, b in enumerate(beats) if b['t0'] <= t + 1e-6); b = beats[k]
    fr = render_beat(b, t)
    if k > 0:
        prev = beats[k - 1]; lt = t - b['t0']
        tr = b.get('trans', 'flip')
        if tr == 'flip' and lt < FLIP:
            fr = page_flip(render_beat(prev, min(t, prev['t1'] - 1 / FPS)), fr, lt / FLIP)
        elif tr == 'dissolve' and lt < XF:
            fr = Image.blend(render_beat(prev, prev['t1'] - 1 / FPS).convert('RGB'), fr.convert('RGB'), smooth(lt / XF))
    enc.stdin.write(fr.convert('RGB').tobytes())
enc.stdin.close(); enc.wait(); print(f'done {P(cfg["out"])} {TOTAL:.1f}s  (audio {I:.1f} LUFS -> {g:+.1f} dB)')
