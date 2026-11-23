"""Event promo reel (vertical poster style): cinematic footage strip, glowing title, lineup drop,
date/time, night-sky end card with logo, booking number and tagline. Music: music_edm.make_edm.
usage: python engine/event_reel.py episodes/event_xxx.json
"""
import json, math, os, re, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from music_edm import make_edm, SR

cfg = json.load(open(sys.argv[1])); base = os.path.dirname(os.path.abspath(sys.argv[1]))
P = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(base, x))
W, H, FPS = 1080, 1920, 30
BAND_Y, BAND_H = 700, 450
SILVER=(236,240,255); VIOLET=(176,120,255); CYAN=(120,225,255); GOLD=(226,186,104); CREAM=(247,240,222); WHITE=(255,255,255)
FD = P(cfg['assets']['fonts'])
INTER = '/usr/share/fonts/opentype/inter/'
def F(name, size):
    path = os.path.join(INTER, name) if name.startswith('Inter') else os.path.join(FD, name)
    return ImageFont.truetype(path, size)
cx = W // 2
ease = lambda p: 1 - (1 - min(max(p, 0), 1)) ** 3
def env(t, a, b, fi=0.35, fo=0.3): return float(np.clip(min((t - a) / fi, (b - t) / fo), 0, 1))
rng = np.random.default_rng(9)

# ---------- text with glow ----------
def tw(d, txt, font, track): return sum(d.textlength(c, font=font) for c in txt) + track * (len(txt) - 1)
def spaced(d, x, y, txt, font, fill, track):
    x = x - tw(d, txt, font, track) / 2
    for c in txt: d.text((x, y), c, font=font, fill=fill); x += d.textlength(c, font=font) + track
def glow_layer(items):
    """items: dicts text,font,size,y,color,glow,track,italic-anchor. Returns RGBA layer (text + coloured glow)."""
    out = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    for it in items:
        im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im); f = F(it['font'], it['size'])
        col = tuple(it.get('color', WHITE))
        if it.get('track'): spaced(d, cx, it['y'], it['text'], f, col, it['track'])
        else: d.text((cx, it['y']), it['text'], font=f, fill=col, anchor='mt')
        a = im.split()[3]
        if it.get('glow'):
            g = Image.new('RGBA', (W, H), tuple(it['glow']) + (0,))
            g.putalpha(a.filter(ImageFilter.GaussianBlur(it.get('glow_r', 18))).point(lambda v: min(255, int(v * 2.2))))
            out.alpha_composite(g)
        sh = Image.new('RGBA', (W, H), (0, 0, 0, 0)); sh.putalpha(a.filter(ImageFilter.GaussianBlur(6)).point(lambda v: int(v * 0.6)))
        out.alpha_composite(sh, (0, 3)); out.alpha_composite(im)
    return out
def put(fr, L, a, dy=0, scale=1.0):
    if a <= 0: return fr
    if scale != 1.0:
        w, h = int(W * scale), int(H * scale); L = L.resize((w, h), Image.BICUBIC)
        L2 = Image.new('RGBA', (W, H), (0, 0, 0, 0)); L2.alpha_composite(L, ((W - w) // 2, (H - h) // 2 + dy) if False else (0, 0), (max(0, (w - W) // 2), max(0, (h - H) // 2 - dy)))
        L = L2; dy = 0
    if a < 1: L = L.copy(); L.putalpha(L.split()[3].point(lambda v: int(v * a)))
    fr.alpha_composite(L, (0, int(dy)))
    return fr

# ---------- sky (stars + moon) ----------
NST = 420
SX, SY = rng.uniform(0, W, NST), rng.uniform(0, H, NST); SR_ = rng.uniform(0.6, 2.2, NST); SP = rng.uniform(0, 6.3, NST); SF = rng.uniform(0.5, 2.0, NST)
def sky(t, moon=(cx, 520, 230)):
    yy = np.linspace(0, 1, H)[:, None]
    g = np.concatenate([np.ones((H, W, 1)) * (8 + 18 * (1 - yy))[..., None][:, :, 0:1] * 0 + 0, np.zeros((H, W, 2))], 2) if False else None
    top = np.array([16, 18, 48], np.float32); bot = np.array([4, 5, 14], np.float32)
    bg = (top * (1 - yy[..., None]) + bot * yy[..., None]) * np.ones((1, W, 1), np.float32)
    im = Image.fromarray(bg.astype(np.uint8)).convert('RGBA')
    d = ImageDraw.Draw(im)
    for i in range(NST):
        b = 0.55 + 0.45 * math.sin(t * SF[i] + SP[i]); r = SR_[i]; c = int(255 * b)
        d.ellipse([SX[i] - r, SY[i] - r, SX[i] + r, SY[i] + r], fill=(c, c, min(255, c + 20), 255))
    if moon:
        mx, my, mr = moon
        glow = Image.new('L', (W, H), 0); ImageDraw.Draw(glow).ellipse([mx - mr * 1.25, my - mr * 1.25, mx + mr * 1.25, my + mr * 1.25], fill=150)
        gl = Image.new('RGBA', (W, H), (190, 200, 255, 0)); gl.putalpha(glow.filter(ImageFilter.GaussianBlur(90))); im.alpha_composite(gl)
        im.alpha_composite(MOON, (int(mx - mr), int(my - mr)))
    return im
def make_moon(r):
    s = 2 * r; yy, xx = np.mgrid[0:s, 0:s]; d = np.sqrt((xx - r) ** 2 + (yy - r) ** 2) / r
    n = np.asarray(Image.fromarray((rng.random((s // 12, s // 12)) * 255).astype(np.uint8)).resize((s, s), Image.BICUBIC)).astype(np.float32) / 255
    n2 = np.asarray(Image.fromarray((rng.random((s // 40, s // 40)) * 255).astype(np.uint8)).resize((s, s), Image.BICUBIC)).astype(np.float32) / 255
    shade = 0.82 + 0.1 * n + 0.12 * n2 - 0.18 * d ** 3
    rgb = np.stack([shade * 236, shade * 238, shade * 250], -1).clip(0, 255)
    a = np.clip((1 - d) * r / 1.5, 0, 1) * 255
    return Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), 'RGBA')

# ---------- footage segments ----------
segs = cfg['segments']; t = cfg['intro']
for s in segs: s['t0'] = t; t += s['dur']; s['t1'] = t
END0 = t; TOTAL = END0 + cfg['end_dur']
MOON = make_moon(cfg.get('moon_r', 230))
def seg_frames(s):
    a, b = s['src_in'], s['src_out']; k = s['dur'] / (b - a); cw, ch, x0, y0 = s['crop']
    fc = (f"[0:v]crop={cw}:{ch}:{x0}:{y0},setpts={k:.4f}*(PTS-STARTPTS),minterpolate=fps={FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,split[a][b];"
          f"[a]scale=-2:{H},crop={W}:{H},boxblur=40:3,eq=brightness=-0.36:contrast=0.85:saturation=1.3,colorchannelmixer=rr=0.8:gg=0.8:bb=0.95[bg];"
          f"[b]scale={W}:{BAND_H}:flags=lanczos,unsharp=5:5:0.7,eq=contrast=1.06:saturation=1.12[fg];"
          f"[bg][fg]overlay=0:{BAND_Y},format=rgb24")
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', str(a), '-t', str(b - a + 0.15), '-i', P(cfg['footage']), '-filter_complex', fc,
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    last = None
    while True:
        r = p.stdout.read(W * H * 3)
        if len(r) < W * H * 3: break
        last = Image.frombytes('RGB', (W, H), r); yield last
    while True: yield last

# ---------- layers ----------
C = cfg
L_pres = glow_layer([{'text': C['presenter'], 'font': 'cinzel-700.ttf', 'size': 40, 'y': 860, 'color': SILVER, 'track': 9, 'glow': (120, 140, 255), 'glow_r': 14},
                     {'text': 'presents', 'font': 'corm-500i.ttf', 'size': 56, 'y': 930, 'color': GOLD}])
L_kick = glow_layer([{'text': C['presenter'] + '  PRESENTS', 'font': 'cinzel-700.ttf', 'size': 38, 'y': 240, 'color': GOLD, 'track': 5}])
L_t1 = glow_layer([{'text': C['title1'], 'font': 'cinzel-700.ttf', 'size': 132, 'y': 315, 'color': SILVER, 'track': 6, 'glow': (150, 170, 255), 'glow_r': 26}])
L_t2 = glow_layer([{'text': C['title2'], 'font': 'Inter-ExtraBold.otf', 'size': 84, 'y': 490, 'color': WHITE, 'track': 22, 'glow': VIOLET, 'glow_r': 22}])
L_live = glow_layer([{'text': 'LIVE ON THE DECKS', 'font': 'cinzel-700.ttf', 'size': 36, 'y': 1225, 'color': GOLD, 'track': 9}])
L_a1 = glow_layer([{'text': C['artists'][0], 'font': 'Inter-ExtraBold.otf', 'size': 104, 'y': 1290, 'color': WHITE, 'track': 6, 'glow': VIOLET, 'glow_r': 24}])
L_amp = glow_layer([{'text': '&', 'font': 'corm-500i.ttf', 'size': 84, 'y': 1405, 'color': GOLD}])
L_a2 = glow_layer([{'text': C['artists'][1], 'font': 'Inter-ExtraBold.otf', 'size': 104, 'y': 1490, 'color': WHITE, 'track': 6, 'glow': CYAN, 'glow_r': 24}])
L_date = glow_layer([{'text': C['date_line'], 'font': 'Inter-ExtraBold.otf', 'size': 74, 'y': 1250, 'color': WHITE, 'track': 4, 'glow': VIOLET, 'glow_r': 18}])
L_time = glow_layer([{'text': C['time_line'], 'font': 'Inter-ExtraBold.otf', 'size': 92, 'y': 1355, 'color': WHITE, 'track': 5, 'glow': CYAN, 'glow_r': 24}])
L_seats = glow_layer([{'text': C['seats_line'], 'font': 'cinzel-700.ttf', 'size': 40, 'y': 1495, 'color': GOLD, 'track': 5}])
logo = Image.open(P(C['assets']['logo'])); LH = 470; LW = int(logo.width * LH / logo.height); LOGO = logo.resize((LW, LH), Image.LANCZOS)
E_title = glow_layer([{'text': C['title1'] + ' ' + C['title2'], 'font': 'cinzel-700.ttf', 'size': 64, 'y': 830, 'color': SILVER, 'track': 4, 'glow': (150, 170, 255), 'glow_r': 16}])
E_art = glow_layer([{'text': C['artists'][0] + '  &  ' + C['artists'][1], 'font': 'Inter-ExtraBold.otf', 'size': 60, 'y': 925, 'color': WHITE, 'track': 3, 'glow': VIOLET, 'glow_r': 18}])
E_when = glow_layer([{'text': C['end_when'], 'font': 'Inter-Bold.otf', 'size': 50, 'y': 1015, 'color': CYAN, 'track': 3}])
def e_book(d, im): pass
E_book = glow_layer([{'text': C['seats_line'], 'font': 'cinzel-700.ttf', 'size': 38, 'y': 1118, 'color': GOLD, 'track': 5},
                     {'text': C['booking_label'], 'font': 'Inter-SemiBold.otf', 'size': 44, 'y': 1182, 'color': SILVER},
                     {'text': C['phone'], 'font': 'Inter-ExtraBold.otf', 'size': 78, 'y': 1240, 'color': WHITE, 'track': 2, 'glow': (120, 140, 255), 'glow_r': 14}])
E_tag = glow_layer([{'text': C['place'], 'font': 'cinzel-700.ttf', 'size': 36, 'y': 1365, 'color': GOLD, 'track': 7},
                    {'text': C['tagline'], 'font': 'corm-500i.ttf', 'size': 62, 'y': 1425, 'color': CREAM}])
rule = Image.new('RGBA', (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(rule)
dr.line([(cx - 240, 1090), (cx - 14, 1090)], fill=GOLD, width=2); dr.line([(cx + 14, 1090), (cx + 240, 1090)], fill=GOLD, width=2)
dr.polygon([(cx, 1083), (cx + 7, 1090), (cx, 1097), (cx - 7, 1090)], fill=GOLD)
frame_line = Image.new('RGBA', (W, H), (0, 0, 0, 0)); fd = ImageDraw.Draw(frame_line)
fd.line([(0, BAND_Y - 2), (W, BAND_Y - 2)], fill=(200, 210, 255, 160), width=2); fd.line([(0, BAND_Y + BAND_H + 1), (W, BAND_Y + BAND_H + 1)], fill=(200, 210, 255, 160), width=2)

# ---------- music ----------
DROP = segs[cfg.get('drop_seg', 1)]['t0']
m = make_edm(TOTAL, DROP, END0, seed=cfg.get('seed', 5), bpm=cfg.get('bpm', 120))
aw = os.path.splitext(P(C['out']))[0] + '_audio.wav'; os.makedirs(os.path.dirname(aw), exist_ok=True)
wavfile.write(aw, SR, (m * 32767).astype(np.int16))
st = subprocess.run(['ffmpeg', '-hide_banner', '-i', aw, '-af', 'ebur128=peak=true', '-f', 'null', '-'], capture_output=True, text=True).stderr
I = float(re.findall(r'I:\s+(-?[\d.]+) LUFS', st)[-1]); pk = float(re.findall(r'Peak:\s+(-?[\d.]+) dBFS', st)[-1])
gn = min(-14 - I, -1.0 - pk); wavfile.write(aw, SR, (np.clip(m * 10 ** (gn / 20), -1, 1) * 32767).astype(np.int16))
B = 60 / cfg.get('bpm', 120); KICKS = np.arange(DROP, END0, B)
def pulse(t):
    if t < DROP or t >= END0: return 0.0
    k = KICKS[KICKS <= t]; return math.exp(-(t - k[-1]) / 0.12) if len(k) else 0.0

# ---------- render ----------
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-i', aw, '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-maxrate', '10M', '-bufsize', '20M', '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', P(C['out'])], stdin=subprocess.PIPE)
gens = {}; cuts = [s['t0'] for s in segs[1:]] + [END0]
for n in range(int(round(TOTAL * FPS))):
    t = n / FPS
    if t < segs[0]['t0']:                                    # intro: night sky + presenter
        fr = sky(t, (cx, 560 - 30 * t, 200))
        fr = put(fr, L_pres, env(t, 0.1, segs[0]['t0'] + 0.2, 0.4), dy=int(20 * (1 - ease(t / 0.5))))
    elif t < END0:
        k = max(i for i, s in enumerate(segs) if s['t0'] <= t + 1e-6); s = segs[k]
        if k not in gens: gens[k] = seg_frames(s)
        for j in list(gens):
            if j < k: gens.pop(j)
        fr = next(gens[k]).convert('RGBA')
        if t < segs[0]['t0'] + 0.5:                          # intro -> footage dissolve
            fr = Image.blend(sky(t, (cx, 560 - 30 * t, 200)).convert('RGB'), fr.convert('RGB'), ease((t - segs[0]['t0']) / 0.5)).convert('RGBA')
        fr.alpha_composite(frame_line)
        T0 = segs[0]['t0']; pu = pulse(t)
        fr = put(fr, L_kick, env(t, T0 + 0.1, END0, 0.4))
        fr = put(fr, L_t1, env(t, T0 + 0.4, END0, 0.5), dy=int(-30 * (1 - ease((t - T0 - 0.4) / 0.6))))
        fr = put(fr, L_t2, env(t, T0 + 1.2, END0, 0.4), dy=int(25 * (1 - ease((t - T0 - 1.2) / 0.5))))
        if pu > 0: fr = put(fr, L_t2, 0.35 * pu)            # title breathes with the kick
        lu0, lu1 = C['lineup_at'], C['date_at']
        fr = put(fr, L_live, env(t, lu0, lu1, 0.3))
        fr = put(fr, L_a1, env(t, lu0 + 0.05, lu1, 0.25), dy=int(40 * (1 - ease((t - lu0) / 0.35))))
        fr = put(fr, L_amp, env(t, lu0 + 0.8, lu1, 0.3))
        fr = put(fr, L_a2, env(t, lu0 + 1.0, lu1, 0.25), dy=int(40 * (1 - ease((t - lu0 - 1.0) / 0.35))))
        fr = put(fr, L_date, env(t, lu1 + 0.05, END0, 0.3), dy=int(30 * (1 - ease((t - lu1) / 0.35))))
        fr = put(fr, L_time, env(t, lu1 + 0.55, END0, 0.3), dy=int(30 * (1 - ease((t - lu1 - 0.5) / 0.35))))
        fr = put(fr, L_seats, env(t, lu1 + 1.4, END0, 0.4))
        for c in cuts[:-1]:                                  # flash on each beat cut
            if 0 <= t - c < 0.25:
                a = math.exp(-(t - c) / 0.07) * 0.45
                fl = Image.new('RGBA', (W, H), (220, 210, 255, int(255 * a))); fr.alpha_composite(fl)
    else:                                                     # end card
        lt = t - END0
        fr = sky(t, (cx, 545, cfg.get('moon_r', 230)))
        if lt < 0.6:
            last = next(gens[len(segs) - 1]).convert('RGBA') if gens else fr
            fr = Image.blend(last.convert('RGB'), fr.convert('RGB'), ease(lt / 0.6)).convert('RGBA')
        if lt < 0.25:
            a = math.exp(-lt / 0.08) * 0.6; fr.alpha_composite(Image.new('RGBA', (W, H), (235, 230, 255, int(255 * a))))
        a = env(lt, 0.2, 99, 0.7); lg = LOGO.copy(); lg.putalpha(lg.split()[3].point(lambda v: int(v * a)))
        sc = 0.92 + 0.08 * ease((lt - 0.2) / 0.9); lw, lh = int(LW * sc), int(LH * sc); lg = lg.resize((lw, lh), Image.BICUBIC)
        fr.alpha_composite(lg, (cx - lw // 2, 545 - lh // 2))
        fr = put(fr, E_title, env(lt, 0.8, 99, 0.4), dy=int(20 * (1 - ease((lt - 0.8) / 0.4))))
        fr = put(fr, E_art, env(lt, 1.2, 99, 0.4), dy=int(20 * (1 - ease((lt - 1.2) / 0.4))))
        fr = put(fr, E_when, env(lt, 1.6, 99, 0.4))
        fr = put(fr, rule, env(lt, 1.9, 99, 0.4))
        fr = put(fr, E_book, env(lt, 2.2, 99, 0.5))
        fr = put(fr, E_tag, env(lt, 3.0, 99, 0.6))
        rem = TOTAL - t
        if rem < 0.4: fr = Image.blend(fr.convert('RGB'), Image.new('RGB', (W, H), 0), 1 - rem / 0.4).convert('RGBA')
    enc.stdin.write(fr.convert('RGB').tobytes())
enc.stdin.close(); enc.wait(); print(f'done {P(C["out"])} {TOTAL:.1f}s (audio {I:.1f} LUFS -> {gn:+.1f} dB)')
