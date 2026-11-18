"""The Chapters reel engine: build a branded vertical reel from a JSON episode config.
usage: python reel.py episode.json
"""
import json, math, subprocess, sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from music import make_music, SR

cfg = json.load(open(sys.argv[1]))
_base = os.path.dirname(os.path.abspath(sys.argv[1]))
P = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(_base, x))
cfg['out'] = P(cfg['out']); os.makedirs(os.path.dirname(cfg['out']), exist_ok=True)
cfg['assets'] = {k: P(v) for k, v in cfg['assets'].items()}
for _sh in cfg['shots']:
    for _k in ('src', 'voice'):
        if _sh.get(_k): _sh[_k] = P(_sh[_k])
A = cfg['assets']
W, H, FPS = 1080, 1920, 30
XF, XJ, PAGE = 0.35, 0.18, 0.45          # dissolve, jump-cut dissolve, page-turn durations
CREAM=(247,240,222); LGOLD=(240,212,150); GOLD=(170,122,48); GREEN=(22,56,38); BOX=(16,40,28)
F = lambda n, s: ImageFont.truetype(os.path.join(A['fonts'], n), s)
INTER = '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'
GRADE = ("curves=all='0/0 0.6/0.56 1/0.93',eq=contrast=1.04:saturation=0.97,"
         "colorbalance=rm=0.03:bm=-0.03:rh=0.02:bh=-0.02,unsharp=5:5:0.5,noise=alls=5:allf=t,vignette=PI/6")
cx = W // 2
ease = lambda p: 1 - (1 - min(max(p, 0), 1)) ** 3
def env(t, a, b, fi=0.35, fo=0.25): return float(np.clip(min((t - a) / fi, (b - t) / fo), 0, 1))

# ---------------- drawing helpers ----------------
def tw(d, txt, font, track): return sum(d.textlength(c, font=font) for c in txt) + track * (len(txt) - 1)
def spaced(d, x, y, txt, font, fill, track):
    x = x - tw(d, txt, font, track) / 2
    for c in txt: d.text((x, y), c, font=font, fill=fill); x += d.textlength(c, font=font) + track
def layer(build, shadow=True):
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im); build(d, im)
    if shadow:
        sh = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        sh.putalpha(im.split()[3].filter(ImageFilter.GaussianBlur(12)).point(lambda v: int(min(255, v * 1.1))))
        o = Image.new('RGBA', (W, H), (0, 0, 0, 0)); o.alpha_composite(sh, (0, 5)); o.alpha_composite(im); im = o
    return np.asarray(im).astype(np.float32) / 255
def blend(f, L, a=1.0, dy=0):
    if a <= 0: return f
    if dy:
        L2 = np.zeros_like(L)
        if dy > 0: L2[dy:] = L[:-dy]
        else: L2[:dy] = L[-dy:]
        L = L2
    al = L[..., 3:4] * a
    return f * (1 - al) + L[..., :3] * al
def rule(d, x, y, half, col):
    d.line([(x - half, y), (x - 16, y)], fill=col, width=3); d.line([(x + 16, y), (x + half, y)], fill=col, width=3)
    d.polygon([(x, y - 8), (x + 8, y), (x, y + 8), (x - 8, y)], fill=col)
rng = np.random.default_rng(5)
def make_paper():
    base = np.ones((H, W, 3), np.float32) * np.array([243, 234, 212], np.float32) / 255
    lo = np.asarray(Image.fromarray((rng.random((H // 40, W // 40)) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
    base *= (0.965 + 0.05 * lo)[..., None]; base += (rng.normal(0, 1, (H, W)).astype(np.float32) * 0.012)[..., None]
    yy, xx = np.mgrid[0:H, 0:W]; r = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    return np.clip(base * (1.04 - 0.32 * r ** 2)[..., None], 0, 1)
PAPER = make_paper()
yy = np.linspace(0, 1, H)[:, None, None]
TOPSCRIM = np.clip(1 - yy / 0.32, 0, 1) ** 1.5 * 0.6
BOTSCRIM = np.clip((yy - 0.55) / 0.45, 0, 1) ** 1.5 * 0.45

def sub_layer(text, hl=''):
    """Subtitle: brand-green rounded box, white text, highlight phrase in gold. Avoids Reels side icons."""
    font = F(INTER, 54) if os.path.isabs(INTER) else None
    font = ImageFont.truetype(INTER, 54)
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    words = text.split(); lines = []; cur = []
    for w in words:
        if d.textlength(' '.join(cur + [w]), font=font) > 800 and cur: lines.append(cur); cur = [w]
        else: cur.append(w)
    lines.append(cur)
    hlw = set(x.strip('.,!?…').lower() for x in hl.split())
    lh = 72; tot = lh * len(lines); y0 = 1390 - tot / 2
    wmax = max(d.textlength(' '.join(l), font=font) for l in lines)
    d.rounded_rectangle([cx - wmax / 2 - 34, y0 - 22, cx + wmax / 2 + 34, y0 + tot + 14], radius=26, fill=BOX + (205,))
    for i, l in enumerate(lines):
        x = cx - d.textlength(' '.join(l), font=font) / 2; y = y0 + i * lh
        for w in l:
            col = LGOLD if w.strip('.,!?…').lower() in hlw else (255, 255, 255)
            d.text((x, y), w, font=font, fill=col); x += d.textlength(w + ' ', font=font)
    return np.asarray(im).astype(np.float32) / 255

# ---------------- timeline ----------------
clips = []   # each: dict(kind, t0, t1, ...)
t = 0.0
for si, sh in enumerate(cfg['shots']):
    if sh['type'] in ('hook', 'chapter', 'broll'):
        parts = sh.get('parts') or [[sh['in'], sh['out']]]
        for pi, (a, b) in enumerate(parts):
            clips.append(dict(kind='foot', shot=si, part=pi, src=sh['src'], a=a, b=b, pre=sh.get('pre', ''),
                              t0=t, t1=t + (b - a)))
            t += b - a
    else:
        clips.append(dict(kind=sh['type'], shot=si, t0=t, t1=t + sh['dur'])); t += sh['dur']
TOTAL = t
# transitions: page turn into card/end, dissolve otherwise
for i, c in enumerate(clips):
    c['tin'] = None if i == 0 else ('page' if c['kind'] in ('card', 'end') else ('jump' if clips[i-1]['shot'] == c['shot'] else 'dis'))
for i, c in enumerate(clips):
    nxt = clips[i + 1]['tin'] if i + 1 < len(clips) else None
    c['xin'] = {'dis': XF, 'jump': XJ}.get(c['tin'], 0)
    c['xout'] = {'dis': XF, 'jump': XJ}.get(nxt, 0)
    c['tail'] = PAGE if nxt == 'page' else 0          # keep rendering under the incoming page
    c['w0'] = c['t0'] - c['xin'] / 2; c['w1'] = c['t1'] + c['xout'] / 2 + c['tail'] + 2 / FPS
shot_span = {}
for c in clips: s = shot_span.setdefault(c['shot'], [c['t0'], c['t1']]); s[1] = c['t1']

# ---------------- per-shot overlay layers ----------------
LAY = {}
for si, sh in enumerate(cfg['shots']):
    if sh['type'] == 'hook':
        def h1(d, im, sh=sh): spaced(d, cx, 760, sh['kicker'], F('cinzel-700.ttf', 46), CREAM, 9)
        def h2(d, im, sh=sh):
            for i, l in enumerate(sh['lines']): d.text((cx, 900 + 120 * i), l, font=F('corm-500i.ttf', 112), fill=CREAM, anchor='mm')
        LAY[si] = (layer(h1), layer(h2))
    elif sh['type'] == 'chapter':
        def b(d, im, sh=sh):
            f = F('cinzel-700.ttf', 40); txt = f"CHAPTER {sh['num']}"; w = tw(d, txt, f, 10)
            spaced(d, cx, 250, txt, f, LGOLD, 10)
            d.line([(cx - w / 2 - 150, 275), (cx - w / 2 - 30, 275)], fill=LGOLD, width=2)
            d.line([(cx + w / 2 + 30, 275), (cx + w / 2 + 150, 275)], fill=LGOLD, width=2)
            d.text((cx, 380), sh['title'], font=F('corm-500i.ttf', 84), fill=CREAM, anchor='mm')
        subs = []
        for (s, e, txt, hl) in sh.get('subs', []):
            subs.append((s, e, sub_layer(txt, hl)))
        LAY[si] = (layer(b), subs)
    elif sh['type'] == 'card':
        def c1(d, im, sh=sh): d.text((cx, 820), sh['small'], font=F('corm-500i.ttf', 110), fill=GREEN, anchor='mm')
        def c2(d, im, sh=sh):
            spaced(d, cx, 890, sh['big'], F('cinzel-700.ttf', sh.get('big_size', 150)), GREEN, 6)
            rule(d, cx, 1110, 250, GOLD)
            if sh.get('foot'): d.text((cx, 1180), sh['foot'], font=F('corm-500i.ttf', 64), fill=GOLD, anchor='mm') if sh.get('foot_italic') else spaced(d, cx, 1150, sh['foot'], F('cinzel-700.ttf', 36), GOLD, 8)
        LAY[si] = (layer(c1, False), layer(c2, False))
    elif sh['type'] == 'end':
        logo = Image.open(A['logo']); LHh = 640; lw = int(logo.width * LHh / logo.height)
        L0 = Image.new('RGBA', (W, H), (0, 0, 0, 0)); lg = logo.resize((lw, LHh), Image.LANCZOS)
        shd = Image.new('RGBA', lg.size, (0, 0, 0, 0)); shd.putalpha(lg.split()[3].point(lambda v: int(v * 0.35)))
        L0.alpha_composite(shd.filter(ImageFilter.GaussianBlur(16)), (cx - lw // 2 + 4, 186)); L0.alpha_composite(lg, (cx - lw // 2, 170))
        def e1(d, im, sh=sh): d.text((cx, 940), sh['line1'], font=F('corm-500i.ttf', 82), fill=GREEN, anchor='mm')
        def e2(d, im, sh=sh): spaced(d, cx, 1010, sh['line2'], F('cinzel-700.ttf', 74), GREEN, 5); rule(d, cx, 1140, 230, GOLD)
        def e3(d, im, sh=sh):
            for i, l in enumerate(sh['info']): d.text((cx, 1205 + 65 * i), l, font=F('corm-600.ttf', 50), fill=GREEN, anchor='mm')
            spaced(d, cx, 1210 + 65 * len(sh['info']), sh['place'], F('cinzel-700.ttf', 36), GOLD, 8)
        LAY[si] = (np.asarray(L0).astype(np.float32) / 255, layer(e1, False), layer(e2, False), layer(e3, False))

# ---------------- frame sources ----------------
class Foot:
    def __init__(self, c):
        self.c = c; ss = c['a'] - c['xin'] / 2; self.lead = 0
        if ss < 0: self.lead = -ss; ss = 0
        dur = c['w1'] - c['w0'] + 0.2
        vf = f"fps={FPS},{c['pre']}scale={W}:{H}:flags=lanczos,setsar=1,{GRADE}"
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', str(ss), '-t', str(dur), '-i', c['src'], '-vf', vf,
                                   '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.idx = -1; self.last = None
    def get(self, t):
        want = int(round((t - self.c['w0'] - self.lead) * FPS))
        while self.idx < want:
            b = self.p.stdout.read(W * H * 3)
            if len(b) < W * H * 3: break
            self.last = np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32) / 255; self.idx += 1
        if self.last is None:
            b = self.p.stdout.read(W * H * 3); self.last = np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32) / 255; self.idx = 0
        return self.last
    def close(self):
        try: self.p.stdout.close(); self.p.kill()
        except Exception: pass

def render_clip(c, src, t):
    sh = cfg['shots'][c['shot']]; si = c['shot']; s0, s1 = shot_span[si]; lt = t - s0
    if c['kind'] == 'foot':
        f = src.get(t).copy()
        if sh['type'] == 'hook':
            l1, l2 = LAY[si]; f = f * (1 - sh.get('dim', 0.42))
            f = blend(f, l1, env(lt, 0.15, 99), dy=int(20 * (1 - ease((lt - 0.15) / 0.4))))
            f = blend(f, l2, env(lt, 1.2, 99), dy=int(30 * (1 - ease((lt - 1.2) / 0.4))))
        elif sh['type'] == 'chapter':
            lab, subs = LAY[si]
            f = f * (1 - TOPSCRIM)
            if subs: f = f * (1 - BOTSCRIM)
            f = blend(f, lab, env(lt, 0.25, (s1 - s0) + 1, 0.4), dy=int(28 * (1 - ease((lt - 0.25) / 0.4))))
            # subtitles: source times -> this clip's timeline
            src_t = c['a'] + (t - c['t0'])
            for (a, b, L) in subs:
                if a - 0.05 <= src_t <= b + 0.1 and c['a'] - 0.3 <= src_t <= c['b'] + 0.3:
                    e = env(src_t, a - 0.05, b + 0.1, 0.12, 0.12)
                    f = blend(f, L, e, dy=int(14 * (1 - ease((src_t - a + 0.05) / 0.15))))
        return f
    if c['kind'] == 'card':
        l1, l2 = LAY[si]; pg = PAPER.copy()
        pg = blend(pg, l1, env(lt, 0.25, 99, 0.3)); pg = blend(pg, l2, env(lt, 0.6, 99, 0.3), dy=int(25 * (1 - ease((lt - 0.6) / 0.3))))
        return pg
    if c['kind'] == 'end':
        lo, l1, l2, l3 = LAY[si]; pg = PAPER.copy()
        pg = blend(pg, lo, env(lt, 0.3, 99, 0.6), dy=int(30 * (1 - ease((lt - 0.3) / 0.6))))
        pg = blend(pg, l1, env(lt, 1.0, 99, 0.4)); pg = blend(pg, l2, env(lt, 1.6, 99, 0.4), dy=int(20 * (1 - ease((lt - 1.6) / 0.4))))
        pg = blend(pg, l3, env(lt, 2.3, 99, 0.5))
        rem = TOTAL - t
        return pg * min(1, rem / 0.6)

def page_in(under, page, p):
    e = ease(p); x = int(W * (1 - e)); out = under.copy()
    if x < W:
        out[:, x:] = page[:, :W - x]
        x0 = max(0, x - 70)
        if x > 0: out[:, x0:x] *= np.linspace(1, 0.5, x - x0)[None, :, None]
    return out

# ---------------- audio ----------------
def build_audio():
    N = int(TOTAL * SR); voice = np.zeros((N, 2), np.float32); duck = np.ones(N, np.float32)
    for c in clips:
        sh = cfg['shots'][c['shot']]
        if c['kind'] != 'foot' or not sh.get('voice'): continue
        sr, x = wavfile.read(sh['voice']); x = x.astype(np.float32) / 32767
        if x.ndim == 1: x = np.stack([x, x], 1)
        if sr != SR: raise SystemExit('voice wav must be 48 kHz')
        a = max(0, c['a'] - c['xin'] / 2); b = c['b'] + c['xout'] / 2
        seg = x[int(a * SR):int(b * SR)].copy(); n = len(seg); tt = np.arange(n) / SR + (c['w0'] + (a - (c['a'] - c['xin'] / 2)))
        g = np.ones(n)
        if c['xin']: g *= np.clip((tt - c['w0']) / c['xin'], 0, 1)
        if c['xout']: g *= np.clip((c['t1'] + c['xout'] / 2 - tt) / c['xout'], 0, 1)
        seg *= g[:, None]; i = int(tt[0] * SR); j = min(N, i + n); voice[i:j] += seg[:j - i]
        duck[int(c['t0'] * SR):int(c['t1'] * SR)] = cfg.get('music_under_voice', 0.24)
    k = int(0.3 * SR); duck = np.convolve(duck, np.ones(k) / k, mode='same')
    chap = [c for c in clips if c['kind'] == 'foot' and cfg['shots'][c['shot']]['type'] == 'chapter']
    groove = chap[0]['t0'] if chap else 3.0
    end = [c for c in clips if c['kind'] == 'end']; outro = end[0]['t0'] if end else TOTAL - 4
    # transition swish sounds are off by default; set "swish": true in an episode to bring them back
    sw = cfg.get('swish', False)
    soft = [c['t0'] for c in clips if c['tin'] == 'dis'] if sw else []
    page = [c['t0'] for c in clips if c['tin'] == 'page'] if sw else []
    m = make_music(TOTAL, groove, outro, soft, page, duck, seed=cfg.get('seed', 11))
    mix = m * 0.9 + voice * cfg.get('voice_gain', 1.0)
    mix /= max(1, np.abs(mix).max() / 0.95)
    out = cfg['out'].rsplit('.', 1)[0] + '_audio.wav'
    wavfile.write(out, SR, (mix * 32767).astype(np.int16))
    # normalise to -14 LUFS (social media level), keep true peak under -1.5 dBFS
    r = subprocess.run(['ffmpeg', '-hide_banner', '-i', out, '-af', 'ebur128=peak=true', '-f', 'null', '-'], capture_output=True, text=True).stderr
    import re
    I = float(re.findall(r'I:\s+(-?[\d.]+) LUFS', r)[-1]); pk = float(re.findall(r'Peak:\s+(-?[\d.]+) dBFS', r)[-1])
    gain = min(-14 - I, -1.5 - pk)
    mix *= 10 ** (gain / 20)
    wavfile.write(out, SR, (np.clip(mix, -1, 1) * 32767).astype(np.int16)); print(f'audio {I:.1f} LUFS -> gain {gain:+.1f} dB')
    return out

audio = build_audio()
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-i', audio, '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-maxrate', '9M', '-bufsize', '18M',
                        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', cfg['out']],
                       stdin=subprocess.PIPE)
srcs = {}
for n in range(int(round(TOTAL * FPS))):
    t = n / FPS
    act = [i for i, c in enumerate(clips) if c['w0'] - 1e-6 <= t < c['w1']]
    for i in act:
        if clips[i]['kind'] == 'foot' and i not in srcs: srcs[i] = Foot(clips[i])
    for i in list(srcs):
        if i not in act: srcs.pop(i).close()
    k = max(i for i, c in enumerate(clips) if c['t0'] <= t + 1e-6)
    c = clips[k]
    f = render_clip(c, srcs.get(k), t)
    if c['tin'] == 'page' and t - c['t0'] < PAGE:
        f = page_in(render_clip(clips[k - 1], srcs.get(k - 1), t), f, (t - c['t0']) / PAGE)
    elif c['xin'] and t < c['t0'] + c['xin'] / 2:
        w = 0.5 + (t - c['t0']) / c['xin']; f = render_clip(clips[k - 1], srcs.get(k - 1), t) * (1 - w) + f * w
    elif c['xout'] and k + 1 < len(clips) and t >= c['t1'] - c['xout'] / 2:
        w = 0.5 - (c['t1'] - t) / c['xout']; f = f * (1 - w) + render_clip(clips[k + 1], srcs.get(k + 1), t) * w
    enc.stdin.write((np.clip(f, 0, 1) * 255).astype(np.uint8).tobytes())
for s in srcs.values(): s.close()
enc.stdin.close(); enc.wait()
print(f'done {cfg["out"]}  {TOTAL:.1f}s')
