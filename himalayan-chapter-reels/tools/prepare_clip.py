"""Prepare a new guest clip for a daily Chapter reel.

usage: python tools/prepare_clip.py footage/ch06_something.mp4 [--num 06] [--title "..."]

1. Cleans the voice -> footage/clean/<name>.wav (48 kHz): 80 Hz high-pass, DeepFilterNet AI noise
   removal (delay-compensated, keeps lip sync), gentle EQ + compression, -14 LUFS.
   Falls back to ffmpeg's afftdn denoiser if the deep-filter binary isn't installed.
2. Transcribes with Whisper (original language + English translation) and prints word timings.
3. Writes episodes/daily_ch<NUM>.json, a draft episode with the clip and auto subtitles.
   Edit the 'parts' (which seconds to use) and subtitle text, then render:
   python engine/reel.py episodes/daily_ch<NUM>.json
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('video'); ap.add_argument('--num', default='06')
ap.add_argument('--title', default='Untitled Chapter'); ap.add_argument('--model', default='medium')
a = ap.parse_args()
name = os.path.splitext(os.path.basename(a.video))[0]
clean = os.path.join(ROOT, 'footage', 'clean', name + '.wav'); os.makedirs(os.path.dirname(clean), exist_ok=True)
POST = ("equalizer=f=220:t=q:w=1.2:g=-2,equalizer=f=3200:t=q:w=1.0:g=3,equalizer=f=7000:t=h:w=0.7:g=1.5,"
        "acompressor=threshold=-24dB:ratio=3:attack=8:release=150:makeup=2")
def run(*c): subprocess.run([str(x) for x in c], check=True)
def loudnorm(src, chain, dst):
    r = subprocess.run(['ffmpeg', '-hide_banner', '-i', src, '-af', f'{chain},loudnorm=I=-14:TP=-1.5:LRA=9:print_format=json',
                        '-f', 'null', '-'], capture_output=True, text=True).stderr
    m = json.loads(r[r.rindex('{'):r.rindex('}') + 1])
    ln = (f"loudnorm=I=-14:TP=-1.5:LRA=9:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    run('ffmpeg', '-v', 'error', '-y', '-i', src, '-af', f'{chain},{ln},aresample=48000', '-ac', '1', '-c:a', 'pcm_s16le', dst)
df = shutil.which('deep-filter') or os.path.expanduser('~/.video-cleanup/bin/deep-filter')
with tempfile.TemporaryDirectory() as tmp:
    wav = os.path.join(tmp, 'speech.wav')
    run('ffmpeg', '-v', 'error', '-y', '-i', a.video, '-map', '0:a:0', '-af', 'pan=mono|c0=0.5*c0+0.5*c1,highpass=f=80:poles=2',
        '-ar', '48000', '-c:a', 'pcm_s16le', wav)
    if os.path.exists(df):
        run(df, '-D', '-a', '35', '-o', os.path.join(tmp, 'df'), wav); src = os.path.join(tmp, 'df', 'speech.wav'); chain = POST
    else:
        print('deep-filter not found: using ffmpeg afftdn (install DeepFilterNet for better results)')
        src = wav; chain = 'afftdn=nr=12:nf=-40,' + POST
    loudnorm(src, chain, clean)
print('clean voice ->', clean)

from faster_whisper import WhisperModel
audio = np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-i', clean, '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'],
                                     capture_output=True).stdout, np.float32)
m = WhisperModel(a.model, device='cpu', compute_type='int8')
res = {}
for task in ('transcribe', 'translate'):
    segs, info = m.transcribe(audio, task=task, word_timestamps=True, vad_filter=True, beam_size=5)
    res[task] = list(segs)
print(f'language: {info.language} ({info.language_probability:.2f})')
for task in ('transcribe', 'translate'):
    print(f'--- {task}')
    for s in res[task]:
        print(f'  {s.start:5.2f}-{s.end:5.2f}  {s.text.strip()}')
        print('      ' + ' '.join(f'{w.word.strip()}@{w.start:.2f}' for w in s.words))
segs = res['translate'] if info.language != 'en' else res['transcribe']
subs = [[round(s.start, 2), round(s.end, 2), s.text.strip(), ''] for s in segs]
start = max(0, subs[0][0] - 0.3) if subs else 0; end = (subs[-1][1] + 0.4) if subs else 8
ep = {'out': f'../out/THC_Daily_Chapter{a.num}.mp4', 'assets': {'fonts': '../assets/fonts', 'logo': '../assets/logo.png'}, 'seed': int(a.num) * 7,
      'shots': [{'type': 'chapter', 'num': a.num, 'title': a.title, 'src': f'../footage/{name}{os.path.splitext(a.video)[1]}',
                 'voice': f'../footage/clean/{name}.wav', 'pre': '', 'parts': [[round(start, 2), round(end, 2)]], 'subs': subs},
                {'type': 'end', 'dur': 4.0, 'line1': 'Every guest is a chapter.', 'line2': 'COME WRITE YOURS.',
                 'info': ['Rooftop views  ·  Live music nights', 'Board games, books & slow evenings'],
                 'place': 'BALAKNATH RD  ·  TAPOVAN  ·  RISHIKESH'}]}
outp = os.path.join(ROOT, 'episodes', f'daily_ch{a.num}.json')
json.dump(ep, open(outp, 'w'), ensure_ascii=False, indent=1)
print('draft episode ->', outp)
