"""Original soundtrack generator for The Chapters reels (no copyright issues)."""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve, lfilter
SR=48000
def make_music(total, groove_start, outro_start, swish_soft=(), swish_page=(), duck=None, seed=11, bpm=100, flute_at=()):
    N=int(SR*total); rng=np.random.default_rng(seed)
    L=np.zeros(N); R=np.zeros(N)
    hz=lambda m:440*2**((m-69)/12)
    def add(sig,t0,pan=0.0,g=1.0):
        i=int(t0*SR)
        if i>=N or i<0: return
        j=min(N,i+len(sig)); s=sig[:j-i]*g
        L[i:j]+=s*np.sqrt((1-pan)/2); R[i:j]+=s*np.sqrt((1+pan)/2)
    lp=lambda x,f,o=2: sosfilt(butter(o,f,'low',fs=SR,output='sos'),x)
    hp=lambda x,f,o=2: sosfilt(butter(o,f,'high',fs=SR,output='sos'),x)
    bp=lambda x,a,b: sosfilt(butter(2,[a,b],'band',fs=SR,output='sos'),x)
    B=60/bpm
    def rhodes(m,dur,v=1.0):
        n=int((dur+1.2)*SR); t=np.arange(n)/SR; f=hz(m)
        mod=np.sin(2*np.pi*f*14*t)*np.exp(-t*9)*1.2
        x=np.sin(2*np.pi*f*t+mod)*np.exp(-t*0.9)+0.35*np.sin(4*np.pi*f*t)*np.exp(-t*2.5)
        return x*np.clip((dur+1.2-t)/1.2,0,1)*np.minimum(1,t/0.004)*0.05*v
    def ks(m,dur=1.2,v=1.0):
        f=hz(m); P=int(SR/f); n=int(dur*SR); exc=np.zeros(n); exc[:P]=lp(rng.standard_normal(P),3500)
        a=np.zeros(P+2); a[0]=1; a[P]=-0.498; a[P+1]=-0.498
        return lfilter([1],a,exc)*0.05*v
    def glock(m,v=1.0):
        n=int(2.0*SR); t=np.arange(n)/SR; f=hz(m)
        x=np.sin(2*np.pi*f*t)*np.exp(-t*2.2)+0.3*np.sin(2*np.pi*f*2.76*t)*np.exp(-t*6)+0.12*np.sin(2*np.pi*f*5.4*t)*np.exp(-t*10)
        return x*np.minimum(1,t/0.001)*0.07*v
    def kick(v=1.0):
        n=int(0.4*SR); t=np.arange(n)/SR; f=50+90*np.exp(-t*35)
        return np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*9)*0.5*v
    def snap(v=1.0):
        n=int(0.25*SR); t=np.arange(n)/SR
        return (bp(rng.standard_normal(n),1200,6000)*np.exp(-t*28)+0.3*np.sin(2*np.pi*190*t)*np.exp(-t*40))*0.2*v
    def shaker(v=1.0):
        n=int(0.12*SR); t=np.arange(n)/SR
        return hp(rng.standard_normal(n),6000)*np.minimum(1,t/0.02)*np.exp(-t*40)*0.05*v
    def bass(m,dur,v=1.0):
        n=int(dur*SR); t=np.arange(n)/SR; f=hz(m)
        return lp(np.sin(2*np.pi*f*t)+0.25*np.sin(4*np.pi*f*t),600)*np.minimum(1,t/0.01)*np.clip((dur-t)/0.08,0,1)*np.exp(-t*1.2)*0.22*v
    def swish(dur,v):
        n=int(dur*SR); t=np.arange(n)/SR
        return lp(bp(rng.standard_normal(n),900,5000),4000)*np.sin(np.pi*t/dur)**2*0.12*v
    CH=[[48,55,59,64],[52,59,62,67],[45,52,55,60],[41,48,52,57]]; BASS=[36,40,33,41]
    nb=int(total/B)+1
    for b in range(0,nb,4):
        t=b*B; c=CH[(b//4)%4]
        if t>=outro_start: break
        v=0.75 if t<groove_start else 1.0
        for k,m in enumerate(c): add(rhodes(m+12,4*B,v),t+k*0.012,pan=-0.25+0.17*k)
        if groove_start<=t:
            bm=BASS[(b//4)%4]; add(bass(bm,2*B),t); add(bass(bm+7,B,0.8),t+2.5*B); add(bass(bm+12,B/2,0.7),t+3.5*B)
    patt=[0,2,1,3,2,1,3,2]
    for b in range(nb*2):
        t=b*B/2
        if t<groove_start-2*B or t>=outro_start: continue
        add(ks(CH[(int(t/B)//4)%4][patt[b%8]]+12,1.2,0.9 if b%2==0 else 0.6),t,pan=0.35)
    for b in range(nb):
        t=b*B
        if groove_start<=t<outro_start-0.05:
            if b%4 in (0,2): add(kick(1.0 if b%4==0 else 0.8),t)
            if b%4==3: add(kick(0.5),t+B/2)
            if b%4 in (1,3): add(snap(),t,pan=-0.1)
            add(shaker(),t,pan=0.4); add(shaker(0.6),t+B/2,pan=0.4)
    for i,m in enumerate([76,79,84,83]): add(glock(m),groove_start-1.8+i*0.3+(0.3 if i==3 else 0),pan=0.2)
    motif=[(0,72),(0.5,76),(1,79),(2,77),(2.5,76),(3,74)]
    base=groove_start+4*B
    while base<outro_start-3:
        for dt,m in motif: add(glock(m+12,0.5),base+dt*B,pan=0.25)
        base+=8*B
    def flute(notes, v=1.0):
        # bansuri-style: sine + soft harmonics, breath noise, vibrato, glides between notes
        tot = max(s + d for s, d, m in notes) * B + 1.0; n = int(tot * SR); t = np.arange(n) / SR
        freq = np.full(n, hz(notes[0][2])); amp = np.zeros(n)
        for s, d, m in notes:
            i0 = int(s * B * SR); i1 = int((s + d) * B * SR); g = int(0.06 * SR)
            prev = freq[i0 - 1] if i0 > 0 else hz(m)
            freq[i0:i0 + g] = np.linspace(prev, hz(m), g); freq[i0 + g:] = hz(m)
            tt = np.arange(i1 - i0) / SR
            amp[i0:i1] = np.maximum(amp[i0:i1], np.minimum(1, tt / 0.07) * np.clip((d * B - tt) / 0.2, 0, 1))
        vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t)
        ph = 2 * np.pi * np.cumsum(freq * vib) / SR
        tone = np.sin(ph) + 0.2 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
        amp = lp(amp, 30, 1)
        return (tone + bp(rng.standard_normal(n), 1500, 6000) * 0.1) * amp * 0.075 * v
    PH = [[(0, 1, 79), (1, .5, 81), (1.5, .5, 79), (2, 2, 76), (4, 1, 74), (5, 1, 76), (6, 2, 79)],
          [(0, 1.5, 81), (1.5, .5, 79), (2, 1, 76), (3, 1, 74), (4, 3, 72), (7, 1, 74)],
          [(0, 1, 76), (1, 1, 79), (2, 1, 81), (3, 1, 84), (4, 2, 83), (6, 2, 79)],
          [(0, 1, 81), (1, 1, 79), (2, 2, 76), (4, 1, 74), (5, 1, 72), (6, 2.5, 72)]]
    for i, ft in enumerate(flute_at): add(flute(PH[i % len(PH)]), ft, pan=0.1)
    for k,m in enumerate([48,55,59,62,64,71]): add(rhodes(m+12,max(3.0,total-outro_start),1.1),outro_start+k*0.03,pan=-0.3+0.12*k)
    add(bass(36,3.0),outro_start); add(kick(0.9),outro_start)
    for i,m in enumerate([79,84,88]): add(glock(m,0.8),outro_start+0.3+i*0.3)
    cr=np.zeros(N); idx=rng.integers(0,N,int(total*25)); cr[idx]=rng.uniform(-1,1,len(idx))
    cr=hp(cr,1500)*0.25+hp(rng.standard_normal(N),3000)*0.004
    L+=cr; R+=np.roll(cr,37)
    n=int(1.8*SR); tt=np.arange(n)/SR; irs=[lp(rng.standard_normal(n),5000)*np.exp(-tt*2.8) for _ in range(2)]
    wL=fftconvolve(L,irs[0])[:N]; wR=fftconvolve(R,irs[1])[:N]; wL/=np.abs(wL).max(); wR/=np.abs(wR).max()
    pk=max(np.abs(L).max(),np.abs(R).max())
    st=np.stack([L/pk+0.22*wL,R/pk+0.22*wR],1); st=hp(st.T,40).T
    st=np.tanh(st*1.4)/np.tanh(1.4); st/=np.abs(st).max()/0.9
    if duck is not None: st*=duck[:N,None]
    # transition sounds are added AFTER ducking so they stay subtle but audible
    sw=np.zeros((N,2))
    for t in swish_soft:
        s=swish(0.45,0.25); i=int((t-0.22)*SR); j=min(N,i+len(s)); sw[i:j,0]+=s[:j-i]*0.8; sw[i:j,1]+=s[:j-i]
    for t in swish_page:
        s=swish(0.5,0.45); i=int((t-0.2)*SR); j=min(N,i+len(s)); sw[i:j,0]+=s[:j-i]; sw[i:j,1]+=s[:j-i]*0.8
    st=st+sw
    t=np.arange(N)/SR; st*=(np.minimum(1,t/0.05)*np.clip((total-t)/1.8,0,1))[:,None]
    return st
