import numpy as np, soundfile as sf, librosa
from scipy.signal import correlate, csd, welch
sr=44100; lag=int(round(3.2993*sr))
def ld(n): x,_=sf.read(n); return x.mean(1)
for part in ['Vocals','Instrumental']:
    a=ld(f'studio_({part})_UVR-MDX-NET-Voc_FT.wav'); b=ld(f'live_({part})_UVR-MDX-NET-Voc_FT.wav')
    b2=b[lag:lag+len(a)]; n=min(len(a),len(b2)); a=a[:n]; b2=b2[:n]
    cs=[]
    for t in range(0,n-4*sr,4*sr):
        x=a[t:t+4*sr]; y=b2[t:t+4*sr]
        if np.std(x)<1e-3 or np.std(y)<1e-3: cs.append('  -'); continue
        # allow small lag
        c=correlate(y,x,'same',method='fft'); m=len(c)//2; w=c[m-200:m+200]
        cs.append(f"{np.max(np.abs(w))/(np.linalg.norm(x)*np.linalg.norm(y)):.2f}")
    print(part,"corr per 4s:",' '.join(cs))
    f,Pab=csd(a,b2,sr,nperseg=8192); _,Paa=welch(a,sr,nperseg=8192); _,Pbb=welch(b2,sr,nperseg=8192)
    coh=np.abs(Pab)**2/(Paa*Pbb)
    print("  coherence @",[f"{fr}:{coh[np.argmin(abs(f-fr))]:.2f}" for fr in [100,300,1000,3000,8000]])
    print("  level live/studio dB:",round(10*np.log10(Pbb.sum()/Paa.sum()),1))
# vocal: reverb? tempo? pitch
for n in ['studio','live']:
    v=ld(f'{n}_(Vocals)_UVR-MDX-NET-Voc_FT.wav').astype(np.float32)
    v=librosa.resample(v,orig_sr=sr,target_sr=16000)
    f0,vf,_=librosa.pyin(v[16000*20:16000*80],fmin=80,fmax=800,sr=16000)
    print(n,"vocal median f0",np.nanmedian(f0), "note",librosa.hz_to_note(np.nanmedian(f0)))
