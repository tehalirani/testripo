import numpy as np, soundfile as sf, librosa
from scipy.signal import stft
sr=44100; lag=int(round(3.2993*sr))
l,_=sf.read('live.wav'); lm=l.mean(1).astype(np.float32)
# 1) intro onsets
seg=lm[:int(4.5*sr)]
on=librosa.onset.onset_detect(y=seg,sr=sr,units='time',backtrack=False)
env=librosa.onset.onset_strength(y=seg,sr=sr)
print("intro onsets (s):",np.round(on,3))
print("beat period at 161.5 BPM:",round(60/161.5,3))
for t in on[:10]:
    x=lm[int(t*sr):int((t+0.08)*sr)]
    S=np.abs(np.fft.rfft(x*np.hanning(len(x)))); f=np.fft.rfftfreq(len(x),1/sr)
    print(f"  {t:.3f}s centroid {np.sum(f*S)/np.sum(S):.0f}Hz level {20*np.log10(np.sqrt((x**2).mean())+1e-9):.0f}dB")
# 2) extra vocal content: live vocal stem minus best-fit studio vocal
def ld(n): x,_=sf.read(n); return x.mean(1)
a=ld('sep/studio_(Vocals)_UVR-MDX-NET-Voc_FT.wav'); b=ld('sep/live_(Vocals)_UVR-MDX-NET-Voc_FT.wav')
ba=np.zeros_like(a); seg=b[lag:lag+len(a)]; ba[:len(seg)]=seg
f,t,A=stft(a,sr,nperseg=4096); _,_,B=stft(ba,sr,nperseg=4096)
G=(B*np.conj(A)).mean(1)/((np.abs(A)**2).mean(1)+1e-12)
R=B-G[:,None]*A
fr=sr/2048
print("live-extra vocal energy (dB rel. live vocal) per 2s, studio time -> live time:")
out=[]
for i in range(0,R.shape[1]-int(2*fr),int(2*fr)):
    r=(np.abs(R[:,i:i+int(2*fr)])**2).sum(); bb=(np.abs(B[:,i:i+int(2*fr)])**2).sum()+1e-12
    out.append(f"{t[i]+3.3:5.0f}s:{10*np.log10(r/bb+1e-12):5.1f}")
print('\n'.join(' '.join(out[k:k+8]) for k in range(0,len(out),8)))
# tail vocal energy
v=b; e=[20*np.log10(np.sqrt((v[int(s*sr):int((s+2)*sr)]**2).mean())+1e-9) for s in range(0,int(len(v)/sr)-2,2)]
print("live vocal-stem level per 2s:",' '.join(f"{x:.0f}" for x in e))
