import numpy as np, soundfile as sf, librosa
from scipy.signal import csd, welch, stft, istft
s,_=sf.read('studio.wav'); l,_=sf.read('live.wav'); sr=44100
lag=int(round(3.2993*sr)); n=len(s)
la=l[lag:lag+n]
for ch,name in [(None,'mid'),]:
    x=s.mean(1); y=la.mean(1)
    f,Pxy=csd(x,y,sr,nperseg=8192); _,Pxx=welch(x,sr,nperseg=8192); _,Pyy=welch(y,sr,nperseg=8192)
    H=np.abs(Pxy)/Pxx; coh=np.abs(Pxy)**2/(Pxx*Pyy)
    print("freq  |H|dB  coherence  live/studio power dB")
    for fr in [40,60,80,100,150,200,300,500,800,1000,1500,2000,3000,4000,6000,8000,10000,12000,15000,18000]:
        i=np.argmin(abs(f-fr)); print(f"{fr:6d} {20*np.log10(H[i]):6.1f} {coh[i]:.2f} {10*np.log10(Pyy[i]/Pxx[i]):6.1f}")
# residual: per-freq wiener on STFT
f,t,X=stft(s.mean(1),sr,nperseg=4096); _,_,Y=stft(la.mean(1),sr,nperseg=4096)
G=(Y*np.conj(X)).mean(1)/((np.abs(X)**2).mean(1)+1e-12)
Rz=Y-G[:,None]*X
_,r=istft(Rz,sr,nperseg=4096)
sf.write('residual.wav',r.astype(np.float32),sr)
print("residual/live energy dB:",10*np.log10((np.abs(Rz)**2).sum()/(np.abs(Y)**2).sum()))
# residual flatness per 5s
fl=librosa.feature.spectral_flatness(y=r.astype(np.float32),hop_length=sr*5)[0]
print("residual flatness x100 per5s:",np.round(fl*100).astype(int))
cen=librosa.feature.spectral_centroid(y=r.astype(np.float32),sr=sr,hop_length=sr*5)[0]
print("residual centroid per5s:",np.round(cen).astype(int))
# reverb estimate: decay after strong transients - compare energy envelope smoothness
def edc_ratio(x):
    e=librosa.feature.rms(y=x.astype(np.float32),frame_length=2048,hop_length=512)[0]
    d=np.diff(20*np.log10(e+1e-9)); return np.percentile(d,5)
print("fast-decay 5th pct dB/frame studio",edc_ratio(s.mean(1)),"live",edc_ratio(la.mean(1)))
# stereo per band
for nm,a in [('studio',s),('live',la)]:
    out=[]
    for lo,hi in [(20,150),(150,1000),(1000,5000),(5000,16000)]:
        from scipy.signal import butter,sosfilt
        so=butter(4,[lo,hi],'band',fs=sr,output='sos'); L=sosfilt(so,a[:,0]);R=sosfilt(so,a[:,1])
        out.append(f"{lo}-{hi}:{20*np.log10(np.std(L-R)/np.std(L+R)):.1f}")
    print(nm,"side/mid by band",out)
