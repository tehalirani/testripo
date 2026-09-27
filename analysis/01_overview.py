import numpy as np, librosa, soundfile as sf
sr=22050
d={}
for n in ['studio','live']:
    y,_=sf.read(n+'.wav'); d[n]=y
    m=librosa.resample(y.mean(1).astype(np.float32),orig_sr=44100,target_sr=sr)
    d[n+'_m']=m
for n in ['studio','live']:
    y=d[n]; m=d[n+'_m']; L,R=y[:,0],y[:,1]
    mid=(L+R)/2; side=(L-R)/2
    print(f"== {n}: dur {len(m)/sr:.1f}s  RMS {20*np.log10(np.sqrt((m**2).mean())):.1f}dB  side/mid {20*np.log10(np.std(side)/np.std(mid)):.1f}dB  LRcorr {np.corrcoef(L,R)[0,1]:.3f}")
    tempo,_=librosa.beat.beat_track(y=m,sr=sr); print("  tempo",np.round(tempo,1))
    S=np.abs(librosa.stft(m,n_fft=4096))**2; f=librosa.fft_frequencies(sr=sr,n_fft=4096)
    tot=S.sum()
    bands=[(20,60),(60,250),(250,1000),(1000,4000),(4000,8000),(8000,11000)]
    print("  bands%:",[f"{a}-{b}:{100*S[(f>=a)&(f<b)].sum()/tot:.1f}" for a,b in bands])
    chroma=librosa.feature.chroma_cqt(y=m,sr=sr).mean(1); print("  key-chroma argmax",librosa.midi_to_note(60+np.argmax(chroma))[:-1])
    rms=librosa.feature.rms(y=m,hop_length=sr)[0]
    print("  per-sec RMS dB:",' '.join(f"{20*np.log10(x+1e-9):.0f}" for x in rms))
    fl=librosa.feature.spectral_flatness(y=m,hop_length=sr)[0]
    print("  per-sec flatness(x100):",' '.join(f"{100*x:.0f}" for x in fl))
# alignment via chroma DTW
h=2048
cs=librosa.feature.chroma_cqt(y=d['studio_m'],sr=sr,hop_length=h)
cl=librosa.feature.chroma_cqt(y=d['live_m'],sr=sr,hop_length=h)
D,wp=librosa.sequence.dtw(X=cs,Y=cl,subseq=True)
wp=wp[::-1]*h/sr
print("DTW studio->live (sample points):")
for i in range(0,len(wp),max(1,len(wp)//25)): print(f"  s {wp[i,0]:6.1f} -> l {wp[i,1]:6.1f}")
# pitch shift test: compare chroma shifts
cm_s=cs.mean(1); cm_l=cl.mean(1)
print("chroma shift corr:",[round(float(np.corrcoef(cm_s,np.roll(cm_l,k))[0,1]),2) for k in range(12)])
