import numpy as np, soundfile as sf, librosa
sr=22050
def ld(n):
    y,_=sf.read(n); return librosa.resample(y.mean(1).astype(np.float32),orig_sr=44100,target_sr=sr)
m=ld('reallive.wav'); v=ld('sep/reallive_(Vocals)_UVR-MDX-NET-Voc_FT.wav'); i=ld('sep/reallive_(Instrumental)_UVR-MDX-NET-Voc_FT.wav')
st=ld('song.wav')
t,_=librosa.beat.beat_track(y=i,sr=sr); print("tempo",t)
# align studio to live via chroma DTW
h=2048
cs=librosa.feature.chroma_cqt(y=st,sr=sr,hop_length=h); cl=librosa.feature.chroma_cqt(y=m,sr=sr,hop_length=h)
D,wp=librosa.sequence.dtw(X=cs,Y=cl); wp=wp[::-1]*h/sr
mm=lambda x:f"{int(x//60)}:{x%60:04.1f}"
print("studio -> live")
for s in [0,11,29,68,108,126,164,203,221,242,250]:
    j=np.argmin(abs(wp[:,0]-s)); print(f"  {mm(s)} -> {mm(wp[j,1])}")
rm=20*np.log10(librosa.feature.rms(y=m,frame_length=sr,hop_length=sr)[0]+1e-9)
rv=20*np.log10(librosa.feature.rms(y=v,frame_length=sr,hop_length=sr)[0]+1e-9)
ri=20*np.log10(librosa.feature.rms(y=i,frame_length=sr,hop_length=sr)[0]+1e-9)
fl=librosa.feature.spectral_flatness(y=m,hop_length=sr)[0]
print("sec mix voc ins flat")
print('  '.join(f"{mm(k)}|{rm[k]:.0f}|{rv[k]:.0f}|{ri[k]:.0f}|{100*fl[k]:.0f}" for k in range(len(rm))))
