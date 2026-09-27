import numpy as np, soundfile as sf, librosa
sr=22050
y,_=sf.read('sep/song_(Vocals)_UVR-MDX-NET-Voc_FT.wav'); v=librosa.resample(y.mean(1).astype(np.float32),orig_sr=44100,target_sr=sr)
segs={'A 0:30-0:46':(30,46),'B 0:49-1:05':(49,65),'C 1:09-1:25':(69,85),'D 1:28-1:48':(88,108),
'E 2:06-2:21':(126,141),'F 2:25-2:41':(145,161),'G 2:45-3:01':(165,181),'H 3:04-3:21':(184,201),'I 3:24-3:39':(204,219),'J 3:42-4:01':(222,241)}
feat={}
for k,(a,b) in segs.items():
    x=v[a*sr:b*sr]
    m=librosa.feature.mfcc(y=x,sr=sr,n_mfcc=25,n_mels=64)[2:]   # drop energy/tilt
    e=librosa.feature.rms(y=x)[0]; keep=e>np.percentile(e,50)
    m=m[:,:len(keep)][:,keep]
    feat[k]=np.concatenate([m.mean(1),m.std(1)])
F=np.array(list(feat.values())); F=(F-F.mean(0))/(F.std(0)+1e-9)
ks=list(feat)
ref1=F[0]; ref2=F[2]
for i,k in enumerate(ks):
    d1=np.linalg.norm(F[i]-ref1); d2=np.linalg.norm(F[i]-ref2)
    print(f"{k}: dist to singer1(A) {d1:5.1f}  to singer2(C) {d2:5.1f} -> {'1' if d1<d2 else '2'}")
