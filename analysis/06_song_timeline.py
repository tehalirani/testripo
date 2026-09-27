import numpy as np, soundfile as sf, librosa
sr=22050
def ld(n):
    y,_=sf.read(n); return librosa.resample(y.mean(1).astype(np.float32),orig_sr=44100,target_sr=sr)
mix=ld('song.wav'); voc=ld('sep/song_(Vocals)_UVR-MDX-NET-Voc_FT.wav'); ins=ld('sep/song_(Instrumental)_UVR-MDX-NET-Voc_FT.wav')
tempo,beats=librosa.beat.beat_track(y=ins,sr=sr); bt=librosa.frames_to_time(beats,sr=sr)
print("tempo",tempo,"first beats",np.round(bt[:6],2))
ch=librosa.feature.chroma_cqt(y=ins,sr=sr).mean(1); print("key-ish",librosa.midi_to_note(60+np.argmax(ch))[:-1], np.round(ch/ch.max(),2))
mm=lambda t:f"{int(t//60)}:{t%60:04.1f}"
# per-second timeline
hop=sr
rv=librosa.feature.rms(y=voc,frame_length=hop,hop_length=hop)[0]; ri=librosa.feature.rms(y=ins,frame_length=hop,hop_length=hop)[0]
vdb=20*np.log10(rv+1e-9); idb=20*np.log10(ri+1e-9)
# pitch per 1s
f0,vf,_=librosa.pyin(voc,fmin=70,fmax=900,sr=sr,hop_length=512)
ft=librosa.times_like(f0,sr=sr,hop_length=512)
mf=librosa.feature.mfcc(y=voc,sr=sr,n_mfcc=20,hop_length=512)
np.save('song_f0.npy',f0)
print("t     voc_dB ins_dB  f0med  note")
for s in range(len(vdb)):
    m=(ft>=s)&(ft<s+1); p=np.nanmedian(f0[m]) if np.any(~np.isnan(f0[m])) else np.nan
    print(f"{mm(s)} {vdb[s]:6.0f} {idb[s]:6.0f} {p:6.0f} {librosa.hz_to_note(p) if p==p else '-'}")
