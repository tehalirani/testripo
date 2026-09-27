import numpy as np, soundfile as sf, librosa
from scipy.signal import correlate
s,_=sf.read('studio.wav'); l,_=sf.read('live.wav'); sr=44100
sm=s.mean(1); lm=l.mean(1)
# exact lag per 2s window, search +-1s around 3.3
print("t  lag(s)  corr   (studio vs live)")
for t in np.arange(1,127,4):
    q=sm[int(t*sr):int((t+2)*sr)]
    lo=int((t+2.3)*sr); seg=lm[lo:lo+len(q)+2*sr]
    c=correlate(seg,q,mode='valid',method='fft')
    en=np.sqrt(np.convolve(seg**2,np.ones(len(q)),'valid'))*np.linalg.norm(q)+1e-9
    nc=c/en; i=np.argmax(nc); print(f"{t:5.0f} {2.3+i/sr:.4f} {nc[i]:.2f}")
