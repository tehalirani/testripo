import numpy as np, soundfile as sf, sys
S='/tmp/claude-0/-home-user-testripo/1c8c3296-3e38-519a-bee2-bc0f6dd6aa78/scratchpad'
x,sr=sf.read(sys.argv[1]); base,_=sf.read(sys.argv[2]); n=min(len(x),len(base)); x=x[:n]; base=base[:n]
mm=lambda t:f"{int(t//60)}:{t%60:05.2f}"
print('len',mm(n/sr),'peak',round(np.abs(x).max(),3),'clipped samples',int((np.abs(x)>0.999).sum()))
# clicks: first-difference outliers vs local level
d=np.abs(np.diff(x.mean(1))); h=int(0.01*sr)
loc=np.sqrt(np.convolve(x.mean(1)**2,np.ones(h)/h,'same'))[1:]+1e-4
z=d/loc; idx=np.where(z>6)[0]
clicks=[]; last=-1
for i in idx:
    if i-last>sr*0.05: clicks.append(i/sr)
    last=i
print('click candidates:',len(clicks),[mm(t) for t in clicks[:15]])
# extra layers = x - base
e=x-base; f=int(0.05*sr)
def env(y): return 20*np.log10(np.sqrt((y[:len(y)//f*f].reshape(-1,f,2)**2).mean((1,2)))+1e-9)
ee=env(e); ex=env(x)
jumps=[(i*0.05,ee[i]-ee[i-2]) for i in range(2,len(ee)) if ee[i]-ee[i-2]>15 and ee[i]>ex[i]-20]
drops=[(i*0.05,ee[i-2]-ee[i]) for i in range(2,len(ee)) if ee[i-2]-ee[i]>15 and ee[i-2]>ex[i-2]-20]
def merge(l):
    out=[]
    for t,v in l:
        if not out or t-out[-1][0]>0.5: out.append((t,v))
    return out
print('abrupt layer onsets (>15dB/100ms, audible):',[(mm(t),round(v)) for t,v in merge(jumps)])
print('abrupt layer cut-offs:',[(mm(t),round(v)) for t,v in merge(drops)])
# high-frequency bursts in added layers
from scipy.signal import butter,sosfilt
sos=butter(4,8000,'high',fs=sr,output='sos'); eh=sosfilt(sos,e.mean(1)); xh=sosfilt(sos,x.mean(1))
g=int(0.1*sr); he=20*np.log10(np.sqrt((eh[:len(eh)//g*g].reshape(-1,g)**2).mean(1))+1e-9); hx=20*np.log10(np.sqrt((xh[:len(xh)//g*g].reshape(-1,g)**2).mean(1))+1e-9)
hb=[(i*0.1,he[i]-np.median(he[max(0,i-20):i+20])) for i in range(len(he)) if he[i]-np.median(he[max(0,i-20):i+20])>12 and he[i]>hx[i]-10]
print('HF bursts from added layers:',[(mm(t),round(v)) for t,v in merge(hb)][:20])
# mono compatibility per 2s
L,R=x[:,0],x[:,1]; bad=[]
for t in range(0,n//sr-2,2):
    a=L[t*sr:(t+2)*sr]; b=R[t*sr:(t+2)*sr]; c=np.corrcoef(a,b)[0,1]
    if c<0.1: bad.append((t,round(c,2)))
print('low L/R correlation (mono risk):',bad)
