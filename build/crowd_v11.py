"""
نسخه ۱۱: همان صداهای Kits.ai، ولی هر «آدم» جمله به جمله متفاوت از خواننده می‌خواند.

بازخورد دانشجو روی نسخه ۱۰: «دقیقاً همون فایل‌ها رو با همون ریتم گذاشتی زیر خواننده؛ شبیه جمعیت نیست،
انگار افکت انداختیم پشتش. ریتم مردم باید یکم فرق کنه، کمی بعد از خواننده بخونن، بعضی جاها رو بکش،
بعضی جاها طولانی‌تر، صدا رو بعضی جاها ببر بالا بعضی جاها بیار پایین؛ طبیعی باشه.»

برای هر نفر و هر جمله (جمله‌ها از فرورفتگی‌های بلندی وکال پیدا می‌شوند، ۱.۵ تا ۴ ثانیه):
  - تأخیر نسبت به خواننده: پایه‌ی ۱۵۰ تا ۲۸۰ میلی‌ثانیه برای هر نفر + ±۳۵ میلی‌ثانیه برای هر جمله
  - سرعت: هر جمله ۷٪ تندتر تا ۷٪ کندتر (بدون تغییر کوک)
  - آخر جمله: ۳۵٪ «می‌کشد» (۱۲ تا ۲۸٪ طولانی‌تر)، ۲۵٪ «زودتر ول می‌کند»، بقیه عادی
  - بلندی: هر جمله ‎-4 تا +3 دسی‌بل؛ ۱۵٪ جمله‌ها را نمی‌خواند

اجرا:  python3 crowd_v11.py <WORK_DIR> <OUT_DIR> <REC_DIR>
"""
import sys, os, glob, json
import numpy as np
import soundfile as sf

WORK, OUT, REC = sys.argv[1], sys.argv[2], sys.argv[3]
import crowd_v10 as V10          # (argv را خودش برای make_live تنظیم می‌کند)
M, V8 = V10.M, V10.V8
from make_live import (SR, N, db, rms, pan, env, place, fade, time_stretch, Pedalboard, HighpassFilter,
                       LowpassFilter, PeakFilter, Reverb, PitchShift)

CH1 = V10.CH1


def segments(t0, t1, lo=1.5, hi=4.0):
    """تقسیم وکال به جمله‌های ۱.۵ تا ۴ ثانیه‌ای، روی آرام‌ترین لحظه‌ها."""
    x = M.voc[:, int(t0 * SR):int(t1 * SR)].mean(0); h = int(0.02 * SR)
    e = 20 * np.log10(np.array([np.sqrt(np.mean(x[i:i + h] ** 2)) for i in range(0, len(x) - h, h)]) + 1e-9)
    t = t0 + np.arange(len(e)) * 0.02
    out, s = [], t0
    while s < t1 - 0.5:
        m = (t >= s + lo) & (t <= min(s + hi, t1))
        if not m.any(): out.append((s, t1)); break
        k = np.argmin(np.where(m, e, np.inf)); b = t[k]
        if t1 - b < lo: b = t1
        out.append((s, b)); s = b
    return out


def humanize(full, segs, r):
    out = np.zeros(N, np.float32)
    base_lag = r.uniform(0.15, 0.28)
    for p0, p1 in segs:
        if r.random() < 0.15:
            continue
        a, b = int((p0 - 0.05) * SR), int((p1 + 0.25) * SR)
        seg = full[a:b].astype(np.float32)
        if rms(seg) < 1e-4:
            continue
        split = int(len(seg) * r.uniform(0.65, 0.8))
        sb = r.uniform(0.93, 1.07)
        mode = r.choice(['hold', 'cut', 'normal'], p=[0.35, 0.25, 0.4])
        st = r.uniform(0.72, 0.88) if mode == 'hold' else sb
        body = time_stretch(seg[None, :split], SR, sb)[0]
        tail = time_stretch(seg[None, split:], SR, st)[0]
        xf = min(int(0.03 * SR), len(body), len(tail))
        ramp = np.linspace(0, 1, xf)
        y = np.concatenate([body[:-xf], body[-xf:] * (1 - ramp) + tail[:xf] * ramp, tail[xf:]])
        if mode == 'cut':
            y = y[:int(len(y) * r.uniform(0.7, 0.9))]
        y = fade(y[None], 0.04, 0.25 if mode == 'cut' else 0.2)[0]
        y *= db(r.uniform(-4, 3))
        lag = base_lag + r.normal(0, 0.035)
        place(out[None], y[None], a / SR + max(0.08, lag))
    return out


def build(level_db, seed=11):
    ref = M.voc[:, int(V10.SRC_T0 * SR):int((V10.SRC_T0 + 41) * SR)].mean(0)
    files = sorted(glob.glob(os.path.join(REC, '*.aac')) + glob.glob(os.path.join(REC, '*.wav')) +
                   glob.glob(os.path.join(REC, '*.mp3')) + glob.glob(os.path.join(REC, '*.m4a')))
    segs = segments(*CH1)
    r = np.random.default_rng(seed)
    W = np.zeros((2, N), np.float32); Mn = np.zeros((2, N), np.float32)
    info = []
    for f in files:
        y = V10.load_voice(f)
        lag = V10.align(y, ref)
        y = y[int(max(lag, 0) * SR):]
        full = np.zeros(N, np.float32); a = int(V10.SRC_T0 * SR)
        full[a:a + len(y)] = y[:N - a]
        male = 'mard' in os.path.basename(f)
        info.append(dict(file=os.path.basename(f), male=male))
        for inst in range(3):
            z = full if inst == 0 else Pedalboard([PitchShift(semitones=float(r.uniform(-0.1, 0.1)))])(full, SR)
            z = humanize(z, segs, r)
            (Mn if male else W)[:] += pan(z, r.uniform(-0.6, 0.6) if male else r.uniform(-1, 1)) * r.uniform(0.7, 1.0)
    hall = Reverb(room_size=V10.HALL_ROOM, damping=0.5, wet_level=0.4, dry_level=0.7, width=1.0)
    W = Pedalboard([HighpassFilter(150), LowpassFilter(7000), PeakFilter(3000, -2.5, 0.9), hall])(W, SR)
    Mn = Pedalboard([HighpassFilter(150), LowpassFilter(5000), hall])(Mn, SR)
    sl = slice(int(CH1[0] * SR), int(CH1[1] * SR))
    if rms(Mn[:, sl]) > 1e-6:
        Mn = Mn / rms(Mn[:, sl]) * rms(W[:, sl]) * db(-6)
    crowd = W + Mn
    g = db(level_db) * rms(M.voc[:, sl] * db(2.0)) / rms(crowd[:, sl])
    e = env(N, [(CH1[0] - 0.3, CH1[1] + 3.0, 0, 0.8, 2.0)])
    return W * g * e, Mn * g * e, info, segs


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    M.load_clear_choir()
    murmur = M.murmur_bed()
    intro = V8.smooth_intro() * db(-4 + M.CHEER_REF)
    cut = lambda x: x[:, int(68 * SR):int(88 * SR)]
    W, Mn, info, segs = build(-6)
    crowd = W + Mn
    wav = M.mix(V8.cfg_v8(intro + crowd), None, murmur, 'K_v11')
    x, _ = sf.read(wav, dtype='float32', always_2d=True)
    V10.write(os.path.join(OUT, 'preview_v11'), cut(x.T))
    V10.write(os.path.join(OUT, 'preview_v11_crowd_only'), cut(crowd))
    sl = slice(int(CH1[0] * SR), int(CH1[1] * SR))
    rep = dict(segments=[(round(a, 2), round(b, 2)) for a, b in segs],
               lead=round(20 * np.log10(rms(M.voc[:, sl] * db(2))), 1),
               women=round(20 * np.log10(rms(W[:, sl])), 1), men=round(20 * np.log10(rms(Mn[:, sl])), 1), voices=info)
    json.dump(rep, open(os.path.join(OUT, 'v11_levels.json'), 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False)); print(wav)
