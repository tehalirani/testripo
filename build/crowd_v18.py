"""
نسخه ۱۸: نسخه ۱۷ + رفع صداهای ناگهانی (بعد از بررسی خودکار کیفیت، build/qa.py)

دانشجو: «خودت هم بررسی کن ببینی جایی یهو صدای عجیبی نباشه و همه چی منطقی باشه.»
بررسی خودکار نسخه ۱۷: کلیپ و تیک نداشت، ولی ۹ شروع ناگهانی لایه (بیش از ۱۵ دسی‌بل در ۱۰۰ میلی‌ثانیه) و
۱۰ جهش صدای تیز (بالای ۸ کیلوهرتز) پیدا شد:
  - ۰:۲۹.۹: شروع جواب مردم (فید ورود فقط ۰.۱ ثانیه بود)
  - ۰:۴۹: شروع صداهای خود دانشجو، با صدای تیز شدید (حرف «ز» یا صدای نفس در ضبط گوشی)
  - ۲:۲۲ تا ۳:۵۶: جیغ‌ها و صداهای کوتاه جمعیت لایو
اصلاح:
  - soften(): بالا رفتن بلندی هر لایه‌ی جمعیت حداکثر ۳۰ دسی‌بل در ثانیه (۳ دسی‌بل در ۱۰۰ میلی‌ثانیه)؛ پایین آمدن آزاد
  - low-pass: صداهای دانشجو ۶ کیلوهرتز، جمعیت لایو ۶.۵ کیلوهرتز
  - فید ورود نرم‌تر: جواب مردم ۰.۳۵ ثانیه، صداهای دانشجو ۰.۸ ثانیه

اجرا:  python3 crowd_v18.py <WORK_DIR> <OUT_DIR> <REC_DIR>
"""
import sys, os, json
import numpy as np
import soundfile as sf

WORK, OUT, REC = sys.argv[1], sys.argv[2], sys.argv[3]
import crowd_v17 as V17
V16, V14, V12, V10, M, V8 = V17.V16, V17.V14, V17.V12, V17.V10, V17.M, V17.V8
from make_live import (SR, N, db, rms, env, fade, place, time_stretch, beat, BEAT0, BEAT, Pedalboard,
                       HighpassFilter, LowpassFilter, PeakFilter, Reverb)


def soften(x, rise_db_per_s=30.0, frame=0.01):
    h = int(frame * SR)
    m = x.mean(0) if x.ndim > 1 else x
    n = len(m) // h
    e = 20 * np.log10(np.sqrt((m[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-7)
    lim = e.copy(); step = rise_db_per_s * frame
    for i in range(1, n):
        lim[i] = min(e[i], lim[i - 1] + step)
    g = 10 ** ((lim - e) / 20)
    g = np.convolve(g, np.ones(3) / 3, 'same')
    gs = np.interp(np.arange(x.shape[-1]), (np.arange(n) + 0.5) * h, g).astype(np.float32)
    return x * gs


def chant_soft():
    B = V12.load_backing()
    seg = B[:, int(V16.CHANT_LIVE[0] * SR):int(V16.CHANT_LIVE[1] * SR)].copy()
    y = time_stretch(seg, SR, 1 / V14.TEMPO_RATIO, -2.0, high_quality=True, preserve_formants=True)
    y = fade(y, 0.35, 0.5)
    out = np.zeros((2, N), np.float32)
    for p in V16.PLACES:
        k = round((p + V16.CHANT_BEAT_OFFSET - BEAT0) / BEAT)
        t0 = beat(k) - V16.CHANT_BEAT_OFFSET
        a = int(t0 * SR); b = a + y.shape[1]
        place(out, y * db(V16.CHANT_REL_DB) * rms(M.voc[:, a:b]) / rms(y), t0)
    return soften(out)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    M.load_clear_choir()
    murmur = M.murmur_bed()
    src = V14.crowd_source(); tl = V14.time_map()
    live = V12.warp_window(src, tl, *V14.SPAN)
    live = Pedalboard([HighpassFilter(150), LowpassFilter(6500), PeakFilter(3000, -2.0, 0.9),
                       Reverb(room_size=0.8, damping=0.6, wet_level=0.25, dry_level=0.85, width=1.0)])(live, SR)
    live = soften(live) * env(N, [(V14.SPAN[0], V14.SPAN[1], 0, 1.0, 3.0)])
    grid = np.arange(60, 130, 0.05); tg = tl(grid)
    rs = slice(int(np.interp(V14.CROWD_REF_LIVE[0], tg, grid) * SR), int(np.interp(V14.CROWD_REF_LIVE[1], tg, grid) * SR))
    sl = slice(int(V12.LEAD_REF[0] * SR), int(V12.LEAD_REF[1] * SR))
    lead_db = 20 * np.log10(rms(M.voc[:, sl] * db(2)))
    live *= db(-3 + lead_db - 20 * np.log10(rms(live[:, rs])))
    chant = chant_soft()
    me, info = V17.build_self_crowd()
    me = Pedalboard([LowpassFilter(6000)])(me, SR)
    me = soften(me) * env(N, [(48.6, 110.0, 0, 0.8, 2.0)])
    intro = V8.smooth_intro() * db(-4 + M.CHEER_REF)
    wav = M.mix(V8.cfg_v8(intro + chant + live + me), None, murmur, 'R_v18_full')
    x, _ = sf.read(wav, dtype='float32', always_2d=True); x = x.T
    orig = sf.info(os.path.join(WORK, 'song.wav')).frames
    y = x[:, :orig].copy(); f = int(5 * SR); y[:, -f:] *= np.linspace(1, 0, f) ** 2
    sf.write(os.path.join(OUT, 'R_v18.wav'), y.T, SR, subtype='PCM_16')
    V10.write(os.path.join(OUT, 'preview_v18'), y[:, int(26 * SR):int(90 * SR)])
    print(json.dumps(info, ensure_ascii=False))
