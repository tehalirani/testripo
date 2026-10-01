"""
نسخه ۲۲: نسخه ۲۱ + سه خواسته‌ی دانشجو

۱. «صدای جمعیت خیلی کم شد، در حد خیلی کمی بیشترش کن»
   -> کل لایه‌های جمعیت +۳ دسی‌بل؛ سقف زیر خواننده از ‎-2 به ‎-1.5 دسی‌بل.
۲. «برای خواننده ۱ از زیرصدای مردم توی لایو ورژن واقعی استفاده نکردی»
   -> قبلاً هم‌ترازی بندها با کرومای سازها انجام می‌شد و جواب نداد (خطای ±۱.۵ ثانیه).
      حالا DTW روی صدای خواننده‌ها (وکال ما در برابر وکال اصلی لایو از Karaoke، ۲ نیم‌پرده جابه‌جا، کروما + شروع هجا):
      خطای بیشتر نقاط کمتر از ۰.۱ ثانیه. جمعیت لایو زیر بند ۱ (۰:۲۹ تا ۱:۰۶) و بند ۲ (۲:۰۵ تا ۲:۴۳).
۳. «۳:۵۴ خواننده‌ها نخونن و فقط صدای مردم باشه که می‌گن هوا هوای خاطرات اوست و بعدش جیغ»
   -> صدای خواننده‌ها (با صدای دوم) در ۳:۵۳.۳ تا ۳:۵۷.۶ قطع می‌شود؛ همخوانی واقعی مردم آنجا بلندتر است
      (و از سقف مستثنی است)؛ بعد جیغ و تشویق.

اجرا:  python3 crowd_v22.py <WORK_DIR> <OUT_DIR> <REC_DIR>
"""
import sys, os, json
import numpy as np
import soundfile as sf

WORK, OUT, REC = sys.argv[1], sys.argv[2], sys.argv[3]
import crowd_v21 as V21
V20, V19, V18, V17, V16, V14, V12, V10, M, V8 = (V21.V20, V21.V19, V21.V18, V21.V17, V21.V16, V21.V14, V21.V12,
                                                 V21.V10, V21.M, V21.V8)
V11 = V12.V11
from make_live import (SR, N, db, rms, env, fade, place, Pedalboard, HighpassFilter, LowpassFilter, PeakFilter,
                       Reverb, Compressor)

BOOST_DB = 3.0
V19.CAP_DB = -1.5
VERSES = {'verse1': (29.3, 66.3), 'verse2': (125.3, 163.0)}
FINAL_LINE = (233.3, 237.6)          # «هوا هوای خاطرات اوست»، فقط مردم


def vocal_map(name):
    d = np.load(os.path.join(WORK, f'dtw_vocal_{name}.npy'))
    st, lt = d[:, 0], d[:, 1]
    grid = np.arange(st.min(), st.max(), 0.05)
    m = np.interp(grid, st, lt)
    k = 20
    m = np.convolve(np.pad(m, k, mode='edge'), np.ones(2 * k + 1) / (2 * k + 1), 'valid')
    m = np.maximum.accumulate(m)
    return lambda t: np.interp(t, grid, m)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    M.load_clear_choir()
    murmur = M.murmur_bed()
    src = V14.crowd_source(); tl = V14.time_map()
    raw = V12.warp_window(src, tl, *V14.SPAN)
    raw *= 1 - env(N, [VERSES['verse2'] + (0, 0.8, 0.8)])          # بند ۲ با نگاشت سازها حذف می‌شود
    for name, (a, b) in VERSES.items():                               # بندها با نگاشت صدای خواننده
        raw += V12.warp_window(src, vocal_map(name), a, b) * env(N, [(a, b, 0, 0.8, 0.8)])
    live = Pedalboard([HighpassFilter(150), LowpassFilter(6500), PeakFilter(3000, -2.0, 0.9)])(raw, SR)
    live = live / rms(live) * db(-30)
    live = Pedalboard([Compressor(threshold_db=-36, ratio=3, attack_ms=30, release_ms=300),
                       Reverb(room_size=0.8, damping=0.6, wet_level=0.25, dry_level=0.85, width=1.0)])(live, SR)
    live = V21.soften2(live) * env(N, [(29.3, V14.SPAN[1], 0, 1.0, 3.0)])
    grid = np.arange(60, 130, 0.05); tg = tl(grid)
    rs = slice(int(np.interp(V14.CROWD_REF_LIVE[0], tg, grid) * SR), int(np.interp(V14.CROWD_REF_LIVE[1], tg, grid) * SR))
    sl = slice(int(V12.LEAD_REF[0] * SR), int(V12.LEAD_REF[1] * SR))
    lead_db = 20 * np.log10(rms(M.voc[:, sl] * db(2)))
    live *= db(V19.LIVE_REF_DB + lead_db - 20 * np.log10(rms(live[:, rs])))
    line = fade(live[:, int(V21.LINE_SRC[0] * SR):int(V21.LINE_SRC[1] * SR)].copy(), 0.3, 0.6)
    copies = np.zeros((2, N), np.float32)
    for t in V21.LINE_COPIES:
        if t != 233.4:
            place(copies, line, t)
    final = np.zeros((2, N), np.float32)
    place(final, line, 233.4, 4.0)                                    # «هوا هوای خاطرات اوست» فقط مردم، بلندتر
    chant = V21.soften2(V18.chant_soft())
    V11.humanize = lambda full, segs, r: full.copy()
    V17.REL_DB = 0.0
    me, info = V17.build_self_crowd()
    me = V21.soften2(Pedalboard([LowpassFilter(4500)])(me, SR))
    me *= env(N, [(48.6, 67.0, -7.0, 0.8, 1.0), (67.0, 110.0, -6.0, 1.0, 2.0)], base=-120)
    women, wparts = V20.women_layer()
    women = V21.soften2(women)
    women *= 1 - (1 - db(-12)) * env(N, [(a, b, 0, 0.5, 0.5) for a, b in V21.KITS_BAD])
    crowd = (chant + live + copies + me + women) * db(BOOST_DB)
    crowd *= 1 - (1 - db(V20.DIP[2])) * env(N, [(V20.DIP[0], V20.DIP[1], 0, 1.5, 1.5)])
    tb = np.array([0, 68.2, 68.3, 108.55, 108.6, 164.2, 164.3, 202.0, 202.1, 220.9, 220.95, 242.1, 242.2, N / SR])
    gb = db(np.array([0, 0, 2, 2, 0, 0, 2.5, 2.5, 0, 0, 1.5, 1.5, 0, 0]))
    crowd, worst = V19.cap_under_lead(crowd, (tb, gb))
    crowd += final
    cfg = V8.cfg_v8(V21.intro_layer() + crowd)
    cfg['cheers'] = [c for c in cfg['cheers'] if c[3] not in (28.95, 28.9)] + [
        (M.SCREAM, 0.0, 2.0, 237.3, -3, 0.05, 1.0),                   # بعد از «هوا هوای خاطرات اوست»: جیغ و تشویق
        (M.POOL_SAFE, 0.0, 7.0, 237.4, -4, 0.3, 4.0)]
    cfg['lead_duck'] = [(FINAL_LINE[0], FINAL_LINE[1], -40)]
    wav = M.mix(cfg, None, murmur, 'V_v22_full')
    x, _ = sf.read(wav, dtype='float32', always_2d=True); x = x.T
    orig = sf.info(os.path.join(WORK, 'song.wav')).frames
    y = x[:, :orig].copy(); f = int(5 * SR); y[:, -f:] *= np.linspace(1, 0, f) ** 2
    sf.write(os.path.join(OUT, 'V_v22.wav'), y.T, SR, subtype='PCM_16')
    V10.write(os.path.join(OUT, 'preview_v22_verse1'), y[:, int(26 * SR):int(70 * SR)])
    V10.write(os.path.join(OUT, 'preview_v22_verse2'), y[:, int(123 * SR):int(166 * SR)])
    V10.write(os.path.join(OUT, 'preview_v22_ending'), y[:, int(218 * SR):int(253 * SR)])
    lvl = lambda a, b: round(20 * np.log10(rms(crowd[:, int(a * SR):int(b * SR)]) + 1e-12) -
                             20 * np.log10(rms(M.voc[:, int(a * SR):int(b * SR)]) + 1e-12), 1)
    rep = {f'{a}-{b}': lvl(a, b) for a, b in [(30, 48), (49, 66), (68, 88), (88, 108), (126, 145), (145, 161),
                                               (164, 202), (221, 233)]}
    json.dump(dict(levels=rep, cap=round(worst, 1)), open(os.path.join(OUT, 'v22_info.json'), 'w'))
    print(rep, worst)
