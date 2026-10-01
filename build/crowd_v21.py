"""
نسخه ۲۱: نسخه ۲۰ + شش اصلاح دانشجو

۱. ۰:۲۸: تشویق اینترو قطع و دوباره شروع می‌شد (تشویق ریتمیک در ۰:۲۹ تمام می‌شد و اوج ورود خواننده کمی بعد شروع می‌شد).
   -> تشویق بی‌وقفه تا اوج ادامه دارد و از لحظه‌ی شروع خواننده ۱ (۰:۲۹.۶۵) ۹ دسی‌بل نرم کم می‌شود تا خواننده واضح باشد.
۲. ۰:۵۲ تا ۱:۰۵: صداهای دانشجو «حمومی» بودند (دو بار ریورب). -> ریورب دوم حذف، ‎-7 دسی‌بل.
۳. ۱:۰۸ تا ۱:۱۰: صدای زنان نبود. علت: soften() بالا رفتن از سکوت را هم محدود می‌کرد (حدود ۲ ثانیه طول می‌کشید).
   -> soften2(): هر شروع حداکثر ۱۲ دسی‌بل نرم می‌شود و در حدود ۰.۴ ثانیه به سطح کامل می‌رسد.
۴. ۱:۲۸ تا ۱:۳۹ و ۳:۰۳ تا ۳:۱۶: صدای رباتی زیر خواننده ۲. هر دو همان جمله‌اند و صدای زنان کُرس ۲ از کُرس ۱ ساخته شده،
   پس تبدیل Kits.ai در این جمله خراب است. -> صدای Kits.ai در این دو جمله ۱۲ دسی‌بل کم می‌شود.
۵. ۲:۰۵ تا ۲:۱۶: همخوانی زیر خواننده ۱ جای درستی نبود. DTW بند ۱ به بند ۲ نشان داد جای «بزن باران بزن» درست است (۲:۰۵.۶)،
   پس مشکل از جمعیت لایوِ هم‌ترازنشده بود. -> جمعیت لایو در ۲:۰۵.۵ تا ۲:۲۵ حذف شد (‎-15 دسی‌بل).
۶. دو همخوانی واقعی لایو در همه‌ی جاهای مشابه:
   - «بزن باران بزن باران بزن» (لایو ۰:۴۱ تا ۰:۴۹): بند ۱ و بند ۲ (مثل قبل)
   - همخوانی مردم بدون خواننده (لایو ۱:۵۰ تا ۱:۵۳، در آهنگ ما ۱:۳۹ تا ۱:۴۳): کپی در جاهای هم‌ملودی
     ۱:۱۹.۸ (شباهت کروما ۰.۹۳۵)، ۳:۱۵.۰ (DTW و کروما، ۰.۹۷۱) و ۳:۵۳.۴ (۰.۹۷۱)

اجرا:  python3 crowd_v21.py <WORK_DIR> <OUT_DIR> <REC_DIR>
"""
import sys, os, json
import numpy as np
import soundfile as sf

WORK, OUT, REC = sys.argv[1], sys.argv[2], sys.argv[3]
import crowd_v20 as V20
V19, V18, V17, V16, V14, V12, V10, M, V8 = (V20.V19, V20.V18, V20.V17, V20.V16, V20.V14, V20.V12, V20.V10,
                                            V20.M, V20.V8)
V11 = V12.V11
from make_live import (SR, N, db, rms, env, fade, place, Pedalboard, HighpassFilter, LowpassFilter, PeakFilter,
                       Reverb, Compressor)

S1_START = 29.65
SWELL = (27.6, 33.5)
LINE_SRC = (99.0, 103.0)                    # همخوانی «فقط مردم» در آهنگ ما
LINE_COPIES = [79.79, 195.0, 233.4]
KITS_BAD = [(87.5, 99.5), (183.0, 196.5)]
LIVE_CUT = (125.5, 145.0)


def soften2(x, rise_db_per_s=30.0, max_cut_db=12.0, frame=0.01):
    h = int(frame * SR); m = x.mean(0) if x.ndim > 1 else x; n = len(m) // h
    e = 20 * np.log10(np.sqrt((m[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-7)
    lim = e.copy(); step = rise_db_per_s * frame
    for i in range(1, n):
        lim[i] = min(e[i], max(lim[i - 1], e[i] - max_cut_db) + step)
    g = np.convolve(10 ** ((lim - e) / 20), np.ones(3) / 3, 'same')
    return x * np.interp(np.arange(x.shape[-1]), (np.arange(n) + 0.5) * h, g).astype(np.float32)


def intro_layer():
    """تشویق ریتمیک بی‌وقفه تا اوج ورود خواننده ۱، بعد کاهش نرم ۹ دسی‌بل از شروع آواز."""
    rhythm = V8.smooth_intro(12.5, 31.5) * db(-4 + M.CHEER_REF)
    swell = np.zeros((2, N), np.float32)
    place(swell, M.cheer(M.POOL_SAFE, 0.0, SWELL[1] - SWELL[0], 1.3, 3.5), SWELL[0], -4 + M.CHEER_REF)
    place(swell, M.cheer(M.SCREAM, 0.0, 2.0, 0.05, 1.0), 28.6, -6 + M.CHEER_REF)
    duck = 1 - (1 - db(-9)) * env(N, [(S1_START, N / SR, 0, 0.5, 0.01)])
    return (rhythm + swell) * duck


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    M.load_clear_choir()
    murmur = M.murmur_bed()
    # جمعیت لایو
    src = V14.crowd_source(); tl = V14.time_map()
    live = V12.warp_window(src, tl, *V14.SPAN)
    live = Pedalboard([HighpassFilter(150), LowpassFilter(6500), PeakFilter(3000, -2.0, 0.9)])(live, SR)
    live = live / rms(live) * db(-30)
    live = Pedalboard([Compressor(threshold_db=-36, ratio=3, attack_ms=30, release_ms=300),
                       Reverb(room_size=0.8, damping=0.6, wet_level=0.25, dry_level=0.85, width=1.0)])(live, SR)
    live = soften2(live) * env(N, [(V14.SPAN[0], V14.SPAN[1], 0, 1.0, 3.0)])
    grid = np.arange(60, 130, 0.05); tg = tl(grid)
    rs = slice(int(np.interp(V14.CROWD_REF_LIVE[0], tg, grid) * SR), int(np.interp(V14.CROWD_REF_LIVE[1], tg, grid) * SR))
    sl = slice(int(V12.LEAD_REF[0] * SR), int(V12.LEAD_REF[1] * SR))
    lead_db = 20 * np.log10(rms(M.voc[:, sl] * db(2)))
    live *= db(V19.LIVE_REF_DB + lead_db - 20 * np.log10(rms(live[:, rs])))
    # ۶: کپی همخوانی «فقط مردم» در جاهای هم‌ملودی
    line = fade(live[:, int(LINE_SRC[0] * SR):int(LINE_SRC[1] * SR)].copy(), 0.3, 0.6)
    copies = np.zeros((2, N), np.float32)
    for t in LINE_COPIES:
        place(copies, line, t)
    # ۵: حذف جمعیت لایوِ هم‌ترازنشده در بند ۲
    live *= 1 - (1 - db(-15)) * env(N, [(LIVE_CUT[0], LIVE_CUT[1], 0, 1.0, 1.0)])
    chant = soften2(V18.chant_soft())
    # ۲: صداهای دانشجو بدون ریورب دوم
    V11.humanize = lambda full, segs, r: full.copy()
    V17.REL_DB = 0.0
    me, info = V17.build_self_crowd()
    me = soften2(Pedalboard([LowpassFilter(4500)])(me, SR))
    me *= env(N, [(48.6, 67.0, -7.0, 0.8, 1.0), (67.0, 110.0, -6.0, 1.0, 2.0)], base=-120)
    # ۳ و ۴: صدای زنان، شروع کامل و کم کردن جمله‌ی خراب
    V20.V18.soften = soften2
    women, wparts = V20.women_layer()
    women = soften2(women)
    women *= 1 - (1 - db(-12)) * env(N, [(a, b, 0, 0.5, 0.5) for a, b in KITS_BAD])
    crowd = chant + live + copies + me + women
    crowd *= 1 - (1 - db(V20.DIP[2])) * env(N, [(V20.DIP[0], V20.DIP[1], 0, 1.5, 1.5)])
    tb = np.array([0, 68.2, 68.3, 108.55, 108.6, 164.2, 164.3, 202.0, 202.1, 220.9, 220.95, 242.1, 242.2, N / SR])
    gb = db(np.array([0, 0, 2, 2, 0, 0, 2.5, 2.5, 0, 0, 1.5, 1.5, 0, 0]))
    crowd, worst = V19.cap_under_lead(crowd, (tb, gb))
    # ۱: اینترو بی‌وقفه؛ اوج قبلیِ ورود خواننده از فهرست تشویق‌ها برداشته می‌شود (در intro_layer ساخته شد)
    cfg = V8.cfg_v8(intro_layer() + crowd)
    cfg['cheers'] = [c for c in cfg['cheers'] if c[3] not in (28.95, 28.9)]
    wav = M.mix(cfg, None, murmur, 'U_v21_full')
    x, _ = sf.read(wav, dtype='float32', always_2d=True); x = x.T
    orig = sf.info(os.path.join(WORK, 'song.wav')).frames
    y = x[:, :orig].copy(); f = int(5 * SR); y[:, -f:] *= np.linspace(1, 0, f) ** 2
    sf.write(os.path.join(OUT, 'U_v21.wav'), y.T, SR, subtype='PCM_16')
    V10.write(os.path.join(OUT, 'preview_v21_start'), y[:, int(20 * SR):int(70 * SR)])
    V10.write(os.path.join(OUT, 'preview_v21_chorus1'), y[:, int(66 * SR):int(112 * SR)])
    V10.write(os.path.join(OUT, 'preview_v21_verse2_chorus2'), y[:, int(123 * SR):int(205 * SR)])
    json.dump(dict(women_vs_lead=wparts, cap_max_reduction=round(worst, 1)), open(os.path.join(OUT, 'v21_info.json'), 'w'))
    print(wparts, worst)
