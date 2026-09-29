"""
نسخه ۱۴: کل صدای جمعیت نسخه لایو (به‌جز خواننده‌ها) زیر آهنگ + «بزن باران» و جواب مردم در اینترو

یافته‌ها (اندازه‌گیری روی بخش «صداهای پشت» خروجی Karaoke):
  - صدای «یک نفر» که دانشجو شنید، یک خواننده‌ی پشت صحنه است: وسط استریو (هم‌بستگی چپ و راست ۰.۹۶ تا ۰.۹۸)
    و روی نت ثابت A3.
  - جمعیت واقعی پخش است (هم‌بستگی نزدیک صفر، مثلاً در ۱:۵۰ تا ۲:۰۲ لایو).
پس لایه‌ی جمعیت = کناره‌های استریوی «صداهای پشت» (خواننده اصلی و خواننده پشت صحنه که وسط‌اند حذف می‌شوند)
+ بخش وسط با ‎-20 دسی‌بل. کانال راست با ۱۲ میلی‌ثانیه تأخیر ساخته می‌شود تا در پخش تک‌کاناله (موبایل)
صدای جمعیت حذف نشود. یک گین ثابت برای کل آهنگ: بلندی طبیعی کنسرت حفظ می‌شود.

اجرا:  python3 crowd_v14.py <WORK_DIR> <OUT_DIR> <REC_DIR>
"""
import sys, os, json
import numpy as np
import soundfile as sf

WORK, OUT, REC = sys.argv[1], sys.argv[2], sys.argv[3]
import crowd_v12 as V12
V10, M, V8 = V12.V10, V12.M, V12.V8
from make_live import (SR, N, db, rms, env, place, fade, time_stretch, beat, Pedalboard, HighpassFilter,
                       LowpassFilter, PeakFilter, Reverb)

TEMPO_RATIO = 103.36 / 100.0          # لایو تندتر است
CROWD_REF_LIVE = (110.0, 122.0)       # ۱:۵۰ تا ۲:۰۲ لایو: فقط مردم
SPAN = (66.3, 244.0)                  # زمان آهنگ ما (از کُرس اول تا تشویق پایانی)


def time_map():
    """DTW نسخه ۱۲. بند اول لایو با آهنگ ما جور نشد (خطای هم‌ترازی ±۱.۵ ثانیه) و جمعیت لایو آنجا
    تقریباً ساکت است (‎-40 تا ‎-57 dB)، پس لایه‌ی جمعیت از کُرس اول شروع می‌شود."""
    return V12.time_map()


def crowd_source():
    B = V12.load_backing()
    mid = (B[0] + B[1]) / 2; side = (B[0] - B[1]) / 2
    d = int(0.012 * SR)
    side_r = np.concatenate([np.zeros(d, np.float32), side[:-d]])
    return np.stack([side + mid * db(-20), side_r + mid * db(-20)]).astype(np.float32)


def intro_call_response():
    """«بزن باران» خواننده + جواب مردم، از وکال کامل لایو (۰:۴۰ تا ۰:۴۸.۸)، ۲ نیم‌پرده پایین و با تمپوی ما.
    ضرب شروع جواب مردم در لایو (۴۴.۱۵) روی ضرب ۲۰ آهنگ ما می‌نشیند."""
    v, _ = sf.read(os.path.join(WORK, 'sep', 'reallive_(Vocals)_UVR-MDX-NET-Voc_FT.wav'), dtype='float32', always_2d=True)
    a, b = 40.0, 48.8
    seg = v.T[:, int(a * SR):int(b * SR)].copy()
    y = time_stretch(seg, SR, 1 / TEMPO_RATIO, -2.0, high_quality=True, preserve_formants=True)
    y = fade(y / rms(y) * db(-22), 0.15, 0.6)
    t0 = beat(20) - (44.149 - a) * TEMPO_RATIO
    out = np.zeros((2, N), np.float32)
    place(out, y, t0)
    return out, t0, t0 + y.shape[1] / SR


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    M.load_clear_choir()
    murmur = M.murmur_bed()
    src = crowd_source()
    tl = time_map()
    crowd = V12.warp_window(src, tl, *SPAN)
    crowd = Pedalboard([HighpassFilter(150), LowpassFilter(8000), PeakFilter(3000, -2.0, 0.9),
                        Reverb(room_size=0.8, damping=0.6, wet_level=0.25, dry_level=0.85, width=1.0)])(crowd, SR)
    crowd *= env(N, [(SPAN[0], SPAN[1], 0, 1.0, 3.0)])
    # یک گین ثابت: جمعیت ۱:۵۰ لایو (در آهنگ ما حدود ۱:۴۰) ۳ دسی‌بل زیر خواننده
    ref_studio = (float(np.interp(CROWD_REF_LIVE[0], tl(np.arange(60, 130, 0.05)), np.arange(60, 130, 0.05))),
                  float(np.interp(CROWD_REF_LIVE[1], tl(np.arange(60, 130, 0.05)), np.arange(60, 130, 0.05))))
    sl = slice(int(V12.LEAD_REF[0] * SR), int(V12.LEAD_REF[1] * SR))
    lead_db = 20 * np.log10(rms(M.voc[:, sl] * db(2)))
    rs = slice(int(ref_studio[0] * SR), int(ref_studio[1] * SR))
    crowd *= db(-3 + lead_db - 20 * np.log10(rms(crowd[:, rs])))
    call, c0, c1 = intro_call_response()
    intro = V8.smooth_intro() * db(-4 + M.CHEER_REF) * (1 - (1 - db(-5)) * env(N, [(c0, c1, 0, 0.5, 0.8)]))
    wav = M.mix(V8.cfg_v8(intro + call + crowd), None, murmur, 'N_v14')
    x, _ = sf.read(wav, dtype='float32', always_2d=True); x = x.T
    V10.write(os.path.join(OUT, 'preview_v14_intro'), x[:, int(10 * SR):int(36 * SR)])
    V10.write(os.path.join(OUT, 'preview_v14_chorus1'), x[:, int(85 * SR):int(115 * SR)])
    V10.write(os.path.join(OUT, 'preview_v14_crowd_only'), (crowd + call)[:, int(85 * SR):int(115 * SR)])
    lvl = lambda a, b: round(20 * np.log10(rms(crowd[:, int(a * SR):int(b * SR)]) + 1e-12) - lead_db, 1)
    rep = dict(crowd_ref_studio=[round(v, 2) for v in ref_studio], intro_call=[round(c0, 2), round(c1, 2)],
               crowd_vs_lead={f'{a}-{b}': lvl(a, b) for a, b in
                              [(68, 88), (88, 99), (99, 112), (126, 145), (145, 161), (164, 183), (183, 202),
                               (203, 220), (221, 242), (242, 256)]})
    json.dump(rep, open(os.path.join(OUT, 'v14_levels.json'), 'w'), indent=1)
    print(json.dumps(rep, indent=1)); print(wav)
