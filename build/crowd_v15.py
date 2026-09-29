"""
نسخه ۱۵: مثل نسخه ۱۴، با یک اصلاح در شروع

بازخورد دانشجو: «رفتی تیکه اول لایو رو گذاشتی اول این و بعدش دوباره اول فایل خام پلی میشه. من میخوام
زیرصدای مردم توی اول فایل لایو بیفته زیر فایل ما.»

  - صدای خواننده لایو («بزن باران») از اینترو حذف شد، چون همان خط را خواننده خودمان کمی بعد می‌خواند.
  - فقط جواب مردم («بزن باران، بزن باران بزن»، لایو ۰:۴۴ تا ۰:۴۸.۸، از بخش «صداهای پشت» Karaoke)
    برداشته شد و زیر همان خط در آهنگ ما قرار گرفت. جای آن با شباهت کروما و الگوی شروع هجاها پیدا شد
    و روی ضرب آهنگ تنظیم شد: بند ۱ حدود ۰:۴۳.۴ و بند ۲ حدود ۲:۱۲.۸.

اجرا:  python3 crowd_v15.py <WORK_DIR> <OUT_DIR> <REC_DIR>
"""
import sys, os, json
import numpy as np
import soundfile as sf

WORK, OUT, REC = sys.argv[1], sys.argv[2], sys.argv[3]
import crowd_v14 as V14
V12, V10, M, V8 = V14.V12, V14.V10, V14.M, V14.V8
from make_live import SR, N, db, rms, place, fade, time_stretch, beat, BEAT0, BEAT, env

CHANT_LIVE = (44.0, 48.8)
CHANT_BEAT_OFFSET = (44.149 - 44.0) * V14.TEMPO_RATIO     # ضرب اول شعار نسبت به شروع تکه
PLACES = [43.35, 132.71]                                   # بهترین جا از مقایسه با وکال (بند ۱ و بند ۲)
CHANT_REL_DB = -6.0                                        # نسبت به خواننده


def chant_under_lines():
    B = V12.load_backing()
    seg = B[:, int(CHANT_LIVE[0] * SR):int(CHANT_LIVE[1] * SR)].copy()
    y = time_stretch(seg, SR, 1 / V14.TEMPO_RATIO, -2.0, high_quality=True, preserve_formants=True)
    y = fade(y, 0.1, 0.5)
    out = np.zeros((2, N), np.float32); placed = []
    for p in PLACES:
        k = round((p + CHANT_BEAT_OFFSET - BEAT0) / BEAT)          # روی نزدیک‌ترین ضرب
        t0 = beat(k) - CHANT_BEAT_OFFSET
        a, b = int(t0 * SR), int(t0 * SR) + y.shape[1]
        lead = rms(M.voc[:, a:b])
        g = db(CHANT_REL_DB) * lead / rms(y)
        place(out, y * g, t0); placed.append(round(t0, 2))
    return out, placed


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    M.load_clear_choir()
    murmur = M.murmur_bed()
    src = V14.crowd_source(); tl = V14.time_map()
    crowd = V12.warp_window(src, tl, *V14.SPAN)
    crowd = V14.Pedalboard([V14.HighpassFilter(150), V14.LowpassFilter(8000), V14.PeakFilter(3000, -2.0, 0.9),
                            V14.Reverb(room_size=0.8, damping=0.6, wet_level=0.25, dry_level=0.85, width=1.0)])(crowd, SR)
    crowd *= env(N, [(V14.SPAN[0], V14.SPAN[1], 0, 1.0, 3.0)])
    grid = np.arange(60, 130, 0.05); tg = tl(grid)
    rs = slice(int(np.interp(V14.CROWD_REF_LIVE[0], tg, grid) * SR), int(np.interp(V14.CROWD_REF_LIVE[1], tg, grid) * SR))
    sl = slice(int(V12.LEAD_REF[0] * SR), int(V12.LEAD_REF[1] * SR))
    lead_db = 20 * np.log10(rms(M.voc[:, sl] * db(2)))
    crowd *= db(-3 + lead_db - 20 * np.log10(rms(crowd[:, rs])))
    chant, placed = chant_under_lines()
    intro = V8.smooth_intro() * db(-4 + M.CHEER_REF)          # اینترو مثل نسخه ۱۳، بدون تکه‌ی خواننده لایو
    wav = M.mix(V8.cfg_v8(intro + chant + crowd), None, murmur, 'O_v15')
    x, _ = sf.read(wav, dtype='float32', always_2d=True); x = x.T
    V10.write(os.path.join(OUT, 'preview_v15_start'), x[:, int(8 * SR):int(50 * SR)])
    V10.write(os.path.join(OUT, 'preview_v15_chant_only'), chant[:, int(40 * SR):int(50 * SR)])
    json.dump(dict(chant_at=placed), open(os.path.join(OUT, 'v15_info.json'), 'w'))
    print(placed, wav)
