"""
نسخه ۱۹: نسخه ۱۸ با صدای جمعیت بلندتر و واضح‌تر

دانشجو: «صدای جمعیت رو بیشتر کن که واضح‌تر باشه.»
  - جمعیت لایو: قبلاً یک گین ثابت داشت و جز ۱:۴۰ و کُرس آخر ۱۵ تا ۲۲ دسی‌بل زیر خواننده بود.
    حالا اول فشرده می‌شود (کمپرسور، نسبت ۳:۱) تا جاهای آرام بالا بیایند، و بعد کل لایه بلندتر می‌شود
    (تیکه‌ی «فقط مردم» ۱ دسی‌بل زیر خواننده).
  - صداهای خود دانشجو: از ‎-6 به ‎-2 دسی‌بل نسبت به خواننده.
  - «بزن باران بزن» مردم: از ‎-6 به ‎-2 دسی‌بل.
  - نرم‌کننده‌ی شروع و فیلترهای نسخه ۱۸ حفظ شده‌اند.

اجرا:  python3 crowd_v19.py <WORK_DIR> <OUT_DIR> <REC_DIR>
"""
import sys, os, json
import numpy as np
import soundfile as sf

WORK, OUT, REC = sys.argv[1], sys.argv[2], sys.argv[3]
import crowd_v18 as V18
V17, V16, V14, V12, V10, M, V8 = V18.V17, V18.V16, V18.V14, V18.V12, V18.V10, V18.M, V18.V8
from make_live import (SR, N, db, rms, env, Pedalboard, HighpassFilter, LowpassFilter, PeakFilter, Reverb,
                       Compressor)

V17.REL_DB = -2.0
V16.CHANT_REL_DB = -2.0
LIVE_REF_DB = -1.0
CAP_DB = -2.0      # هر جا خواننده می‌خواند، کل جمعیت حداکثر ۲ دسی‌بل زیر خواننده


def cap_under_lead(crowd, boost):
    """سقف بلندی جمعیت نسبت به خواننده، پنجره‌های ۰.۵ ثانیه‌ای (گین هموار). جاهای بی‌آواز آزاد است."""
    w = int(0.5 * SR); n = N // w
    c = np.sqrt((crowd[:, :n * w].reshape(2, n, w) ** 2).mean((0, 2))) + 1e-9
    l = np.sqrt((M.voc[:, :n * w].reshape(2, n, w) ** 2).mean((0, 2))) + 1e-9
    t = (np.arange(n) + 0.5) * w / SR
    l = l * np.interp(t, *boost)
    active = 20 * np.log10(l) > -35
    over = 20 * np.log10(c) - (20 * np.log10(l) + CAP_DB)
    g_db = np.where(active & (over > 0), -np.minimum(over, 8.0), 0.0)   # حداکثر ۸ دسی‌بل کاهش (بدون سوراخ در صدا)
    g_db = np.convolve(np.pad(g_db, 2, mode='edge'), np.ones(5) / 5, 'valid')
    g = np.interp(np.arange(N), (np.arange(n) + 0.5) * w, 10 ** (g_db / 20)).astype(np.float32)
    return crowd * g, float(np.min(g_db))

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    M.load_clear_choir()
    murmur = M.murmur_bed()
    src = V14.crowd_source(); tl = V14.time_map()
    live = V12.warp_window(src, tl, *V14.SPAN)
    live = Pedalboard([HighpassFilter(150), LowpassFilter(6500), PeakFilter(3000, -2.0, 0.9)])(live, SR)
    live = live / rms(live) * db(-30)
    live = Pedalboard([Compressor(threshold_db=-36, ratio=3, attack_ms=30, release_ms=300),
                       Reverb(room_size=0.8, damping=0.6, wet_level=0.25, dry_level=0.85, width=1.0)])(live, SR)
    live = V18.soften(live) * env(N, [(V14.SPAN[0], V14.SPAN[1], 0, 1.0, 3.0)])
    grid = np.arange(60, 130, 0.05); tg = tl(grid)
    rs = slice(int(np.interp(V14.CROWD_REF_LIVE[0], tg, grid) * SR), int(np.interp(V14.CROWD_REF_LIVE[1], tg, grid) * SR))
    sl = slice(int(V12.LEAD_REF[0] * SR), int(V12.LEAD_REF[1] * SR))
    lead_db = 20 * np.log10(rms(M.voc[:, sl] * db(2)))
    live *= db(LIVE_REF_DB + lead_db - 20 * np.log10(rms(live[:, rs])))
    chant = V18.chant_soft()
    me, info = V17.build_self_crowd()
    me = Pedalboard([LowpassFilter(6000)])(me, SR)
    me = V18.soften(me) * env(N, [(48.6, 110.0, 0, 0.8, 2.0)])
    intro = V8.smooth_intro() * db(-4 + M.CHEER_REF)
    # افزایش خواننده در کُرس‌ها (از D_final_v4): ۲، ۲.۵ و ۱.۵ دسی‌بل
    tb = np.array([0, 68.2, 68.3, 108.55, 108.6, 164.2, 164.3, 202.0, 202.1, 220.9, 220.95, 242.1, 242.2, N / SR])
    gb = db(np.array([0, 0, 2, 2, 0, 0, 2.5, 2.5, 0, 0, 1.5, 1.5, 0, 0]))
    crowd_all, worst = cap_under_lead(chant + live + me, (tb, gb))
    print('cap: max reduction dB', round(worst, 1))
    wav = M.mix(V8.cfg_v8(intro + crowd_all), None, murmur, 'S_v19_full')
    x, _ = sf.read(wav, dtype='float32', always_2d=True); x = x.T
    orig = sf.info(os.path.join(WORK, 'song.wav')).frames
    y = x[:, :orig].copy(); f = int(5 * SR); y[:, -f:] *= np.linspace(1, 0, f) ** 2
    sf.write(os.path.join(OUT, 'S_v19.wav'), y.T, SR, subtype='PCM_16')
    V10.write(os.path.join(OUT, 'preview_v19'), y[:, int(26 * SR):int(115 * SR)])
    crowd = crowd_all
    lvl = lambda a, b: round(20 * np.log10(rms(crowd[:, int(a * SR):int(b * SR)]) + 1e-12) -
                             20 * np.log10(rms(M.voc[:, int(a * SR):int(b * SR)] * float(np.interp((a + b) / 2, tb, gb))) + 1e-12), 1)
    rep = {f'{a}-{b}': lvl(a, b) for a, b in [(29, 34), (49, 66), (68, 88), (88, 99), (99, 108), (108, 112), (126, 145),
                                               (145, 161), (164, 183), (183, 202), (203, 220), (221, 242)]}
    json.dump(rep, open(os.path.join(OUT, 'v19_levels.json'), 'w'), indent=1)
    print(json.dumps(rep))
