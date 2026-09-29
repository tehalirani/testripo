"""
نسخه ۱۳: جمعیت واقعی نسخه ۱۲، پخش‌تر و نرم‌تر

بازخورد دانشجو روی نسخه ۱۲: «صدای جمعیت انگار صدای یه نفر خاص هست؛ می‌خوام از شارپ بودن خارج بشه و
پخش بشه که حس بده جمعیت داره آهنگ رو می‌خونه.»

روی لایه‌ی جمعیتِ واقعیِ نسخه ۱۲ (نه روی صدای خواننده):
  ۱. ۶ کپی از همان جمعیت، هر کدام با تأخیر متغیر (۲۰ تا ۸۰ میلی‌ثانیه، آرام تغییر می‌کند)، ±۱۵ سنت کوک،
     فیلتر و جای استریوی متفاوت  -> یک صدای متمرکز به جمعیتی پخش تبدیل می‌شود
  ۲. کاهش تیزی: ‎-4.5 دسی‌بل در ۳.۵ کیلوهرتز، ‎-3 دسی‌بل در ۶ کیلوهرتز و بالاتر، low-pass ۵ کیلوهرتز
  ۳. ریورب سالن بیشتر (RT60 حدود ۳ ثانیه، wet بیشتر) و بردن به کناره‌های استریو

اجرا:  python3 crowd_v13.py <WORK_DIR> <OUT_DIR> <REC_DIR>
"""
import sys, os, json
import numpy as np
import soundfile as sf

WORK, OUT, REC = sys.argv[1], sys.argv[2], sys.argv[3]
import crowd_v12 as V12
V10, V11, M, V8 = V12.V10, V12.V11, V12.M, V12.V8
from make_live import (SR, N, db, rms, pan, env, wander_delay, to_sides, Pedalboard, HighpassFilter,
                       LowpassFilter, PeakFilter, HighShelfFilter, Reverb, PitchShift)


def diffuse(x, copies=6, seed=13):
    r = np.random.default_rng(seed)
    mono = x.mean(0).astype(np.float32)
    out = np.zeros((2, N), np.float32)
    for i in range(copies):
        y = Pedalboard([PitchShift(semitones=float(r.uniform(-0.15, 0.15))),
                        LowpassFilter(r.uniform(3800, 5500)),
                        PeakFilter(r.uniform(600, 1500), r.uniform(-3, 2), 1.0)])(mono, SR)
        y = wander_delay(y, r.uniform(0.035, 0.065), 0.02, r.uniform(0.15, 0.35), r)
        out += pan(y, -1 + 2 * (i + r.uniform(0.2, 0.8)) / copies) * r.uniform(0.7, 1.0)
    out = Pedalboard([HighpassFilter(180),
                      PeakFilter(3500, -4.5, 0.8), HighShelfFilter(6000, -3.0), LowpassFilter(5000),
                      Reverb(room_size=0.85, damping=0.6, wet_level=0.55, dry_level=0.5, width=1.0)])(out, SR)
    return to_sides(out, mid_db=-5.0)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    M.load_clear_choir()
    murmur = M.murmur_bed()
    intro = V8.smooth_intro() * db(-4 + M.CHEER_REF)
    live = V12.build_live_crowd()
    wide = diffuse(live)
    sl = slice(int(V12.LEAD_REF[0] * SR), int(V12.LEAD_REF[1] * SR))
    lead_db = 20 * np.log10(rms(M.voc[:, sl] * db(2)))
    cut = lambda x: x[:, int(85 * SR):int(115 * SR)]
    report = {}
    for lv in (-5, -2):
        crowd = wide * db(lv + lead_db - 20 * np.log10(rms(wide[:, sl])))
        wav = M.mix(V8.cfg_v8(intro + crowd), None, murmur, f'M_v13_{-lv}dB')
        x, _ = sf.read(wav, dtype='float32', always_2d=True)
        V10.write(os.path.join(OUT, f'preview_v13_{-lv}dB'), cut(x.T))
        if lv == -2:
            V10.write(os.path.join(OUT, 'preview_v13_crowd_only'), cut(crowd))
        report[lv] = {f'{a:.0f}-{b:.0f}': round(20 * np.log10(rms(crowd[:, int(a * SR):int(b * SR)])) - lead_db, 1)
                      for a, b in V12.WINDOWS}
    json.dump(report, open(os.path.join(OUT, 'v13_levels.json'), 'w'), indent=1)
    print(json.dumps(report, indent=1))
