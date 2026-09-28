"""
نسخه ۸: جمعیت بدون کلمه، هم‌ریتم با خواننده‌ها

چرا روش عوض شد:
  همخوانی‌های قبلی از روی صدای خود خواننده ساخته می‌شدند. هر کپیِ کمی دیرترِ صدای خواننده را
  گوش به‌صورت اکو و ریورب خود خواننده می‌شنود («حمومی»)، و تغییر کوک هم صدا را ناکوک و رباتی می‌کرد.
  در این نسخه هیچ کپی‌ای از صدای خواننده پخش نمی‌شود. جمعیت از جیغ و تشویقِ واقعیِ نسخه لایو ساخته می‌شود
  و بلندی‌اش با هجاهای خواننده بالا و پایین می‌رود، پس نت ندارد و هیچ‌وقت ناکوک نمی‌شود.

اجرا:  python3 crowd_v8.py <WORK_DIR> <OUT_DIR> [A1 A2 A3]
"""
import sys, os, json
import numpy as np

WORK, OUT = sys.argv[1], sys.argv[2]
ONLY = sys.argv[3:]
sys.argv = [sys.argv[0], WORK, OUT]
import make_live as M
from make_live import (SR, N, db, rms, pan, fade, env, place, beat, BEAT, BEAT0, Pedalboard,
                       HighpassFilter, LowpassFilter, HighShelfFilter, PitchShift, to_sides)


def phrases(x, thr_db=-35, min_gap=0.15):
    h = int(0.02 * SR)
    e = np.array([np.sqrt(np.mean(x[i:i + h] ** 2)) for i in range(0, len(x) - h, h)])
    act = 20 * np.log10(e + 1e-9) > thr_db
    runs, st = [], None
    for i, a in enumerate(act):
        if a and st is None: st = i
        if not a and st is not None: runs.append([st, i]); st = None
    if st is not None: runs.append([st, len(act)])
    out = []
    for r_ in runs:
        if out and (r_[0] - out[-1][1]) * 0.02 < min_gap: out[-1][1] = r_[1]
        else: out.append(r_)
    return [(a * 0.02, b * 0.02) for a, b in out if (b - a) * 0.02 > 0.3]

# بخش‌های آوازی و سطح پایه‌ی جمعیت در هر بخش (نسبت به سطح پایه‌ی هر نسخه)
SECTIONS = [
    ((29.6, 66.0), -6),      # بند ۱، خواننده ۱
    ((68.3, 108.55), 0),     # کُرس ۱، خواننده ۲
    ((125.6, 144.8), 0),     # بند ۲، خواننده ۱ با جمعیت
    ((144.9, 161.0), +3),    # «بزن باران که من هم ابریم…»: جمعیت پرشورتر
    ((164.3, 202.0), +1),    # کُرس ۲
    ((203.0, 220.4), -6),    # بریج
    ((220.95, 242.1), +2),   # کُرس آخر
]

STYLES = {
    # A1: نرم و ملایم؛ جمعیت آرام با خواننده نفس می‌کشد
    'A1': dict(base=-12, floor=0.35, lp=6000, accents=False),
    # A2: واضح‌تر و روشن‌تر
    'A2': dict(base=-8, floor=0.25, lp=8000, accents=False),
    # A3: زمینه‌ی آرام + جیغ و تشویق در شروع هر جمله
    'A3': dict(base=-14, floor=0.4, lp=6500, accents=True),
}


def lead_envelope():
    """بلندیِ صدای خواننده، هموارشده (حمله ۶۰ و رهاسازی ۳۵۰ میلی‌ثانیه) در بازه‌ی ۰ تا ۱."""
    x = M.voc.mean(0); h = int(0.01 * SR)
    fr = np.sqrt(np.mean(x[:len(x) // h * h].reshape(-1, h) ** 2, axis=1))
    d = 20 * np.log10(fr + 1e-9)
    g = np.clip((d + 45) / 30, 0, 1)
    out = np.zeros_like(g); a, r_ = np.exp(-0.01 / 0.06), np.exp(-0.01 / 0.35); y = 0.0
    for i, v in enumerate(g):
        y = a * y + (1 - a) * v if v > y else r_ * y + (1 - r_) * v
        out[i] = y
    return np.interp(np.arange(N), np.arange(len(out)) * h, out).astype(np.float32)


def crowd_bed(seed, lp):
    """بستر جیغ و تشویقِ واقعی (فقط تکه‌های بدون آواز)، دو لایه با شروع متفاوت؛
    لایه‌ی دوم کمی زیرتر تا جیغ‌های زنانه غالب شوند."""
    r = np.random.default_rng(seed)
    src = M.POOL_SAFE
    reps = int(np.ceil(N / src.shape[1])) + 2
    loop = np.concatenate([fade(src, 0.4, 0.4)] * reps, axis=1)
    o1 = int(r.uniform(0, src.shape[1]))
    l1 = loop[:, o1:o1 + N]
    l2 = Pedalboard([PitchShift(semitones=1.5)])(loop[:, :N][:, ::-1].copy(), SR)
    bed = l1 + 0.7 * l2
    bed = Pedalboard([HighpassFilter(250), LowpassFilter(lp), HighShelfFilter(4000, -2.0)])(bed, SR)
    return to_sides(bed / rms(bed) * 0.05, mid_db=-6.0)


def wordless(style, seed=41):
    st = STYLES[style]
    e = lead_envelope()
    bed = crowd_bed(seed, st['lp'])
    shaped = bed * (st['floor'] + (1 - st['floor']) * e ** 1.2)
    out = np.zeros((2, N), np.float32)
    r = np.random.default_rng(seed)
    for (t0, t1), off in SECTIONS:
        a, b = int(t0 * SR), int(t1 * SR)
        lead = rms(M.voc[:, a:b]); crowd = rms(shaped[:, a:b])
        g = db(st['base'] + off) * lead / crowd
        w = env(N, [(t0, t1, 0, 1.0, 1.5)])
        out += shaped * g * w
        if st['accents']:
            seg = M.voc[:, a:b].mean(0)
            for p0, _ in phrases(seg):
                t = t0 + p0
                sw = M.cheer(M.POOL_SAFE, r.uniform(0, 10), 2.5, 0.3, 1.8)
                place(out, sw * g * db(4), t - 0.2)
                sc = M.SCREAM[:, :int(0.6 * SR)]
                place(out, fade(sc, 0.02, 0.3) * g * db(0) * r.uniform(0.6, 1.0), t - 0.1 + r.normal(0, 0.03))
    return out


def smooth_intro(t0=12.5, t1=29.0):
    """هی‌هی‌های اول آهنگ، نرم: بستر تشویق که روی ضرب‌ها ملایم (حدود ۴ دسی‌بل) بالا و پایین می‌رود،
    حمله‌ی نرم ۵۰ و افت ۳۵۰ میلی‌ثانیه، بدون جیغ‌های کوتاه؛ با فید ورود ۲.۵ و خروج ۰.۸ ثانیه."""
    n0, n1 = int(t0 * SR), int(t1 * SR); n = n1 - n0
    src = M.POOL_SAFE
    bed = np.concatenate([src] * (int(np.ceil(n / src.shape[1])) + 1), axis=1)[:, :n]
    bed = Pedalboard([LowpassFilter(5000)])(bed, SR)
    t = np.arange(n) / SR + t0
    pulse = np.full(n, 0.55, np.float32)
    for k in range(int(np.ceil((t0 - BEAT0) / BEAT)), int((t1 - BEAT0) / BEAT) + 1):
        d = t - beat(k)
        strong = 0.45 if k % 2 == 0 else 0.3
        att = np.clip(d / 0.05, 0, 1)
        pulse += (strong * att * np.exp(-np.clip(d, 0, None) / 0.35) * (d >= 0)).astype(np.float32)
    out = np.zeros((2, N), np.float32)
    out[:, n0:n1] = bed * pulse / pulse.max()
    return out * env(N, [(t0, t1, 0, 2.5, 0.8)])


def cfg_v8(extra):
    v = dict(M.VARIANTS['D_final_v4'])
    v['choir'] = []                      # هیچ کپی‌ای از صدای خواننده
    v['chant_loop'] = []
    v['lead_duck'] = []                  # خواننده هیچ‌جا کم نمی‌شود
    v['lead_dry'] = [(0.0, N / SR)]      # خواننده در کل آهنگ خشک
    v['duet_offset'] = 0.0               # صدای دوم کُرس آخر بدون تأخیر (تأخیر = حس اکو)
    v['cheers'] = [c for c in v['cheers'] if c[3] != 27.4] + [
        (M.SCREAM, 0.0, 2.0, 28.95, -3, 0.03, 1.0),     # اوج ناگهانی برای خواننده ۱
        (M.POOL_SAFE, 0.0, 7.0, 28.9, -4, 0.25, 4.5)]
    v['extra_fx'] = extra
    return v


if __name__ == '__main__':
    M.load_clear_choir()                 # فقط برای سازگاری با mix(); سطحش صفر است
    murmur = M.murmur_bed()
    intro = smooth_intro() * db(-4 + M.CHEER_REF)
    levels = {}
    for k in (ONLY or list(STYLES)):
        print('building', k)
        crowd = wordless(k)
        wav = M.mix(cfg_v8(intro + crowd), None, murmur, f'H_v8_{k}')
        levels[k] = {f'{t0:.0f}-{t1:.0f}': round(20 * np.log10(rms(crowd[:, int(t0 * SR):int(t1 * SR)])) -
                                               20 * np.log10(rms(M.voc[:, int(t0 * SR):int(t1 * SR)])), 1)
                     for (t0, t1), _ in SECTIONS}
        print(wav, levels[k])
    json.dump(levels, open(os.path.join(OUT, 'v8_levels.json'), 'w'), indent=1)
