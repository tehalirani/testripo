"""
نسخه ۷: همخوانی جمعیت زیر خواننده ۲ در کُرس‌ها، بر پایه D_final_v4 (بقیه آهنگ بدون تغییر)

اجرا:
  python3 crowd_v7.py <WORK_DIR> <OUT_DIR> test          -> test_a..test_d (1:08 تا 1:28) + فایل‌های جدای هر گروه
  python3 crowd_v7.py <WORK_DIR> <OUT_DIR> full <a|b|c|d> -> کل آهنگ با تنظیمات انتخاب‌شده
"""
import sys, os, json
import numpy as np
import soundfile as sf

WORK, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
PICK = sys.argv[4] if len(sys.argv) > 4 else None
sys.argv = [sys.argv[0], WORK, OUT]
import make_live as M
from make_live import (SR, N, db, rms, pan, fade, env, time_stretch, wander_delay, Pedalboard,
                       HighpassFilter, LowpassFilter, PeakFilter, Reverb)

CHORUSES = {'ch1': (68.3, 108.55), 'ch2': (164.3, 202.0), 'ch3': (220.95, 242.1)}
TARGET_DB = {'ch1': -8.0, 'ch2': -5.0, 'ch3': -3.0}     # جمعیت نسبت به خواننده
LEAD_BOOST = {'ch1': 2.0, 'ch2': 2.5, 'ch3': 1.5}       # از D_final_v4
PRE, POST = 0.3, 3.0                                     # حاشیه (دنباله ریورب)
HALL_ROOM = 0.82                                         # RT60 اندازه‌گیری‌شده ≈ ۲.۶ ثانیه

# ۴ نسخه آزمایشی
TESTS = {
    'a': dict(f_shift=(1.0, 1.5), people=20, level=0.0),
    'b': dict(f_shift=(2.0, 2.5), people=20, level=0.0),
    'c': dict(f_shift=(2.0, 2.5), people=30, level=+2.0),
    'd': dict(f_shift=(2.5, 3.0), people=30, level=+3.0),
    # نسخه کامل: تعداد و رنگ صدای c با سطوح دقیق مشخصات دانشجو (‎-8/‎-5/‎-3)
    'c0': dict(f_shift=(2.0, 2.5), people=30, level=0.0),
}


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


def slow_curve(n, rate, r, lo=-1.0, hi=1.0):
    k = max(4, int(n / SR * rate) + 2)
    c = np.interp(np.linspace(0, k - 1, n), np.arange(k), r.uniform(lo, hi, k))
    return np.convolve(c, np.ones(4096) / 4096, 'same')


def blocky(c, step=4096):
    """time_stretch تغییرات سریع‌تر از ۱۰۲۴ نمونه را نادیده می‌گیرد؛ هر ۴۰۹۶ نمونه ثابت می‌کنیم تا سریع‌تر شود."""
    idx = (np.arange(len(c)) // step) * step
    return c[idx].astype(np.float64)


def fit(y, n):
    y = y[:n]
    return np.pad(y, (0, n - len(y)))


def person(seg, kind, cfg, phr, r):
    n = len(seg)
    sign = r.choice([-1, 1])
    cents = sign * r.uniform(15, 30) + 8 * slow_curve(n, 0.3, r)        # انحراف کوک ۱۵ تا ۳۰ سنت که آرام تغییر می‌کند
    if kind == 'w':
        fs = r.uniform(*cfg['f_shift'])                                  # فرمنت بالاتر، متفاوت برای هر نفر
        y = time_stretch(seg[None], SR, 1.0, blocky(fs + cents / 100), preserve_formants=False)[0]
        y = time_stretch(fit(y, n)[None], SR, 1.0, -fs, preserve_formants=True)[0]
    else:
        g = r.uniform(1.0, 2.0)                                          # یک اکتاو پایین + فرمنت ۱ تا ۲ نیم‌پرده پایین‌تر
        y = time_stretch(seg[None], SR, 1.0, blocky(-12 + cents / 100), preserve_formants=True)[0]
        y = time_stretch(fit(y, n)[None], SR, 1.0, -g, preserve_formants=False)[0]
        y = time_stretch(fit(y, n)[None], SR, 1.0, g, preserve_formants=True)[0]
    y = fit(y, n).astype(np.float32)
    # زمان‌بندی: تأخیر پایه ۲۰ تا ۸۰ میلی‌ثانیه + جلو و عقب رفتن آرام (بدون تأخیر ثابت)
    base = r.uniform(0.02, 0.08)
    depth = min(0.015, base - 0.008)
    y = wander_delay(y, base, depth, r.uniform(0.15, 0.4), r)
    # بلندی: ±۳ دسی‌بل آرام
    y = y * db(3 * slow_curve(n, 0.25, r))
    # بعضی جمله‌ها را نمی‌خواند
    gate = np.ones(n, np.float32); t = np.arange(n) / SR
    for a, b in phr:
        if r.random() < 0.15:
            w = np.clip((t - a) / 0.15, 0, 1) * np.clip((b - t) / 0.15, 0, 1)
            gate *= 1 - w
    y = y * gate
    p = r.uniform(-1, 1) if kind == 'w' else r.uniform(-0.8, 0.8)
    return pan(y, p) * r.uniform(0.7, 1.0)


def build_window(name, cfg, seed):
    t0, t1 = CHORUSES[name]
    a, b = int((t0 - PRE) * SR), int((t1 + POST) * SR)
    seg = M.voc[:, a:b].mean(0).astype(np.float32)
    seg[int((t1 - t0 + PRE) * SR):] *= np.linspace(1, 0, len(seg) - int((t1 - t0 + PRE) * SR)) ** 4
    phr = [(x + 0.0, y + 0.0) for x, y in phrases(seg)]
    r = np.random.default_rng(seed)
    nw = round(cfg['people'] * 0.7); nm = cfg['people'] - nw
    W = sum(person(seg, 'w', cfg, phr, r) for _ in range(nw))
    Mn = sum(person(seg, 'm', cfg, phr, r) for _ in range(nm))
    hall = Reverb(room_size=HALL_ROOM, damping=0.5, wet_level=0.45, dry_level=0.65, width=1.0)
    W = Pedalboard([HighpassFilter(150), LowpassFilter(7000),
                    PeakFilter(3000, -2.5, 0.9),          # ۲ تا ۴ کیلوهرتز: جا برای خواننده
                    hall])(W.astype(np.float32), SR)
    Mn = Pedalboard([HighpassFilter(150), LowpassFilter(5000), hall])(Mn.astype(np.float32), SR)
    Mn = Mn / rms(Mn) * rms(W) * db(-6)                  # مردها ۶ دسی‌بل آرام‌تر
    mur = M.murmur_bed(M.POOL_SAFE)[:, :b - a]
    mur = mur / rms(mur) * rms(W) * db(-20)               # همهمه‌ی واقعی خیلی کم
    return a, b, W, Mn, mur


def assemble(cfg, seed=77, windows=('ch1', 'ch2', 'ch3')):
    parts = {k: np.zeros((2, N), np.float32) for k in ('w', 'm', 'mur')}
    levels = {}
    for i, name in enumerate(windows):
        t0, t1 = CHORUSES[name]
        a, b, W, Mn, mur = build_window(name, cfg, seed + i)
        crowd = W + Mn + mur
        lead = M.voc[:, int(t0 * SR):int(t1 * SR)] * db(LEAD_BOOST[name])
        lead_db = 20 * np.log10(rms(lead))
        crowd_db = 20 * np.log10(rms(crowd[:, int(PRE * SR):int((t1 - t0 + PRE) * SR)]))
        g = db(TARGET_DB[name] + cfg['level'] + lead_db - crowd_db)
        e = env(b - a, [(0, (b - a) / SR, 0, 0.8, 1.5)]) if True else 1
        for k, x in (('w', W), ('m', Mn), ('mur', mur)):
            parts[k][:, a:b] += x * g * e
        seg = lambda x: x[:, a + int(PRE * SR):a + int((t1 - t0 + PRE) * SR)]
        levels[name] = dict(lead=round(lead_db, 1),
                            women=round(20 * np.log10(rms(seg(parts['w']))), 1),
                            men=round(20 * np.log10(rms(seg(parts['m']))), 1),
                            murmur=round(20 * np.log10(rms(seg(parts['mur']))), 1),
                            crowd_vs_lead=round(20 * np.log10(rms(seg(parts['w'] + parts['m'] + parts['mur']))) - lead_db, 1))
    return parts, levels


def base_cfg(extra):
    v = dict(M.VARIANTS['D_final_v4'])
    chorus_spans = [(M.CH1A[0], M.CH1B_E[1]), (M.CH2A[0], M.CH2B_E[1]), (220.95, 242.1)]
    inside = lambda a, b: any(a >= s - 0.01 and b <= e + 0.01 for s, e in chorus_spans)
    v['choir'] = [c for c in v['choir'] if not inside(c[0], c[1])]   # همخوانی قبلی کُرس‌ها برداشته شد
    v['lead_dry'] = list(CHORUSES.values())
    # اینترو: به‌جای شعار «بزن باران»، جیغ و تشویق ریتمیک تا ورود خواننده ۱، و یک اوج ناگهانی برای خواننده ۱
    v['chant_loop'] = []
    v['rhythm_cheer'] = [(12.5, 29.2, -1)]
    v['cheers'] = [c for c in v['cheers'] if not (c[3] == 27.4)] + [
        (M.SCREAM, 0.0, 2.0, 28.95, -3, 0.03, 1.0),
        (M.POOL_SAFE, 0.0, 7.0, 28.9, -4, 0.25, 4.5)]
    v['extra_fx'] = extra
    return v


def write(path, x):
    sf.write(path + '.wav', x.T, SR, subtype='PCM_16')
    os.system(f'ffmpeg -y -loglevel error -i "{path}.wav" -b:a 320k "{path}.mp3" && rm "{path}.wav"')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    M.load_clear_choir()
    murmur = M.murmur_bed()
    cut = lambda x: x[:, int(68 * SR):int(88 * SR)]
    if MODE == 'test':
        report = {}
        for k, cfg in TESTS.items():
            print('test', k)
            parts, levels = assemble(cfg, windows=('ch1',))
            extra = parts['w'] + parts['m'] + parts['mur']
            wav = M.mix(base_cfg(extra), None, murmur, f'_full_test_{k}')
            x, _ = sf.read(wav, dtype='float32', always_2d=True); os.remove(wav)
            write(os.path.join(OUT, f'test_{k}'), cut(x.T))
            write(os.path.join(OUT, f'test_{k}_women_only'), cut(parts['w']))
            write(os.path.join(OUT, f'test_{k}_men_only'), cut(parts['m']))
            write(os.path.join(OUT, f'test_{k}_crowd_only'), cut(extra))
            report[k] = dict(cfg, **levels['ch1'])
        json.dump(report, open(os.path.join(OUT, 'test_levels.json'), 'w'), indent=1)
        print(json.dumps(report, indent=1))
    else:
        cfg = TESTS[PICK]
        parts, levels = assemble(cfg)
        extra = parts['w'] + parts['m'] + parts['mur']
        wav = M.mix(base_cfg(extra), None, murmur, f'G_final_v7_{PICK}')
        json.dump(levels, open(os.path.join(OUT, f'G_final_v7_{PICK}_levels.json'), 'w'), indent=1)
        print(json.dumps(levels, indent=1)); print(wav)
