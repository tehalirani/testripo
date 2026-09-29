"""
نسخه ۱۷: صدای خود دانشجو (۳ ضبط با صداهای مختلف) زیر ۰:۴۹ تا ۱:۴۸، روی پایه‌ی نسخه ۱۶

هر ضبط: «بزن باران که من هم ابری‌ام… سرگردان بماند» + کُرس اول (ضبط ۳ فقط تا اولین «هوا هوای خاطرات اوست»).
دانشجو بدون هم‌زمانی با آهنگ خوانده: کوک متفاوت (۳ تا ۶ نیم‌پرده) و سرعت کمی متفاوت. پس:
  ۱. هم‌ترازی زمانی: هر بخش (پیش‌کُرس، کُرس) جدا، با مرزهای جمله‌ها و بعد DTW روی ویژگی‌های مستقل از کوک
     (بلندی، وجود صدا، شروع هجاها، تغییر نت از فریمی به فریم بعد).
  ۲. تنظیم کوک: نت هر لحظه به نت خواننده در همان لحظه (در اکتاوی که به صدای دانشجو نزدیک‌تر است) منتقل می‌شود
     (شبیه اتوتیون با مرجع، time_stretch با آرایه‌ی تغییر کوک، با حفظ فرمنت).
  ۳. از هر ضبط ۲ نفر با humanize نسخه ۱۱ (تأخیر ۱۵۰ تا ۲۸۰ میلی‌ثانیه، کشش، بلندی، جا انداختن جمله).

اجرا:  python3 crowd_v17.py <WORK_DIR> <OUT_DIR> <REC_DIR>
"""
import sys, os, json
import numpy as np
import soundfile as sf
import librosa

WORK, OUT, REC = sys.argv[1], sys.argv[2], sys.argv[3]
import crowd_v16 as V16
V14, V12, V11, V10, M, V8 = V16.V14, V16.V12, V16.V12.V11, V16.V10, V16.M, V16.V8
from make_live import (SR, N, db, rms, env, time_stretch, Pedalboard, HighpassFilter, LowpassFilter,
                       PeakFilter, Reverb, to_sides)

HOP = 0.05
SONG_PRE, SONG_CH = (48.9, 65.2), (68.3, 108.55)
TAKES = {   # فایل: (کانال، بخش پیش‌کُرس در ضبط، بخش کُرس در ضبط، پایان کُرس در آهنگ)
    'self_voice1': ('R', (9.64, 26.44), (29.4, 68.08), 108.55),
    'self_voice2': ('mix', (0.94, 19.24), (20.44, 59.86), 108.55),
    'self_voice3': ('mix', (0.34, 18.22), (19.26, 39.58), 86.6),
}
REL_DB = -6.0


def f0_track(x):
    y = librosa.resample(x.astype(np.float32), orig_sr=SR, target_sr=16000)
    f, vf, _ = librosa.pyin(y, fmin=70, fmax=900, sr=16000, hop_length=800)
    m = librosa.hz_to_midi(f); m[~vf] = np.nan
    return m


def feats(x, f0):
    h = int(HOP * SR)
    e = librosa.feature.rms(y=x.astype(np.float32), frame_length=2 * h, hop_length=h)[0]
    o = librosa.onset.onset_strength(y=x.astype(np.float32), sr=SR, hop_length=h)
    n = min(len(e), len(o), len(f0))
    e = 20 * np.log10(e[:n] + 1e-6); e = (e - e.max()) / 30
    o = o[:n] / (o[:n].max() + 1e-9)
    v = (~np.isnan(f0[:n])).astype(float)
    d = np.nan_to_num(np.diff(f0[:n], prepend=np.nan)); d = np.clip(d, -3, 3) / 3
    return np.stack([e, o, v, d * v])


def align_section(xr, fr, r0, r1, xs, fs, s0, s1):
    a, b = int(r0 / HOP), int(r1 / HOP); c, d = int(s0 / HOP), int(s1 / HOP)
    FR = feats(xr, fr)[:, a:b]; FS = feats(xs, fs)[:, c:d]
    _, wp = librosa.sequence.dtw(X=FR, Y=FS, metric='euclidean')
    wp = wp[::-1]
    rt = r0 + wp[:, 0] * HOP; st = s0 + wp[:, 1] * HOP
    # نگاشت یکنوا و هموار: زمان آهنگ -> زمان ضبط
    grid = np.arange(s0, s1, HOP)
    m = np.interp(grid, st, rt)
    m = np.convolve(np.pad(m, 10, mode='edge'), np.ones(21) / 21, 'valid')
    return grid, np.maximum.accumulate(m)


def build_take(name):
    ch, pre, cho, s_end = TAKES[name]
    x, _ = sf.read(os.path.join(WORK, 'self', f'{name}_(Vocals)_UVR-MDX-NET-Voc_FT.wav'), dtype='float32', always_2d=True)
    x = x[:, 1] if ch == 'R' else x.mean(1)
    fr = f0_track(x)
    song = M.voc.mean(0)
    fs_all = np.load(os.path.join(WORK, 'song_f0_50ms.npy'))
    g1, m1 = align_section(x, fr, *pre, song, fs_all, *SONG_PRE)
    g2, m2 = align_section(x, fr, *cho, song, fs_all, SONG_CH[0], s_end)
    grid = np.concatenate([g1, g2]); rmap = np.concatenate([m1, m2])
    # ۲. کوک: برای هر فریم ضبط، نت هدف = نت خواننده در زمان متناظر آهنگ، در نزدیک‌ترین اکتاو به صدای دانشجو
    rec_t = np.arange(len(fr)) * HOP
    song_t = np.interp(rec_t, rmap, grid, left=np.nan, right=np.nan)
    tgt = np.interp(song_t, np.arange(len(fs_all)) * HOP, np.nan_to_num(fs_all, nan=-1))
    tgt[(tgt < 0) | np.isnan(song_t)] = np.nan
    med = np.nanmedian(fr)
    tgt = tgt + 12 * np.round((med - np.nanmedian(tgt)) / 12)     # اکتاو نزدیک به صدای دانشجو
    shift = tgt - fr
    shift = shift - 12 * np.round(shift / 12)                        # کمترین جابه‌جایی (بدون پرش اکتاو)
    ok = ~np.isnan(shift)
    shift = np.interp(np.arange(len(shift)), np.where(ok)[0], shift[ok]) if ok.any() else np.zeros(len(shift))
    shift = np.convolve(np.pad(shift, 2, mode='edge'), np.ones(5) / 5, 'valid')
    arr = np.interp(np.arange(len(x)), np.arange(len(shift)) * HOP * SR, shift).astype(np.float64)
    arr = arr[(np.arange(len(x)) // 1024) * 1024]                   # هر ۱۰۲۴ نمونه ثابت (سریع‌تر)
    tuned = time_stretch(x[None], SR, 1.0, arr, high_quality=True, preserve_formants=True)[0]
    tuned = np.pad(tuned, (0, max(0, len(x) - len(tuned))))[:len(x)]
    # ۱. زمان: بلوک‌های ۰.۵ ثانیه‌ای زمان آهنگ، هر کدام از بازه‌ی متناظر ضبط، کشیده شده
    out = np.zeros(N, np.float32)
    blk, ov = 0.5, 0.12; L = int((blk + ov) * SR)
    win = np.ones(L, np.float32); r_ = int(ov * SR); win[:r_] = np.linspace(0, 1, r_); win[-r_:] = np.linspace(1, 0, r_)
    for s0, s1 in (SONG_PRE, (SONG_CH[0], s_end)):
        t = s0
        while t < s1 - 0.05:
            a = np.interp(t, grid, rmap); b = np.interp(min(t + blk + ov, s1 + ov), grid, rmap)
            src = tuned[int(a * SR):int(b * SR)]
            if len(src) > 2000:
                y = time_stretch(src[None], SR, float((b - a) / (blk + ov)), 0.0)[0][:L]
                y = np.pad(y, (0, L - len(y)))
                i = int(t * SR); out[i:i + L] += y * win
            t += blk
    return out, dict(key_shift_median=round(float(np.nanmedian(tgt - fr)), 2),
                     voice_median=librosa.midi_to_note(med), octave_target=librosa.midi_to_note(np.nanmedian(tgt)))


def build_self_crowd(seed=17):
    r = np.random.default_rng(seed)
    segs = V11.segments(48.9, 108.55)
    total = np.zeros((2, N), np.float32); info = {}
    for name in TAKES:
        tr, inf = build_take(name); info[name] = inf
        for inst in range(2):
            z = V11.humanize(tr, segs, r) if inst else tr.copy()
            total += M.pan(z, r.uniform(-0.9, 0.9)) * r.uniform(0.75, 1.0)
    total = Pedalboard([HighpassFilter(120), LowpassFilter(7000), PeakFilter(3000, -2.5, 0.9),
                        Reverb(room_size=0.8, damping=0.55, wet_level=0.35, dry_level=0.75, width=1.0)])(total, SR)
    total = to_sides(total, mid_db=-4.0) * env(N, [(48.6, 109.5, 0, 0.6, 2.0)])
    sl = slice(int(68.3 * SR), int(108.55 * SR))
    lead = rms(M.voc[:, sl] * db(2))
    return total * db(REL_DB) * lead / rms(total[:, sl]), info


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    M.load_clear_choir()
    murmur = M.murmur_bed()
    src = V14.crowd_source(); tl = V14.time_map()
    live = V12.warp_window(src, tl, *V14.SPAN)
    live = Pedalboard([HighpassFilter(150), LowpassFilter(8000), PeakFilter(3000, -2.0, 0.9),
                       Reverb(room_size=0.8, damping=0.6, wet_level=0.25, dry_level=0.85, width=1.0)])(live, SR)
    live *= env(N, [(V14.SPAN[0], V14.SPAN[1], 0, 1.0, 3.0)])
    grid = np.arange(60, 130, 0.05); tg = tl(grid)
    rs = slice(int(np.interp(V14.CROWD_REF_LIVE[0], tg, grid) * SR), int(np.interp(V14.CROWD_REF_LIVE[1], tg, grid) * SR))
    sl = slice(int(V12.LEAD_REF[0] * SR), int(V12.LEAD_REF[1] * SR))
    lead_db = 20 * np.log10(rms(M.voc[:, sl] * db(2)))
    live *= db(-3 + lead_db - 20 * np.log10(rms(live[:, rs])))
    chant, placed = V16.chant_under_lines()
    me, info = build_self_crowd()
    intro = V8.smooth_intro() * db(-4 + M.CHEER_REF)
    wav = M.mix(V8.cfg_v8(intro + chant + live + me), None, murmur, 'Q_v17_full')
    x, _ = sf.read(wav, dtype='float32', always_2d=True); x = x.T
    orig = sf.info(os.path.join(WORK, 'song.wav')).frames
    y = x[:, :orig].copy(); f = int(5 * SR); y[:, -f:] *= np.linspace(1, 0, f) ** 2
    sf.write(os.path.join(OUT, 'Q_v17.wav'), y.T, SR, subtype='PCM_16')
    V10.write(os.path.join(OUT, 'preview_v17'), y[:, int(46 * SR):int(90 * SR)])
    V10.write(os.path.join(OUT, 'preview_v17_my_voices_only'), me[:, int(46 * SR):int(110 * SR)])
    json.dump(info, open(os.path.join(OUT, 'v17_info.json'), 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(info, ensure_ascii=False))
