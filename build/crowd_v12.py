"""
نسخه ۱۲: همخوانی واقعی جمعیت از نسخه لایو واقعی آهنگ

۱. وکال نسخه لایو (از قبل با UVR-MDX-NET-Voc_FT جدا شده) با مدل Karaoke (UVR_MDXNET_KARA_2) به
   «خواننده اصلی» و «صداهای پشت» جدا شد. صدای جمعیت در بخش دوم است.
   جمعیت در لایو در ۱:۳۸ تا ۲:۱۲ (۱:۵۰ تا ۲:۰۲ بدون خواننده)، ۳:۱۲ تا ۳:۳۳ و ۳:۴۸ تا ۴:۱۳ می‌خواند.
۲. هم‌ترازی زمانی: DTW روی کروماگرام سازهای دو نسخه (کرومای لایو ۲ نیم‌پرده جابه‌جا شده، چون لایو
   ۲ نیم‌پرده بالاتر است). دقت در جاهایی که خواننده می‌خواند با هم‌بستگی پوش صدا چک شد: حدود ۰.۱ ثانیه.
۳. صدای جمعیت در بلوک‌های ۱ ثانیه‌ای (با ۰.۲۵ ثانیه هم‌پوشانی) با time_stretch به زمان آهنگ ما رسانده شد
   و در همان مرحله ۲ نیم‌پرده پایین آمد (با حفظ فرمنت).

اجرا:  python3 crowd_v12.py <WORK_DIR> <OUT_DIR> <REC_DIR>
"""
import sys, os, json
import numpy as np
import soundfile as sf

WORK, OUT, REC = sys.argv[1], sys.argv[2], sys.argv[3]
import crowd_v11 as V11
V10, M, V8 = V11.V10, V11.M, V11.V8
from make_live import SR, N, db, rms, env, time_stretch, Pedalboard, HighpassFilter, LowpassFilter, PeakFilter

WINDOWS = [(87.0, 113.0), (182.5, 203.0), (219.5, 237.0)]   # زمان آهنگ ما
LEAD_REF = (87.85, 99.0)                                    # جایی که هم خواننده ما و هم جمعیت لایو هست


def load_backing():
    p = os.path.join(WORK, 'kara', 'livevoc_(Instrumental)_UVR_MDXNET_KARA_2.wav')
    x, sr = sf.read(p, dtype='float32', always_2d=True)
    return x.T.copy()


def time_map():
    wp = np.load(os.path.join(WORK, 'dtw_studio_live.npy'))
    s, l = wp[:, 0], wp[:, 1]
    grid = np.arange(0, s.max(), 0.05)
    lg = np.interp(grid, s, l)
    k = 40                                          # هموارسازی ۲ ثانیه‌ای
    lg = np.convolve(np.pad(lg, k, mode='edge'), np.ones(2 * k + 1) / (2 * k + 1), 'valid')
    lg = np.maximum.accumulate(lg)
    return lambda t: np.interp(t, grid, lg)


def warp_window(B, tl, t0, t1, blk=1.0, ov=0.25):
    out = np.zeros((2, N), np.float32)
    L = int((blk + ov) * SR); win = np.ones(L, np.float32)
    r = int(ov * SR); win[:r] = np.linspace(0, 1, r); win[-r:] = np.linspace(1, 0, r)
    t = t0
    while t < t1:
        a, b = tl(t), tl(t + blk + ov)
        src = B[:, int(a * SR):int(b * SR)]
        if src.shape[1] > 1000:
            f = (b - a) / (blk + ov)                           # >۱ یعنی تندتر (لایو سریع‌تر است)
            y = time_stretch(src, SR, float(f), -2.0, high_quality=True, preserve_formants=True)
            y = y[:, :L]
            if y.shape[1] < L: y = np.pad(y, ((0, 0), (0, L - y.shape[1])))
            i = int(t * SR)
            out[:, i:i + L] += y * win
        t += blk
    return out


def build_live_crowd():
    B = load_backing(); tl = time_map()
    out = np.zeros((2, N), np.float32)
    for t0, t1 in WINDOWS:
        out += warp_window(B, tl, t0, t1) * env(N, [(t0, t1, 0, 0.6, 1.5)])
    out = Pedalboard([HighpassFilter(150), LowpassFilter(9000), PeakFilter(3000, -2.0, 0.9)])(out, SR)
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    M.load_clear_choir()
    murmur = M.murmur_bed()
    intro = V8.smooth_intro() * db(-4 + M.CHEER_REF)
    live = build_live_crowd()
    sl = slice(int(LEAD_REF[0] * SR), int(LEAD_REF[1] * SR))
    lead_db = 20 * np.log10(rms(M.voc[:, sl] * db(2)))
    report = {}
    cut = lambda x: x[:, int(85 * SR):int(115 * SR)]           # ۱:۲۵ تا ۱:۵۵
    for lv in (-6, -3):
        g = db(lv + lead_db - 20 * np.log10(rms(live[:, sl])))
        crowd = live * g
        name = f'L_v12_live_{-lv}dB'
        wav = M.mix(V8.cfg_v8(intro + crowd), None, murmur, name)
        x, _ = sf.read(wav, dtype='float32', always_2d=True)
        V10.write(os.path.join(OUT, f'preview_v12_{-lv}dB'), cut(x.T))
        if lv == -3:
            V10.write(os.path.join(OUT, 'preview_v12_crowd_only'), cut(crowd))
        report[lv] = {f'{a:.0f}-{b:.0f}': round(20 * np.log10(rms(crowd[:, int(a * SR):int(b * SR)])) - lead_db, 1)
                      for a, b in WINDOWS}
    # ترکیب: جمعیت واقعی + جمعیت Kits.ai نسخه ۱۱ در نیمه اول کُرس اول (جایی که جمعیت لایو نمی‌خواند)
    W, Mn, info, segs = V11.build(-8)
    kits = (W + Mn) * env(N, [(68.3, 88.5, 0, 0.8, 1.5)])
    g = db(-3 + lead_db - 20 * np.log10(rms(live[:, sl])))
    wav = M.mix(V8.cfg_v8(intro + live * g + kits), None, murmur, 'L_v12_live_plus_kits')
    report['mix_with_kits'] = 'kits crowd -8 dB in 1:08-1:28 + live crowd -3 dB'
    json.dump(report, open(os.path.join(OUT, 'v12_levels.json'), 'w'), indent=1)
    print(json.dumps(report, indent=1))
