"""
نسخه ۱۰: همخوانی جمعیت از صداهای تبدیل‌شده با Kits.ai (هر فایل = یک آدم واقعاً متفاوت)

ورودی: recordings/ai/*.aac (کُرس اول خواننده ۲، تبدیل‌شده به ۴ صدای مختلف؛ Pitch Correction خاموش)
- صدای اصلی فقط در کانال راست است (کانال چپ نسخه‌ی معکوس و ضعیف دارد)، پس فقط کانال راست استفاده می‌شود.
- هر فایل ۱.۵ تا ۲.۴ ثانیه سکوت اول دارد؛ با هم‌بستگی پوش صدا با وکال خواننده ۲ هم‌زمان می‌شود (بدون جابه‌جایی در طول فایل).
- هر صدا ۳ بار با زمان‌بندی متفاوت استفاده می‌شود (۱۲ نفر): تأخیر متغیر، بلندی متغیر، جا انداختن جمله‌ها.

اجرا:  python3 crowd_v10.py <WORK_DIR> <OUT_DIR> <REC_DIR>
"""
import sys, os, glob, json
import numpy as np
import soundfile as sf

WORK, OUT, REC = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], WORK, OUT]
import make_live as M
from make_live import (SR, N, db, rms, pan, env, Pedalboard, HighpassFilter, LowpassFilter,
                       PeakFilter, Reverb, PitchShift, wander_delay)
import crowd_v8 as V8
from scipy.signal import correlate

CH1 = (68.3, 108.55)
SRC_T0 = 68.0          # فایل singer2_chorus1_1m08s از ثانیه ۶۸ آهنگ شروع می‌شود
HALL_ROOM = 0.82       # RT60 ≈ ۲.۶ ثانیه


def load_voice(path):
    tmp = os.path.join(WORK, 'tmp_voice.wav')
    os.system(f'ffmpeg -y -loglevel error -i "{path}" -ar {SR} -ac 2 "{tmp}"')
    x, _ = sf.read(tmp, dtype='float32', always_2d=True)
    return x[:, 1].copy()                       # فقط کانال راست


def align(y, ref):
    """تأخیر فایل نسبت به وکال مرجع، با هم‌بستگی پوش صدا (دقت ۱ میلی‌ثانیه)."""
    k = 44
    e = lambda z: np.abs(z)[:len(z) // k * k].reshape(-1, k).mean(1)
    ey, er = e(y), e(ref)
    c = correlate(ey - ey.mean(), er - er.mean(), 'full')
    return (np.argmax(c) - (len(er) - 1)) * k / SR


def build(level_db, seed=5):
    ref = M.voc[:, int(SRC_T0 * SR):int((SRC_T0 + 41) * SR)].mean(0)
    files = sorted(glob.glob(os.path.join(REC, '*.aac')) + glob.glob(os.path.join(REC, '*.wav')) +
                   glob.glob(os.path.join(REC, '*.mp3')) + glob.glob(os.path.join(REC, '*.m4a')))
    r = np.random.default_rng(seed)
    W = np.zeros((2, N), np.float32); Mn = np.zeros((2, N), np.float32)
    info = []
    seg = M.voc[:, int(CH1[0] * SR):int(CH1[1] * SR)].mean(0)
    phr = [(CH1[0] + a, CH1[0] + b) for a, b in V8.phrases(seg)]
    for f in files:
        y = load_voice(f)
        lag = align(y, ref)
        y = y[int(max(lag, 0) * SR):]
        full = np.zeros(N, np.float32); a = int(SRC_T0 * SR)
        full[a:a + len(y)] = y[:N - a]
        # صدای مرد = یک اکتاو پایین‌تر از خواننده
        import librosa
        s16 = librosa.resample(y[int(5 * SR):int(20 * SR)], orig_sr=SR, target_sr=16000)
        p, _, _ = librosa.pyin(s16, fmin=80, fmax=900, sr=16000)
        male = np.nanmedian(p) < 220
        info.append(dict(file=os.path.basename(f), lag=round(lag, 3), male=bool(male)))
        for inst in range(3):                               # هر صدا ۳ نفر
            z = Pedalboard([PitchShift(semitones=float(r.uniform(-0.1, 0.1)))])(full, SR) if inst else full.copy()
            base = [0.0, 0.035, 0.07][inst] + r.uniform(0.0, 0.02)
            z = wander_delay(z, base + 0.015, 0.012, r.uniform(0.15, 0.4), r)
            z = z * db(3 * V8_slow(N, 0.25, r))
            gate = np.ones(N, np.float32); t = np.arange(N) / SR
            for p0, p1 in phr:
                if r.random() < 0.15:
                    gate *= 1 - np.clip((t - p0) / 0.15, 0, 1) * np.clip((p1 - t) / 0.15, 0, 1)
            z = z * gate
            p_ = r.uniform(-1, 1) if not male else r.uniform(-0.6, 0.6)
            (Mn if male else W)[:] += pan(z, p_) * r.uniform(0.7, 1.0)
    hall = Reverb(room_size=HALL_ROOM, damping=0.5, wet_level=0.4, dry_level=0.7, width=1.0)
    W = Pedalboard([HighpassFilter(150), LowpassFilter(7000), PeakFilter(3000, -2.5, 0.9), hall])(W, SR)
    Mn = Pedalboard([HighpassFilter(150), LowpassFilter(5000), hall])(Mn, SR)
    sl = slice(int(CH1[0] * SR), int(CH1[1] * SR))
    if rms(Mn[:, sl]) > 1e-6:
        Mn = Mn / rms(Mn[:, sl]) * rms(W[:, sl]) * db(-6)
    crowd = W + Mn
    lead = rms(M.voc[:, sl] * db(2.0))
    g = db(level_db) * lead / rms(crowd[:, sl])
    e = env(N, [(CH1[0] - 0.3, CH1[1] + 3.0, 0, 0.8, 2.0)])
    return W * g * e, Mn * g * e, info


def V8_slow(n, rate, r):
    k = max(4, int(n / SR * rate) + 2)
    c = np.interp(np.linspace(0, k - 1, n), np.arange(k), r.uniform(-1, 1, k))
    return np.convolve(c, np.ones(4096) / 4096, 'same')


def write(path, x):
    sf.write(path + '.wav', x.T, SR, subtype='PCM_16')
    os.system(f'ffmpeg -y -loglevel error -i "{path}.wav" -b:a 320k "{path}.mp3" && rm "{path}.wav"')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    M.load_clear_choir()
    murmur = M.murmur_bed()
    intro = V8.smooth_intro() * db(-4 + M.CHEER_REF)
    cut = lambda x: x[:, int(68 * SR):int(88 * SR)]
    report = {}
    for lv in (-9, -6, -3):
        W, Mn, info = build(lv)
        crowd = W + Mn
        name = f'J_v10_ch1_{-lv}dB'
        wav = M.mix(V8.cfg_v8(intro + crowd), None, murmur, name)
        x, _ = sf.read(wav, dtype='float32', always_2d=True)
        write(os.path.join(OUT, f'preview_{-lv}dB'), cut(x.T))
        if lv == -6:
            write(os.path.join(OUT, 'preview_crowd_only'), cut(crowd))
        else:
            os.remove(wav)
        sl = slice(int(CH1[0] * SR), int(CH1[1] * SR))
        report[lv] = dict(lead=round(20 * np.log10(rms(M.voc[:, sl] * db(2))), 1),
                          women=round(20 * np.log10(rms(W[:, sl])), 1),
                          men=round(20 * np.log10(rms(Mn[:, sl]) + 1e-12), 1), voices=info)
    json.dump(report, open(os.path.join(OUT, 'v10_levels.json'), 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))
