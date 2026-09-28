"""
ساخت نسخه‌های لایو از «Ehaam - Bezan Baran»

پیش‌نیاز: جداسازی وکال/موزیک با audio-separator (مدل UVR-MDX-NET-Voc_FT)
روی نسخه استودیویی (song.wav) و نسخه لایو واقعی (reallive.wav)؛ خروجی‌ها در WORK/sep.

اجرا:  python3 make_live.py <WORK_DIR> <OUT_DIR>
"""
import sys, os
import numpy as np
import soundfile as sf
from pedalboard import time_stretch
from scipy.signal import fftconvolve, welch
from pedalboard import (Pedalboard, Reverb, HighpassFilter, LowpassFilter, LowShelfFilter,
                        HighShelfFilter, PeakFilter, Compressor, Limiter, PitchShift, Delay, Gain, Chorus)

WORK, OUT = sys.argv[1], sys.argv[2]
SR = 44100
rng_global = np.random.default_rng(7)

# ---------- زمان‌بندی آهنگ (از تحلیل: analysis/06, 07 و تشخیص ضرب) ----------
BEAT0, BEAT = 11.155, 0.60004           # اولین ضرب ورود گروه، طول هر ضرب (۱۰۰ BPM)
BAR = 4 * BEAT
def beat(k): return BEAT0 + k * BEAT

CHORUS1 = (68.3, 108.55); CHORUS1_B = (87.85, 108.55)       # خواننده ۲
CHORUS2 = (164.3, 202.0); CHORUS2_B = (183.85, 202.0)
CHORUS3 = (220.95, 242.1)
PRE1 = (48.9, 65.1); PRE2 = (144.9, 161.0); BRIDGE = (203.0, 220.4)   # اوج‌گیری خواننده ۱
BREAK = (beat(164), 125.5)                                   # بخش بی‌کلام
TAIL = 16.0                                                 # ثانیه‌های اضافه برای تشویق پایانی


def load(p):
    x, sr = sf.read(p, dtype='float32', always_2d=True)
    assert sr == SR
    return x.T.copy()          # (2, n)

def db(g): return 10 ** (g / 20)

def rms(x): return float(np.sqrt(np.mean(x ** 2)) + 1e-12)

def env(n, segs, base=-120.0):
    """منحنی گین: segs = [(t0, t1, dB, fade_in, fade_out)]"""
    g = np.full(n, db(base), dtype=np.float32)
    t = np.arange(n) / SR
    for t0, t1, d, fi, fo in segs:
        w = np.clip((t - t0) / max(fi, 1e-3), 0, 1) * np.clip((t1 - t) / max(fo, 1e-3), 0, 1)
        g = np.maximum(g, db(d) * w.astype(np.float32))
    return g

def place(dst, clip, t, gain_db=0.0):
    i = int(t * SR)
    if i >= dst.shape[1]: return
    m = min(clip.shape[1], dst.shape[1] - i)
    dst[:, i:i + m] += clip[:, :m] * db(gain_db)

def fade(x, fi=0.05, fo=0.5):
    x = x.copy(); n = x.shape[1]
    a, b = int(fi * SR), int(fo * SR)
    if a: x[:, :a] *= np.linspace(0, 1, a)
    if b: x[:, -b:] *= np.linspace(1, 0, b)
    return x

def pan(mono, p):  # p در بازه -1 تا 1
    a = (p + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)])

def widen(x, side_db):
    m, s = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2 * db(side_db)
    return np.stack([m + s, m - s])


# ---------- منابع ----------
sep = os.path.join(WORK, 'sep')
voc = load(os.path.join(sep, 'song_(Vocals)_UVR-MDX-NET-Voc_FT.wav'))
ins = load(os.path.join(sep, 'song_(Instrumental)_UVR-MDX-NET-Voc_FT.wav'))
live_v = load(os.path.join(sep, 'reallive_(Vocals)_UVR-MDX-NET-Voc_FT.wav'))
N = voc.shape[1] + int(TAIL * SR)
voc = np.pad(voc, ((0, 0), (0, N - voc.shape[1])))
ins = np.pad(ins, ((0, 0), (0, N - ins.shape[1])))

def lclip(a, b):
    """تکه‌ای از صدای جمعیت نسخه لایو واقعی، نرمال‌شده."""
    c = live_v[:, int(a * SR):int(b * SR)]
    c = Pedalboard([HighpassFilter(120)])(c, SR)
    return c / rms(c) * 0.05

# برچسب‌ها با گوش دانشجو تأیید شد (clips_for_review)
SCREAM = lclip(78, 80)      # clip3: فقط جیغ
# در تکه‌های تشویق چند لحظه آواز بود (مثلاً ثانیه ۱۳.۵ و ۱۲۳ تا ۱۳۳ نسخه لایو) که در نسخه اول
# زیر اینترو شنیده می‌شد. با تشخیص صدای نت‌دار (pyin) فقط بازه‌های جیغ و تشویق خالی نگه داشته شد:
CLEAN_RANGES = [(14.0, 20.5), (25.0, 31.0), (110.0, 122.0), (250.0, 254.0)]
def build_pool():
    xf = int(0.3 * SR); out = None
    for a, b in CLEAN_RANGES:
        c = live_v[:, int(a * SR):int(b * SR)]
        c = Pedalboard([HighpassFilter(120)])(c, SR)
        c = c / rms(c) * 0.05
        if out is None: out = c; continue
        r = np.linspace(0, 1, xf)
        out = np.concatenate([out[:, :-xf], out[:, -xf:] * (1 - r) + c[:, :xf] * r, c[:, xf:]], axis=1)
    return out
POOL = build_pool()         # حدود ۲۸ ثانیه جیغ و تشویق خالی
# نسخه ۳: دانشجو در ثانیه ۱ تا ۵ نسخه ۲ (که از بازه ۱۱۰ تا ۱۲۲ برداشته شده بود) آواز جمعیت شنید.
# همخوانی گروهی را pyin تشخیص نمی‌دهد، پس کل clip4 کنار گذاشته شد.
CLEAN_RANGES = [(14.0, 20.5), (25.0, 31.0), (250.0, 254.0)]
POOL_SAFE = build_pool()    # حدود ۱۶ ثانیه
INTRO_V1 = lclip(4, 31)[:, :int(9 * SR)]   # اینتروی نسخه اول که دانشجو تأیید کرد (با همان نرمال‌سازی)
CHANT = lclip(44.2, 48.7)   # clip2: «بزن باران بزن» (همخوانی واقعی)

def cheer(src, start, dur, fi=0.4, fo=3.5):
    # جیغ‌ها آرام محو می‌شوند و یک‌باره قطع نمی‌شوند
    return fade(src[:, int(start * SR):int((start + dur) * SR)], fi, fo)


# ---------- شمارش درامر (چوب درام) ----------
def stick_click(seed):
    r = np.random.default_rng(seed)
    n = int(0.08 * SR); t = np.arange(n) / SR
    y = np.zeros(n)
    for f, d, a in [(2250, 0.018, 1.0), (3400, 0.012, 0.6), (5100, 0.008, 0.4), (1200, 0.025, 0.3)]:
        y += a * np.sin(2 * np.pi * f * r.uniform(0.97, 1.03) * t) * np.exp(-t / d)
    y[:int(0.002 * SR)] += r.normal(0, 0.8, int(0.002 * SR))
    return (y / np.abs(y).max()).astype(np.float32)

def count_in():
    x = np.zeros((2, N), np.float32)
    for k in range(4):
        c = pan(stick_click(k), -0.15) * 0.2
        place(x, c, beat(-4 + k))
    return Pedalboard([Reverb(room_size=0.6, wet_level=0.25, dry_level=0.9)])(x, SR)


# ---------- دست زدن ریتمیک جمعیت ----------
def claps(t0, t1, seed=3):
    r = np.random.default_rng(seed)
    x = np.zeros((2, N), np.float32)
    n = int(0.06 * SR); tt = np.arange(n) / SR
    k0 = int(np.ceil((t0 - BEAT0) / BEAT)); k1 = int((t1 - BEAT0) / BEAT)
    for k in range(k0, k1 + 1):
        for _ in range(35):                      # ۳۵ نفر
            burst = r.normal(0, 1, n) * np.exp(-tt / r.uniform(0.006, 0.02))
            burst = Pedalboard([PeakFilter(r.uniform(900, 2500), 6, 1.2), HighpassFilter(500)])(
                burst.astype(np.float32), SR)
            place(x, pan(burst, r.uniform(-1, 1)) * r.uniform(0.3, 1.0) * 0.02,
                  beat(k) + r.normal(0, 0.018))
    # ورود و خروج تدریجی
    x *= env(N, [(t0, t1, 0, 2.0, 1.0)])
    return Pedalboard([Reverb(room_size=0.85, wet_level=0.45, dry_level=0.6)])(x, SR)


# ---------- همخوانی جمعیت (ساخته‌شده از صدای خود خواننده) ----------
def build_crowd_choir(seed=11, voices=14):
    r = np.random.default_rng(seed)
    mono = voc.mean(0)
    out = np.zeros((2, N), np.float32)
    for i in range(voices):
        male = r.random() < 0.4                     # بخشی از جمعیت یک اکتاو پایین‌تر می‌خوانند
        semis = (-12.0 if male else 0.0) + r.normal(0, 0.15)   # کمی خارج از کوک
        delay = r.uniform(0.03, 0.07)                # کمی دیرتر از خواننده
        chain = Pedalboard([
            PitchShift(semitones=float(semis)),
            HighpassFilter(r.uniform(150, 260)),
            LowpassFilter(r.uniform(1800, 3200)),     # دورتر = تیره‌تر
                    ])
        y = chain(mono, SR)
        y = np.roll(y, int(delay * SR)); y[:int(delay * SR)] = 0
        # بلندی نامنظم هر نفر
        lfo = 1 + 0.35 * np.sin(2 * np.pi * r.uniform(0.05, 0.25) * np.arange(N) / SR + r.uniform(0, 6))
        out += pan(y * lfo, r.uniform(-1, 1)) * r.uniform(0.5, 1.0)
    out = Pedalboard([Reverb(room_size=0.8, damping=0.6, wet_level=0.45, dry_level=0.5, width=1.0),
                      Compressor(threshold_db=-24, ratio=3)])(out, SR)
    return out / rms(out) * rms(voc)                 # هم‌سطح با وکال اصلی (قبل از گین)


def build_clear_choir(seed=21):
    """همخوانی واضح (نسخه ۳): همان کلمات و ملودی خواننده، با فرکانس (کوک) تغییر یافته.
    - گروه آقایان: یک اکتاو پایین‌تر با حفظ فرمنت، تا کلمات واضح بمانند
    - گروه هم‌صدا: هم‌اکتاو، با ۱۰ تا ۲۵ سنت خارج از کوک
    - تأخیر کوتاه (۱۵ تا ۴۵ میلی‌ثانیه) تا مثل دوبله شدن صدا شنیده شود نه اکو
    - ریورب کم و فیلتر روشن‌تر تا کلمات قابل تشخیص باشند"""
    r = np.random.default_rng(seed)
    mono = voc.mean(0).astype(np.float32)
    octave = time_stretch(mono[None], SR, 1.0, -12.0, high_quality=True, preserve_formants=True)[0]
    out = np.zeros((2, N), np.float32)
    plan = [('oct', 5), ('uni', 5)]
    for kind, n in plan:
        for i in range(n):
            src = octave if kind == 'oct' else mono
            cents = r.uniform(10, 25) * r.choice([-1, 1])
            y = Pedalboard([PitchShift(semitones=cents / 100),
                            HighpassFilter(r.uniform(120, 200)),
                            LowpassFilter(r.uniform(4500, 6500)),
                            PeakFilter(r.uniform(700, 1400), r.uniform(-3, 3), 1.0)])(src, SR)
            d = int(r.uniform(0.015, 0.045) * SR)
            y = np.roll(y, d); y[:d] = 0
            lfo = 1 + 0.2 * np.sin(2 * np.pi * r.uniform(0.05, 0.2) * np.arange(N) / SR + r.uniform(0, 6))
            out += pan(y * lfo, r.uniform(-0.9, 0.9)) * r.uniform(0.6, 1.0)
    out = Pedalboard([Reverb(room_size=0.7, damping=0.5, wet_level=0.25, dry_level=0.85, width=1.0),
                      Compressor(threshold_db=-24, ratio=3)])(out, SR)
    return out / rms(out) * rms(voc)


def formant_shift(x, semis):
    """تغییر رنگ صدا (فرمنت) بدون تغییر نت: اول نت و فرمنت با هم بالا می‌روند،
    بعد فقط نت (با حفظ فرمنت) برمی‌گردد."""
    y = time_stretch(x[None], SR, 1.0, semis, high_quality=True, preserve_formants=False)
    y = time_stretch(y, SR, 1.0, -semis, high_quality=True, preserve_formants=True)[0][:len(x)]
    return np.pad(y, (0, len(x) - len(y)))

def wander_delay(x, base, depth, rate, r):
    """تأخیری که آرام و نامنظم تغییر می‌کند (بین base-depth و base+depth ثانیه)
    تا کپی‌ها مثل آدم‌های جدا شنیده شوند، نه صدای فلزی و رباتی."""
    n = len(x); k = max(4, int(n / SR * rate))
    ctrl = r.uniform(-1, 1, k)
    curve = np.interp(np.linspace(0, k - 1, n), np.arange(k), ctrl)
    curve = np.convolve(curve, np.ones(2048) / 2048, 'same')
    d = (base + depth * curve) * SR
    idx = np.arange(n) - d
    return np.interp(idx, np.arange(n), x, left=0.0).astype(np.float32)

def build_soft_choir(seed=31):
    """همخوانی ملایم (نسخه ۵): بیشترِ جمعیت زن‌اند.
    - ۵ صدای زنانه: همان نت، فرمنت ۲.۵ تا ۴ نیم‌پرده بالاتر
    - ۲ صدای مردانه: همان نت، فرمنت کمی پایین‌تر و آرام‌تر
    - بدون اکتاو پایین (عامل اصلی صدای رباتی در نسخه ۴)
    - تأخیر متغیر ۲۵ تا ۷۰ میلی‌ثانیه، فیلتر ملایم و ریورب سالن"""
    r = np.random.default_rng(seed)
    mono = voc.mean(0).astype(np.float32)
    out = np.zeros((2, N), np.float32)
    voices = [('f', r.uniform(2.5, 4.0), 1.0) for _ in range(5)] + [('m', r.uniform(-1.2, -0.5), 0.45) for _ in range(2)]
    for kind, fs, g in voices:
        y = formant_shift(mono, float(fs))
        y = Pedalboard([PitchShift(semitones=float(r.uniform(-0.08, 0.08))),
                        HighpassFilter(220 if kind == 'f' else 140),
                        LowpassFilter(r.uniform(3500, 5000)),
                        PeakFilter(4000, -3.0, 1.0)])(y, SR)
        y = wander_delay(y, r.uniform(0.04, 0.055), 0.02, r.uniform(0.2, 0.5), r)
        out += pan(y, r.uniform(-0.95, 0.95)) * g * r.uniform(0.7, 1.0)
    out = Pedalboard([Reverb(room_size=0.82, damping=0.55, wet_level=0.4, dry_level=0.7, width=1.0),
                      Compressor(threshold_db=-26, ratio=2.5)])(out, SR)
    return out / rms(out) * rms(voc)


def to_sides(x, mid_db=-9.0):
    """نسخه ۶: جمعیت به کناره‌های استریو می‌رود تا صدای خواننده (وسط) آن را نپوشاند."""
    m, sd = (x[0] + x[1]) / 2 * db(mid_db), (x[0] - x[1]) / 2
    return np.stack([m + sd, m - sd])


# ---------- خواننده اول در ترجیع‌بند آخر (دونفره) ----------
def ltas(x):
    f, p = welch(x, SR, nperseg=4096); return f, p

SINGER1_REF = (29.6, 45.6)          # بند اول، فقط خواننده ۱
def duet_voice(t0, t1):
    """همان ملودی خواننده ۲، یک اکتاو پایین‌تر (در محدوده صدای خواننده ۱) با حفظ فرمنت،
    و رنگ صدای نزدیک به خواننده ۱ با تطبیق طیف (EQ matching)."""
    a, b = int((t0 - 0.5) * SR), int((t1 + 0.5) * SR)
    seg = voc[:, a:b].mean(0).astype(np.float32)
    y = time_stretch(seg[None], SR, 1.0, -12.0, high_quality=True, preserve_formants=True)[0]
    ref = voc[:, int(SINGER1_REF[0] * SR):int(SINGER1_REF[1] * SR)].mean(0)
    f, pr = ltas(ref); _, py = ltas(y)
    sm = lambda p: np.convolve(p, np.ones(9) / 9, 'same')
    g = np.sqrt(sm(pr) / (sm(py) + 1e-12)); g /= np.sqrt(np.mean(g[(f > 200) & (f < 4000)] ** 2))
    g = np.clip(g, db(-9), db(9)); g[f < 90] = 0
    fir = np.fft.irfft(g); fir = np.roll(fir, len(fir) // 2) * np.hanning(len(fir))
    y = fftconvolve(y, fir, 'same').astype(np.float32)
    y = Pedalboard([Compressor(threshold_db=-20, ratio=3), HighpassFilter(90)])(y, SR)
    y = y / rms(y) * rms(seg)
    out = np.zeros((2, N), np.float32)
    place(out, pan(y, -0.25), (a / SR) + 0.012)        # ۱۲ میلی‌ثانیه اختلاف، کمی چپ
    return out * env(N, [(t0, t1, 0, 0.3, 0.8)])


def murmur_bed(src=None):
    src = POOL if src is None else src
    src = Pedalboard([LowpassFilter(2500)])(src, SR)
    reps = int(np.ceil(N / src.shape[1])) + 1
    xf = int(1.0 * SR); seg = fade(src, 1.0, 1.0)
    out = np.zeros((2, N + src.shape[1]), np.float32); pos = 0
    for _ in range(reps):
        m = min(seg.shape[1], out.shape[1] - pos)
        out[:, pos:pos + m] += seg[:, :m]; pos += seg.shape[1] - xf
        if pos >= N: break
    return out[:, :N]


# ---------- میکس ----------
def mix(cfg, choir, murmur, name):
    rv = cfg['reverb']
    # سازها: بیس کمتر، میانه پررنگ‌تر، استریو پهن‌تر (مطابق تحلیل نمونه استاد)
    I = Pedalboard([HighpassFilter(30), LowShelfFilter(90, -3.0), PeakFilter(500, 2.0, 0.8),
                    HighShelfFilter(10000, -1.5)])(ins, SR)
    I = widen(I, 2.5)
    for t0, t1, d, back in cfg.get('ins_dip', []):   # کم شدن تدریجی سازها و برگشت روی ضرب
        g = np.ones(N, np.float32); t = np.arange(N) / SR
        ramp = np.clip((t - t0) / (t1 - t0), 0, 1)
        g = 1 - (1 - db(d)) * ramp
        g[t >= back] = 1.0
        k = int(back * SR); w = int(0.25 * SR)
        g[k:k + w] = np.linspace(db(d), 1, w)
        I = I * g
    # وکال: کمپرسور و حضور؛ بدون اکو تا صدای خواننده واضح و قوی بماند
    V = Pedalboard([Compressor(threshold_db=-20, ratio=3, attack_ms=5, release_ms=120),
                    PeakFilter(3000, 1.5, 1.0), HighpassFilter(90)])(voc, SR) * db(1.0)
    for t0, t1 in cfg.get('duet', []):
        V = V + duet_voice(t0, t1) * db(-2.0)
    for t0, t1, d in cfg.get('lead_boost', []):   # نسخه ۳: خواننده روی همخوانی واضح‌تر
        V = V * (1 + (db(d) - 1) * env(N, [(t0, t1, 0, 0.5, 0.5)]))
    for t0, t1, d in cfg.get('lead_duck', []):   # لحظه‌هایی که فقط جمعیت می‌خواند
        g = 1 - (1 - db(d)) * env(N, [(t0, t1, 0, 0.25, 0.4)])
        V = V * g
    hall = Pedalboard([Reverb(room_size=rv['size'], damping=0.5, wet_level=1.0, dry_level=0.0, width=1.0)])
    vsend = V * rv['voc'] * 0.4
    if cfg.get('lead_dry'):   # نسخه ۷: خواننده ۲ در کُرس‌ها کاملاً خشک (بدون ریورب)
        vsend = vsend * (1 - env(N, [(a, b, 0, 0.5, 0.5) for a, b in cfg['lead_dry']]))
    wet = hall(I * rv['ins'] + vsend, SR)
    music = I + V + wet

    fx = np.zeros((2, N), np.float32)
    # همهمه
    if cfg.get('murmur'):
        fx += (murmur if cfg.get('choir_src') == 'wash' else MURMUR_SAFE[0]) * env(N, [(a, b, d + CHEER_REF, 2.0, 2.0) for a, b, d in cfg['murmur']])
    # جیغ و تشویق
    for src, st, dur, t, g, fi, fo in cfg['cheers']:
        place(fx, cheer(src, st, dur, fi, fo), t, g + CHEER_REF + cfg.get('cheer_gain', 0))
    for t0, reps, gains in cfg.get('chant_loop', []):
        fx += chant_loop(t0, reps, [g + CHEER_REF for g in gains])
    for t, g in cfg.get('chant', []):
        place(fx, fade(CHANT, 0.05, 0.3), t, g + CHEER_REF)
    if cfg.get('claps'):
        cl = claps(*BREAK) * db(cfg['claps'] + CLAP_REF)
        for t0, t1, d, back in cfg.get('ins_dip', []):
            cl *= 1 - (1 - db(d)) * np.clip((np.arange(N) / SR - t0) / (t1 - t0), 0, 1)
        fx += cl
    fx += count_in()
    fx = Pedalboard([Reverb(room_size=rv['size'], wet_level=0.2, dry_level=1.0)])(fx, SR)
    # همخوانی (بدون ریورب دوباره)
    ch = cfg.get('choir_src', 'wash')
    src = {'wash': lambda: choir, 'clear': lambda: CLEAR_CHOIR[0], 'soft': lambda: SOFT_CHOIR[0],
           'soft_sides': lambda: to_sides(SOFT_CHOIR[0])}[ch]()
    fx += src * env(N, [(a, b, d, 1.0, 1.2) for a, b, d in cfg['choir']])
    if cfg.get('extra_fx') is not None:   # نسخه ۷: لایه‌ی جمعیت کُرس‌ها (crowd_v7.py)
        fx += cfg['extra_fx']

    out = music + fx
    out = Pedalboard([Compressor(threshold_db=-14, ratio=2, attack_ms=20, release_ms=200),
                      Limiter(threshold_db=-1.0, release_ms=150)])(out, SR)
    # فید پایانی
    out *= env(N, [(0, N / SR - 0.05, 0, 0.001, 5.0)])
    out *= min(1.0, db(-1.0) / np.abs(out).max())      # جلوگیری از کلیپ
    os.makedirs(OUT, exist_ok=True)
    wav = os.path.join(OUT, name + '.wav')
    sf.write(wav, out.T, SR, subtype='PCM_16')
    return wav


# ---------- جیغ و تشویق مشترک: لحظه‌های ورود ----------
# تکه‌های جمعیت روی ‎-26 dBFS نرمال شده‌اند؛ موزیک حدود ‎-15 dBFS است.
CHEER_REF = 10.0   # جیغ/تشویق در لحظه‌های اوج حدود ۵ تا ۸ دسی‌بل زیر موزیک
CLAP_REF = 14.0
END = 242.5
BASE_CHEERS = [   # (منبع، شروع در منبع، مدت، زمان در آهنگ، گین، فید ورود، فید خروج)
    (POOL, 12.0, 10.0, 0.0, -17, 2.0, 3.0),     # شروع: جمعیت منتظر اجرا
    (POOL, 0.0, 6.0, 11.0, -8, 0.4, 3.0),       # ورود گروه
    (POOL, 24.0, 4.0, 29.0, -12, 0.4, 2.5),     # ورود خواننده ۱
    (SCREAM, 0.0, 2.0, 67.6, -6, 0.05, 0.8),    # تعویض خواننده: ۱ -> ۲
    (POOL, 14.0, 8.0, 67.8, -7, 0.4, 4.0),
    (POOL, 6.5, 7.0, 108.6, -8, 0.4, 3.5),      # پایان کُرس اول
    (POOL, 18.0, 6.0, 125.2, -11, 0.4, 3.0),    # تعویض: ۲ -> ۱
    (SCREAM, 0.0, 2.0, 163.6, -6, 0.05, 0.8),   # تعویض: ۱ -> ۲
    (POOL, 0.0, 8.0, 163.8, -7, 0.4, 4.0),
    (POOL, 20.0, 6.0, 202.6, -11, 0.4, 3.0),    # تعویض: ۲ -> ۱ (بریج)
    (SCREAM, 0.0, 2.0, 220.3, -5, 0.05, 0.8),   # کُرس آخر
    (POOL, 8.0, 9.0, 220.5, -6, 0.4, 4.0),
    (POOL, 0.0, 27.5, END, -6, 1.0, 8.0),       # تشویق پایانی
    (POOL, 12.0, 14.0, END + 1.5, -9, 1.0, 6.0),
]
MIX_LIGHT = dict(size=0.55, ins=0.10, voc=0.14)
MIX_MED = dict(size=0.75, ins=0.14, voc=0.20)
MIX_BIG = dict(size=0.92, ins=0.20, voc=0.28)
GAPS = [(0, 11.5, -21), BREAK + (-20,), (END, N / SR, -14)]

VARIANTS = {
    # --- سری الف: شدت متفاوت ---
    'A1_malayem': dict(reverb=MIX_LIGHT, cheer_gain=-4, cheers=BASE_CHEERS,
                       choir=[CHORUS3 + (-10,)]),
    'A2_motadel': dict(reverb=MIX_MED, cheers=BASE_CHEERS, murmur=GAPS, claps=-2,
                       choir=[CHORUS1 + (-14,), CHORUS2 + (-9,), CHORUS3 + (-5,)],
                       lead_duck=[(CHORUS3[0], CHORUS3[0] + 3.0, -22)]),
    'A3_porshoor': dict(reverb=MIX_BIG, cheer_gain=3, cheers=BASE_CHEERS,
                        murmur=[(0, N / SR, -22)] + GAPS, claps=0,
                        choir=[CHORUS1 + (-5,), CHORUS2 + (-5,), CHORUS3 + (-3,)],
                        lead_duck=[(CHORUS3[0], CHORUS3[0] + 4 * BAR, -22)],
                        chant=[(beat(4 * 4), -4), (beat(4 * 6), -3)]),
    # --- سری ب: جای همخوانی متفاوت (میکس متعادل) ---
    'B1_kors_kamel': dict(reverb=MIX_MED, cheers=BASE_CHEERS, murmur=GAPS, claps=-2,
                          choir=[CHORUS1 + (-9,), CHORUS2 + (-8,), CHORUS3 + (-6,)]),
    'B2_porsesh_pasokh': dict(reverb=MIX_MED, cheers=BASE_CHEERS, murmur=GAPS, claps=-2,
                              # صدای خواننده ۲ قوی می‌ماند و جمعیت زیرش می‌خواند (بدون کم کردن صدای خواننده)
                              choir=[CHORUS1_B + (-12,), CHORUS2_B + (-11,), CHORUS3 + (-10,)],
                              duet=[CHORUS3]),          # ترجیع‌بند آخر: دو خواننده با هم
    'B3_garm_shodan': dict(reverb=MIX_MED, cheers=BASE_CHEERS, murmur=GAPS, claps=-2,
                           choir=[PRE1 + (-13,), PRE2 + (-11,), BRIDGE + (-9,),
                                  CHORUS1 + (-10,), CHORUS2 + (-8,), CHORUS3 + (-6,)],
                           chant=[(beat(164 + 4), -4), (beat(164 + 12), -3)]),
}

# ---------- نسخه ۳ (نهایی): بر اساس B2 و بازخورد دانشجو ----------
CHEERS_V3 = [
    (INTRO_V1, 0.0, 9.0, 0.0, -6, 0.15, 1.5),       # اینترو دقیقاً مثل نسخه اول
    (POOL_SAFE, 13.0, 3.5, 11.0, -8, 0.3, 2.5),      # ورود گروه (از clip6)
    (POOL_SAFE, 7.0, 4.0, 29.0, -12, 0.4, 2.5),      # ورود خواننده ۱
    (SCREAM, 0.0, 2.0, 67.6, -6, 0.05, 0.8),         # تعویض ۱ -> ۲
    (POOL_SAFE, 0.0, 6.5, 67.8, -7, 0.4, 4.0),
    (POOL_SAFE, 6.5, 6.0, 108.6, -8, 0.4, 3.5),      # پایان کُرس اول
    (POOL_SAFE, 1.0, 5.5, 125.2, -11, 0.4, 3.0),     # تعویض ۲ -> ۱
    (SCREAM, 0.0, 2.0, 163.6, -6, 0.05, 0.8),        # تعویض ۱ -> ۲
    (POOL_SAFE, 6.2, 6.0, 163.8, -7, 0.4, 4.0),
    (POOL_SAFE, 0.5, 6.0, 202.6, -11, 0.4, 3.0),     # تعویض ۲ -> ۱
    (SCREAM, 0.0, 2.0, 220.3, -5, 0.05, 0.8),        # کُرس آخر
    (POOL_SAFE, 7.0, 8.0, 220.5, -6, 0.4, 4.0),
    (POOL_SAFE, 0.0, 16.0, END, -6, 1.0, 6.0),       # تشویق پایانی (دو لایه با شروع متفاوت)
    (POOL_SAFE, 6.0, 10.0, END + 5.0, -9, 1.0, 5.0),
]
S1 = [(29.5, 66.0), (126.0, 162.5), (203.0, 220.6)]      # بخش‌های خواننده ۱
S2 = [CHORUS1, CHORUS2, CHORUS3]                           # بخش‌های خواننده ۲
VARIANTS['C_final_v3'] = dict(
    reverb=MIX_MED, cheers=CHEERS_V3, claps=-2, choir_src='clear',
    murmur=[BREAK + (-20,), (END, N / SR, -14)],           # بدون همهمه در اینترو
    choir=[a + (-14,) for a in S1] + [CHORUS1 + (-6,), CHORUS2 + (-5,), CHORUS3 + (-4,)],
    lead_boost=[c + (1.5,) for c in S2],
    duet=[CHORUS3])

CLEAR_CHOIR, MURMUR_SAFE, SOFT_CHOIR = [None], [None], [None]

def load_clear_choir():
    cache = os.path.join(WORK, 'clear_choir_21.npy')
    if os.path.exists(cache): CLEAR_CHOIR[0] = np.load(cache)
    else:
        print('building clear choir...'); CLEAR_CHOIR[0] = build_clear_choir(); np.save(cache, CLEAR_CHOIR[0])
    MURMUR_SAFE[0] = murmur_bed(POOL_SAFE)


# ---------- نسخه ۴: بر اساس نکات دانشجو روی نسخه ۳ ----------
# شعار واقعی «بزن باران، بزن باران بزن» از نسخه لایو: از ضرب ۴۴.۱۵ تا ۴۸.۸۶ ثانیه = ۸ ضرب با تمپوی ۱۰۳.۴
LIVE_BEAT = 60 / 103.36
CHANT_BAR2 = lclip(44.10, 44.10 + 8 * LIVE_BEAT)

def chant_loop(t_start, reps, gains, seed=5):
    """شعار را با تمپوی آهنگ ما (۱۰۰) هماهنگ می‌کند و هر ۸ ضرب تکرار می‌کند.
    هر تکرار با چند لایه‌ی کمی متفاوت ساخته می‌شود تا عین هم شنیده نشود."""
    r = np.random.default_rng(seed)
    base = time_stretch(CHANT_BAR2, SR, LIVE_BEAT / BEAT, 0.0, high_quality=True, preserve_formants=True)
    out = np.zeros((2, N), np.float32)
    for k in range(reps):
        rep = base.copy()
        for _ in range(2):   # دو لایه‌ی اضافه = جمعیت بزرگ‌تر
            y = Pedalboard([PitchShift(semitones=float(r.uniform(-0.2, 0.2))),
                            LowpassFilter(r.uniform(3000, 5000))])(base, SR)
            d = int(r.uniform(0.015, 0.04) * SR)
            y = np.roll(y, d, axis=1); y[:, :d] = 0
            y = y[::-1] if r.random() < 0.5 else y   # جابه‌جایی چپ و راست
            rep = rep + y * r.uniform(0.5, 0.8)
        rep = fade(rep / rms(rep) * 0.05, 0.03, 0.4)
        place(out, rep, t_start + k * 8 * BEAT, gains[k])
    return out


CH1A, CH1A_E, CH1B, CH1B_E = (68.3, 80.5), (80.75, 86.6), (87.85, 102.6), (102.7, 108.55)
CH2A, CH2A_E, CH2B, CH2B_E = (164.3, 176.4), (176.75, 182.55), (183.85, 195.8), (195.95, 202.0)
BAND_IN = beat(3)                       # «بعد از ضرب سوم» ورود گروه
CHEERS_V4 = [
    (INTRO_V1, 0.0, 9.0, 0.0, -6, 0.15, 4.0),        # اینترو؛ این بار آرام تا شمارش درامر کم می‌شود
    (POOL_SAFE, 13.0, 3.5, 11.0, -8, 0.3, 2.5),      # ورود گروه
    (POOL_SAFE, 0.0, 7.0, 27.4, -8, 1.8, 4.0),       # ورود خواننده ۱: بالا می‌رود و آرام پایین می‌آید
    (SCREAM, 0.0, 2.0, 67.6, -6, 0.05, 0.8),         # ورود خواننده ۲
    (POOL_SAFE, 0.0, 6.5, 67.8, -7, 0.4, 4.0),
    (POOL_SAFE, 6.5, 6.0, 108.6, -8, 0.4, 3.5),      # پایان کُرس اول
    (SCREAM, 0.0, 2.0, 121.6, -6, 0.5, 1.0),         # آخر بخش درامز: جیغ و بعد فید
    (POOL_SAFE, 6.5, 6.5, 121.0, -6, 1.2, 4.0),
    (SCREAM, 0.0, 2.0, 163.6, -6, 0.05, 0.8),        # ورود دوباره خواننده ۲
    (POOL_SAFE, 6.2, 6.0, 163.8, -7, 0.4, 4.0),
    (POOL_SAFE, 0.5, 6.0, 202.6, -11, 0.4, 3.0),     # بریج
    (SCREAM, 0.0, 2.0, 220.3, -5, 0.05, 0.8),        # کُرس آخر
    (POOL_SAFE, 7.0, 8.0, 220.5, -6, 0.4, 4.0),
    (POOL_SAFE, 0.0, 16.0, 240.5, -6, 1.5, 7.0),     # تشویق بعد از دقیقه ۴، با فید
    (POOL_SAFE, 6.0, 10.0, 244.0, -9, 1.5, 6.0),
]
VARIANTS['D_final_v4'] = dict(
    reverb=MIX_MED, cheers=CHEERS_V4, claps=-2, choir_src='clear',
    murmur=[BREAK + (-20,), (242.0, N / SR, -14)],
    # «بزن باران، بزن باران بزن» واقعی، با ریتم آهنگ، سه بار تا ورود خواننده ۱
    chant_loop=[(BAND_IN, 3, [-4, -2.5, -1])],
    # بخش درامز (۱:۵۰): آخرش تا ‎-10dB محو می‌شود و گروه روی ۲:۰۵.۴ برمی‌گردد
    ins_dip=[(118.5, 125.2, -10, 125.4)],
    choir=[(29.6, 66.0, -10),                          # زیر خواننده ۱: کم ولی شنیدنی
           CH1A + (-4,), CH1A_E + (-1,), CH1B + (-3,), CH1B_E + (-1,),   # خواننده ۲: بلند، با اوج روی «به داد من برس»
           (125.6, 144.8, -4),                         # ۲:۰۶ خواننده ۱ با جمعیت، بلند
           (144.9, 161.0, 1),                          # ۲:۲۵ تا ۲:۴۱ فقط جمعیت
           CH2A + (-3,), CH2A_E + (0,), CH2B + (-2,), CH2B_E + (0,),     # کُرس دوم: همراهی بیشتر
           (203.0, 220.4, -10),                        # بریج
           (220.95, 242.1, -6)],                       # آخر: هر دو خواننده + جمعیت، خواننده‌ها غالب
    lead_duck=[(144.9, 161.0, -20)],                  # صدای خواننده ۱ تقریباً حذف
    lead_boost=[(68.3, 108.55, 2.0), (164.3, 202.0, 2.5), (220.95, 242.1, 1.5)],
    duet=[CHORUS3])

# ---------- نسخه ۵: همخوانی ملایم با غلبه صدای زنان، بدون شعار اول ----------
_v5 = dict(VARIANTS['D_final_v4'])
_v5.update(
    choir_src='soft',
    chant_loop=[],                                   # شعار اول آهنگ فعلاً حذف شد
    choir=[(29.6, 66.0, -14),                        # زیر خواننده ۱: خیلی ملایم
           CH1A + (-10,), CH1A_E + (-8,), CH1B + (-10,), CH1B_E + (-8,),
           (125.6, 144.8, -11),
           (144.9, 161.0, -1),                       # فقط مردم
           CH2A + (-9,), CH2A_E + (-7,), CH2B + (-9,), CH2B_E + (-7,),
           (203.0, 220.4, -14),
           (220.95, 242.1, -11)],
    lead_duck=[(144.9, 161.0, -18)])
VARIANTS['E_final_v5'] = _v5

# ---------- نسخه ۶: همان همخوانی نسخه ۵، در کناره‌های استریو و کمی بلندتر ----------
_v6 = dict(VARIANTS['E_final_v5'])
_v6.update(
    choir_src='soft_sides',
    # گین‌ها بعد از حذف بخش وسط اندازه‌گیری شدند: همخوانی حدود ۴ تا ۶ دسی‌بل زیر خواننده
    # و در کناره‌ها حدود ۳ دسی‌بل بالاتر از سازها
    choir=[(29.6, 66.0, -5),
           CH1A + (0,), CH1A_E + (2,), CH1B + (0,), CH1B_E + (2,),
           (125.6, 144.8, -1),
           (144.9, 161.0, 4),                        # فقط مردم
           CH2A + (1,), CH2A_E + (3,), CH2B + (1,), CH2B_E + (3,),
           (203.0, 220.4, -5),
           (220.95, 242.1, -1)])
VARIANTS['F_final_v6'] = _v6

if __name__ == '__main__':
    only = sys.argv[3:] or list(VARIANTS)
    print('building choir...'); choir = build_crowd_choir()
    murmur = murmur_bed()
    if any(VARIANTS[k].get('choir_src') == 'clear' for k in only):
        load_clear_choir()
    if any(VARIANTS[k].get('choir_src') in ('soft', 'soft_sides') for k in only):
        cache = os.path.join(WORK, 'soft_choir_31.npy')
        if os.path.exists(cache): SOFT_CHOIR[0] = np.load(cache)
        else:
            print('building soft choir...'); SOFT_CHOIR[0] = build_soft_choir(); np.save(cache, SOFT_CHOIR[0])
    MURMUR_SAFE[0] = murmur_bed(POOL_SAFE)
    for k in only:
        print('mixing', k); print(' ->', mix(VARIANTS[k], choir, murmur, k))
