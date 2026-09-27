"""
ساخت نسخه‌های لایو از «Ehaam - Bezan Baran»

پیش‌نیاز: جداسازی وکال/موزیک با audio-separator (مدل UVR-MDX-NET-Voc_FT)
روی نسخه استودیویی (song.wav) و نسخه لایو واقعی (reallive.wav)؛ خروجی‌ها در WORK/sep.

اجرا:  python3 make_live.py <WORK_DIR> <OUT_DIR>
"""
import sys, os
import numpy as np
import soundfile as sf
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
CHEER_A = lclip(4, 31)      # clip1: تشویق و جیغ
SCREAM = lclip(78, 80)      # clip3: فقط جیغ
CHEER_B = lclip(110, 133)   # clip4: تشویق و جیغ
CHEER_C = lclip(250, 255)   # clip6
CHEER_D = lclip(266, 270)   # clip7
CHANT = lclip(44.2, 48.7)   # clip2: «بزن باران بزن» (همخوانی واقعی)

def cheer(src, start, dur, fo=1.5):
    return fade(src[:, int(start * SR):int((start + dur) * SR)], 0.15, fo)


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
        delay = r.uniform(0.02, 0.11)                # دیرتر از خواننده
        chain = Pedalboard([
            PitchShift(semitones=float(semis)),
            HighpassFilter(r.uniform(150, 260)),
            LowpassFilter(r.uniform(2200, 4500)),     # دورتر = تیره‌تر
            Chorus(rate_hz=r.uniform(0.2, 0.6), depth=0.3, mix=0.5),
        ])
        y = chain(mono, SR)
        y = np.roll(y, int(delay * SR)); y[:int(delay * SR)] = 0
        # بلندی نامنظم هر نفر
        lfo = 1 + 0.35 * np.sin(2 * np.pi * r.uniform(0.05, 0.25) * np.arange(N) / SR + r.uniform(0, 6))
        out += pan(y * lfo, r.uniform(-1, 1)) * r.uniform(0.5, 1.0)
    out = Pedalboard([Reverb(room_size=0.95, damping=0.4, wet_level=0.7, dry_level=0.35, width=1.0),
                      Compressor(threshold_db=-24, ratio=3)])(out, SR)
    return out / rms(out) * rms(voc)                 # هم‌سطح با وکال اصلی (قبل از گین)


def murmur_bed():
    src = np.concatenate([CHEER_A, CHEER_B], axis=1)
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
    # وکال: کمپرسور، حضور، اکوی کوتاه صحنه
    V = Pedalboard([Compressor(threshold_db=-20, ratio=3, attack_ms=5, release_ms=120),
                    PeakFilter(3000, 1.5, 1.0), HighpassFilter(90)])(voc, SR)
    V = V + Pedalboard([Delay(delay_seconds=0.095, feedback=0.1, mix=1.0), Gain(-18)])(V, SR)
    for t0, t1, d in cfg.get('lead_duck', []):   # لحظه‌هایی که فقط جمعیت می‌خواند
        g = 1 - (1 - db(d)) * env(N, [(t0, t1, 0, 0.25, 0.4)])
        V = V * g
    hall = Pedalboard([Reverb(room_size=rv['size'], damping=0.5, wet_level=1.0, dry_level=0.0, width=1.0)])
    wet = hall(I * rv['ins'] + V * rv['voc'], SR)
    music = I + V + wet

    fx = np.zeros((2, N), np.float32)
    # همخوانی
    fx += choir * env(N, [(a, b, d, 1.0, 1.2) for a, b, d in cfg['choir']])
    # همهمه
    if cfg.get('murmur'):
        fx += murmur * env(N, [(a, b, d + CHEER_REF, 2.0, 2.0) for a, b, d in cfg['murmur']])
    # جیغ و تشویق
    for src, st, dur, t, g in cfg['cheers']:
        place(fx, cheer(src, st, dur), t, g + CHEER_REF + cfg.get('cheer_gain', 0))
    for t, g in cfg.get('chant', []):
        place(fx, fade(CHANT, 0.05, 0.3), t, g + CHEER_REF)
    if cfg.get('claps'):
        fx += claps(*BREAK) * db(cfg['claps'] + CLAP_REF)
    fx += count_in()
    fx = Pedalboard([Reverb(room_size=rv['size'], wet_level=0.2, dry_level=1.0)])(fx, SR)

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
BASE_CHEERS = [
    (CHEER_A, 0.0, 9.0, 0.0, -6),        # شروع: تشویق قبل از اجرا
    (CHEER_B, 2.0, 4.0, 11.0, -8),       # ورود گروه
    (CHEER_C, 0.0, 4.0, 29.3, -12),      # ورود خواننده ۱
    (SCREAM, 0.0, 2.0, 67.8, -6),        # ورود خواننده ۲
    (CHEER_B, 8.0, 4.0, 68.0, -12),
    (CHEER_D, 0.0, 4.0, 108.6, -8),      # پایان کُرس اول
    (SCREAM, 0.0, 2.0, 163.8, -7),       # کُرس دوم
    (CHEER_C, 0.0, 4.0, 164.0, -12),
    (SCREAM, 0.0, 2.0, 220.5, -5),       # کُرس آخر
    (CHEER_B, 12.0, 5.0, 220.8, -10),
    (CHEER_A, 9.0, 18.0, END, -6),       # تشویق پایانی
    (CHEER_B, 0.0, 10.0, END + 1.5, -9),
]
MIX_LIGHT = dict(size=0.55, ins=0.10, voc=0.14)
MIX_MED = dict(size=0.75, ins=0.14, voc=0.20)
MIX_BIG = dict(size=0.92, ins=0.20, voc=0.28)
GAPS = [(0, 11.5, -16), BREAK + (-20,), (END, N / SR, -14)]

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
                              choir=[CHORUS1_B + (-8,), CHORUS2_B + (-7,),
                                     (CHORUS3[0], CHORUS3[0] + 4 * BAR, -3), (CHORUS3[0] + 4 * BAR, CHORUS3[1], -9)],
                              lead_duck=[(CHORUS3[0], CHORUS3[0] + 4 * BAR, -22)]),
    'B3_garm_shodan': dict(reverb=MIX_MED, cheers=BASE_CHEERS, murmur=GAPS, claps=-2,
                           choir=[PRE1 + (-13,), PRE2 + (-11,), BRIDGE + (-9,),
                                  CHORUS1 + (-10,), CHORUS2 + (-8,), CHORUS3 + (-6,)],
                           chant=[(beat(164 + 4), -4), (beat(164 + 12), -3)]),
}

if __name__ == '__main__':
    only = sys.argv[3:] or list(VARIANTS)
    print('building choir...'); choir = build_crowd_choir()
    murmur = murmur_bed()
    for k in only:
        print('mixing', k); print(' ->', mix(VARIANTS[k], choir, murmur, k))
