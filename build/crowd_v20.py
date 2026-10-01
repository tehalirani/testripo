"""
نسخه ۲۰: صدای زنان بیشتر زیر خواننده ۲، صداهای تنظیم‌شده‌ی دانشجو محوتر و دقیقاً هم‌زمان، ۲:۰۸ تا ۲:۲۰ آرام‌تر

دانشجو: «صدای زن‌ها رو بیشتر کن، مخصوصاً زیر خواننده دوم. اونایی که با هوش مصنوعی درست کردیم و انداختیم زیر دومی
خیلی بد هست؛ یکم محوش کن و زمانش رو هم دقیقاً بنداز زیر خواننده؛ یکم نرم و ملایم‌تر بشه. از ۲:۰۸ تا ۲:۲۰
زیرصدا یکم بد هست؛ یکم کمترش کن ولی باشه که نشون بده دارن همخوانی می‌کنن.»

  - زنان: سه صدای زنانه‌ی Kits.ai (voice1، voice3، voice4) دقیقاً هم‌زمان با خواننده (بدون تأخیر اضافه).
    این فایل‌ها فقط کُرس اول را دارند؛ برای کُرس دوم و آخر، وکال خواننده ۲ در کُرس اول با DTW (کروما + MFCC)
    به کُرس دوم و آخر هم‌تراز شد و صداهای زنانه با همان نگاشت به آنجا برده شدند (همان ملودی و شعر).
    محو و نرم: low-pass ۵ کیلوهرتز، کاهش ۳ کیلوهرتز، ریورب سالن، کناره‌های استریو.
  - صداهای خود دانشجو (کوک‌شده): نسخه‌ی تأخیردار (humanize) حذف شد، دقیقاً هم‌زمان؛ low-pass ۴.۵ کیلوهرتز و ریورب بیشتر؛
    پیش‌کُرس ‎-4 و زیر خواننده ۲ ‎-6 دسی‌بل.
  (دانشجو گفت منظورش از «صداهای هوش مصنوعی» صداهای Kits.ai است؛ این صداها در نسخه ۱۹ نبودند، پس احتمالاً
   صداهای کوک‌شده‌ی خودش را شنیده بود. هر دو محو، نرم و دقیقاً هم‌زمان شدند.)
  - ۲:۰۸ تا ۲:۲۰: کل جمعیت ۶ دسی‌بل آرام‌تر.

اجرا:  python3 crowd_v20.py <WORK_DIR> <OUT_DIR> <REC_DIR>
"""
import sys, os, json
import numpy as np
import soundfile as sf
import librosa

WORK, OUT, REC = sys.argv[1], sys.argv[2], sys.argv[3]
import crowd_v19 as V19
V18, V17, V16, V14, V12, V10, M, V8 = V19.V18, V19.V17, V19.V16, V19.V14, V19.V12, V19.V10, V19.M, V19.V8
V11 = V12.V11
from make_live import (SR, N, db, rms, env, pan, time_stretch, to_sides, Pedalboard, HighpassFilter,
                       LowpassFilter, PeakFilter, Reverb, Compressor)

CH1, CH2, CH3 = (68.3, 108.55), (164.3, 202.0), (220.95, 242.1)
WOMEN = ['kits_voice1_hamoktav.aac', 'kits_voice3_hamoktav.aac', 'kits_voice4_hamoktav_roshan.aac']
WOMEN_DB = {'ch1': -4.0, 'ch2': -3.0, 'ch3': -3.0}
SELF_PRE_DB, SELF_CH_DB = -4.0, -6.0
DIP = (128.0, 140.0, -6.0)          # ۲:۰۸ تا ۲:۲۰


def map_sections(src, dst, hop=0.05):
    """DTW بین وکال خواننده ۲ در دو بخش؛ برمی‌گرداند: زمان مقصد -> زمان مبدأ."""
    sr = 22050; h = int(hop * sr)
    def F(a, b):
        y = librosa.resample(M.voc[:, int(a * SR):int(b * SR)].mean(0), orig_sr=SR, target_sr=sr)
        c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=h)
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=h)[1:]
        m = (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-6)
        n = min(c.shape[1], m.shape[1]); return np.vstack([c[:, :n] * 2, m[:, :n] * 0.3])
    A, B = F(*src), F(*dst)
    _, wp = librosa.sequence.dtw(X=A, Y=B, metric='cosine'); wp = wp[::-1]
    st = src[0] + wp[:, 0] * hop; dt = dst[0] + wp[:, 1] * hop
    grid = np.arange(dst[0], dst[1], hop)
    m = np.interp(grid, dt, st)
    m = np.convolve(np.pad(m, 6, mode='edge'), np.ones(13) / 13, 'valid')
    return grid, np.maximum.accumulate(m)


def warp_track(x, grid, smap, blk=0.5, ov=0.12):
    out = np.zeros(N, np.float32); L = int((blk + ov) * SR)
    win = np.ones(L, np.float32); r = int(ov * SR); win[:r] = np.linspace(0, 1, r); win[-r:] = np.linspace(1, 0, r)
    t, t1 = grid[0], grid[-1]
    while t < t1:
        a = np.interp(t, grid, smap); b = np.interp(min(t + blk + ov, t1), grid, smap)
        src = x[int(a * SR):int(b * SR)]
        if len(src) > 2000:
            y = time_stretch(src[None], SR, float((b - a) / (blk + ov)), 0.0)[0][:L]
            y = np.pad(y, (0, L - len(y))); i = int(t * SR); out[i:i + L] += y * win
        t += blk
    return out


def women_layer(seed=20):
    r = np.random.default_rng(seed)
    ref = M.voc[:, int(V10.SRC_T0 * SR):int((V10.SRC_T0 + 41) * SR)].mean(0)
    tracks = []
    for f in WOMEN:
        y = V10.load_voice(os.path.join(REC, f)); lag = V10.align(y, ref)
        y = y[int(max(lag, 0) * SR):]
        full = np.zeros(N, np.float32); a = int(V10.SRC_T0 * SR); full[a:a + len(y)] = y[:N - a]
        full *= env(N, [(CH1[0] - 0.3, CH1[1] + 0.5, 0, 0.3, 0.6)])
        tracks.append(full)
    g2, m2 = map_sections(CH1, CH2)
    g3, m3 = map_sections((CH1[0], CH1[0] + (CH3[1] - CH3[0]) + 1.0), CH3)
    out = np.zeros((2, N), np.float32); parts = {}
    for k, (name, rng) in enumerate([('ch1', CH1), ('ch2', CH2), ('ch3', CH3)]):
        layer = np.zeros((2, N), np.float32)
        for i, tr in enumerate(tracks):
            z = tr if name == 'ch1' else warp_track(tr, *(g2, m2) if name == 'ch2' else (g3, m3))
            layer += pan(z, [-0.7, 0.75, -0.2][i] + r.uniform(-0.1, 0.1)) * r.uniform(0.8, 1.0)
        layer = Pedalboard([HighpassFilter(180), LowpassFilter(5000), PeakFilter(3000, -3.0, 0.9),
                            Compressor(threshold_db=-30, ratio=2.5, attack_ms=20, release_ms=200),
                            Reverb(room_size=0.82, damping=0.6, wet_level=0.45, dry_level=0.6, width=1.0)])(layer, SR)
        layer = to_sides(layer, mid_db=-4.0) * env(N, [(rng[0] - 0.3, rng[1] + 1.5, 0, 0.6, 1.5)])
        sl = slice(int(rng[0] * SR), int(rng[1] * SR))
        boost = {'ch1': 2.0, 'ch2': 2.5, 'ch3': 1.5}[name]
        layer *= db(WOMEN_DB[name]) * rms(M.voc[:, sl] * db(boost)) / rms(layer[:, sl])
        out += layer; parts[name] = round(20 * np.log10(rms(layer[:, sl])) - 20 * np.log10(rms(M.voc[:, sl] * db(boost))), 1)
    return out, parts


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    M.load_clear_choir()
    murmur = M.murmur_bed()
    # جمعیت لایو و جواب مردم مثل نسخه ۱۹
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
    live *= db(V19.LIVE_REF_DB + lead_db - 20 * np.log10(rms(live[:, rs])))
    chant = V18.chant_soft()
    # صداهای دانشجو: فقط نسخه‌ی هم‌زمان، محوتر و آرام‌تر زیر خواننده ۲
    V11.humanize = lambda full, segs, r: full.copy()
    V17.REL_DB = 0.0
    me, info = V17.build_self_crowd()
    me = Pedalboard([LowpassFilter(4500),
                     Reverb(room_size=0.82, damping=0.6, wet_level=0.35, dry_level=0.7, width=1.0)])(me, SR)
    me = V18.soften(me)
    me *= env(N, [(48.6, 67.0, SELF_PRE_DB, 0.8, 1.0), (67.0, 110.0, SELF_CH_DB, 1.0, 2.0)], base=-120)
    women, wparts = women_layer()
    women = V18.soften(women)
    crowd = chant + live + me + women
    crowd *= 1 - (1 - db(DIP[2])) * env(N, [(DIP[0], DIP[1], 0, 1.5, 1.5)])
    tb = np.array([0, 68.2, 68.3, 108.55, 108.6, 164.2, 164.3, 202.0, 202.1, 220.9, 220.95, 242.1, 242.2, N / SR])
    gb = db(np.array([0, 0, 2, 2, 0, 0, 2.5, 2.5, 0, 0, 1.5, 1.5, 0, 0]))
    crowd, worst = V19.cap_under_lead(crowd, (tb, gb))
    intro = V8.smooth_intro() * db(-4 + M.CHEER_REF)
    wav = M.mix(V8.cfg_v8(intro + crowd), None, murmur, 'T_v20_full')
    x, _ = sf.read(wav, dtype='float32', always_2d=True); x = x.T
    orig = sf.info(os.path.join(WORK, 'song.wav')).frames
    y = x[:, :orig].copy(); f = int(5 * SR); y[:, -f:] *= np.linspace(1, 0, f) ** 2
    sf.write(os.path.join(OUT, 'T_v20.wav'), y.T, SR, subtype='PCM_16')
    V10.write(os.path.join(OUT, 'preview_v20_chorus1'), y[:, int(64 * SR):int(112 * SR)])
    V10.write(os.path.join(OUT, 'preview_v20_chorus2'), y[:, int(160 * SR):int(205 * SR)])
    V10.write(os.path.join(OUT, 'preview_v20_women_only'), women[:, int(64 * SR):int(112 * SR)])
    rep = dict(women_vs_lead=wparts, cap_max_reduction=round(worst, 1))
    json.dump(rep, open(os.path.join(OUT, 'v20_info.json'), 'w'), indent=1)
    print(json.dumps(rep))
