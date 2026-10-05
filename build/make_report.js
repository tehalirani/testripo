const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
        AlignmentType, BorderStyle, LevelFormat, HeadingLevel } = require('docx');

const FONT = 'B Nazanin';
const LATIN = 'Times New Roman'; // B Nazanin has no Latin glyphs
const SZ = 25; // 12.5pt
const r = (t, o = {}) => new TextRun({ text: t, font: { ascii: LATIN, hAnsi: LATIN, cs: FONT, eastAsia: FONT }, size: o.size || SZ, sizeComplexScript: o.size || SZ, bold: o.bold, boldComplexScript: o.bold, color: o.color, rightToLeft: true });
const p = (runs, o = {}) => new Paragraph({
  bidirectional: true, alignment: o.align || AlignmentType.START,
  spacing: { after: o.after ?? 80, line: 276 },
  children: Array.isArray(runs) ? runs : [r(runs)], ...(o.extra || {}),
});
const h = (t) => new Paragraph({
  bidirectional: true, alignment: AlignmentType.START, spacing: { before: 120, after: 50 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: '7A9CC6', space: 1 } },
  children: [r(t, { bold: true, size: 29, color: '1F3B63' })],
});
const bullet = (runs) => new Paragraph({
  bidirectional: true, alignment: AlignmentType.START, spacing: { after: 30, line: 252 },
  numbering: { reference: 'b', level: 0 }, children: Array.isArray(runs) ? runs : [r(runs)],
});

const W = 9900; // content width (A4, 1.6cm margins)
const cell = (t, w, head) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  shading: head ? { type: ShadingType.CLEAR, fill: 'DCE6F2', color: 'auto' } : undefined,
  margins: { top: 30, bottom: 30, left: 70, right: 70 },
  children: [new Paragraph({ bidirectional: true, alignment: AlignmentType.START,
    children: [r(t, { bold: head, size: 23 })] })],
});
const table = (cols, rows) => new Table({
  width: { size: W, type: WidthType.DXA }, columnWidths: cols, visuallyRightToLeft: true,
  rows: rows.map((row, i) => new TableRow({ tableHeader: i === 0, children: row.map((t, j) => cell(t, cols[j], i === 0)) })),
});

const children = [
  new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [r('گزارش نهایی پروژه: تبدیل نسخه استودیویی به نسخه لایو با کمک هوش مصنوعی', { bold: true, size: 32, color: '1F3B63' })] }),
  new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [r('آهنگ «بزن باران» از ایهام  |  ابزار اصلی: Claude Code به‌همراه مدل‌های جداسازی صدا و Kits.ai', { size: 21, color: '555555' })] }),

  h('۱. هدف و ایده'),
  p('هدف، تبدیل یک آهنگ استودیویی به اجرای زنده‌ی کنسرتی بود. برخلاف ابزارهایی مثل Suno که آهنگ را از نو می‌سازند، در این پروژه صدای اصلی خواننده‌ها و گروه حفظ شد و «فضای کنسرت» (شمارش درامر، تشویق، همخوانی مردم و صدای سالن) دور آن ساخته شد. کل کار با گفتگو و بازخورد شنیداری دانشجو هدایت شد: ۵۱ پرامپت و ۲۹ نسخه.'),

  h('۲. تحلیل اولیه'),
  bullet([r('نمونه استاد: ', { bold: true }), r('وکال اصلی حفظ و موزیک بازسازی شده بود. بیس زیر ۶۰ هرتز از ۲۵٪ به ۱۱٪ انرژی و میانه‌ها (۲۵۰ تا ۱۰۰۰ هرتز) از ۲۱٪ به ۳۵٪ رسیده بود و استریو پهن‌تر شده بود. شمارش درامر، جیغ و همخوانی مردم هم داشت.')]),
  bullet([r('آهنگ دانشجو: ', { bold: true }), r('تمپو ۱۰۰، دو خواننده (خواننده ۲ در کُرس‌ها، یک اکتاو بالاتر)، و زمان‌بندی دقیق بخش‌ها از روی رنگ صدا (MFCC) و نت.')]),
  bullet([r('نسخه لایو واقعی همین آهنگ: ', { bold: true }), r('۲ نیم‌پرده بالاتر و ۳٪ تندتر اجرا شده است. از آن صدای واقعی تشویق و همخوانی مردم برداشته شد.')]),

  h('۳. ابزارها'),
  p('Python (librosa، numpy، scipy، pedalboard، ffmpeg)؛ مدل‌های هوش مصنوعی جداسازی صدا UVR-MDX-NET-Voc_FT (وکال و موزیک) و UVR_MDXNET_KARA_2 (خواننده اصلی و صداهای پشت)؛ Kits.ai برای تبدیل صدای خواننده به صدای زنانه؛ و Claude Code برای تحلیل، ساخت، بررسی خودکار کیفیت و گزارش‌نویسی.'),

  h('۴. مسیر کار: روش‌های امتحان‌شده برای همخوانی جمعیت'),
  table([2500, 3700, 3700], [
    ['روش', 'نتیجه', 'درس'],
    ['کپی صدای خواننده با تغییر کوک و تأخیر (نسخه ۱ تا ۷)', 'حس «حمام»، اکو و صدای رباتی', 'کپیِ کمی دیرترِ صدای خواننده را گوش اکوی خود او می‌شنود'],
    ['جمعیت بی‌کلام هم‌ریتم با خواننده (نسخه ۸ و ۹)', 'تمیز، ولی همخوانی شنیده نمی‌شد', 'بدون کلمه، حس همخوانی منتقل نمی‌شود'],
    ['صداهای Kits.ai با زمان‌بندی جمله‌به‌جمله (نسخه ۱۰ و ۱۱)', 'طبیعی‌تر؛ برخی جمله‌ها رباتی', 'هر صدا یک «آدم» متفاوت است؛ کیفیت تبدیل در جمله‌ها فرق دارد'],
    ['همخوانی واقعی از نسخه لایو (نسخه ۱۲ تا ۱۶)', 'واقعی‌ترین؛ در بند اول ناهم‌زمان', 'یک خواننده‌ی پشت صحنه وسط استریو بود؛ با برداشتن فقط کناره‌ها حذف شد'],
    ['ضبط صدای خود دانشجو + تنظیم کوک خودکار (نسخه ۱۷)', 'کوک شد (اختلاف میانه ۰.۳ نیم‌پرده)', 'هم‌زمانی بدون پخش آهنگ هنگام ضبط سخت است'],
  ]),
  p('', { after: 30 }),

  h('۵. نسخه نهایی (نسخه ۲۹)'),
  bullet([r('اینترو: ', { bold: true }), r('تشویق واقعی، شمارش درامر روی ضرب، هی‌هی‌های ریتمیک نرم و اوج ناگهانی برای ورود خواننده ۱ که با شروع آواز ۹ دسی‌بل کم می‌شود.')]),
  bullet([r('بند اول و دوم: ', { bold: true }), r('جواب واقعی مردم («بزن باران بزن») در مکث خواننده؛ صدای خود دانشجو (کوک‌شده) زیر پیش‌کُرس؛ همخوانی واقعی لایو در بند دوم، هم‌ترازشده با DTW روی صدای خواننده‌ها (خطای کمتر از ۰.۱ ثانیه).')]),
  bullet([r('کُرس‌ها: ', { bold: true }), r('خواننده ۲ خشک و غالب؛ زنان Kits.ai دور و نرم، مثل تماشاچی؛ همخوانی واقعی «فقط مردمِ» لایو (۱:۵۰ تا ۱:۵۳) در همه‌ی جاهای هم‌ملودی (۱:۲۰، ۱:۳۹، ۳:۱۵، ۳:۵۳) با پایان کشیده و فید.')]),
  bullet([r('پایان: ', { bold: true }), r('در ۳:۵۳ خواننده‌ها ساکت می‌شوند و فقط مردم «هوا هوای خاطرات اوست» را می‌خوانند؛ بعد جیغ و تشویق. طول آهنگ برابر اصلی (۴:۱۳).')]),
  bullet([r('میکس کنسرتی: ', { bold: true }), r('بیس کمتر، میانه بیشتر، استریو پهن‌تر، ریورب سالن روی گروه، کمپرسور و لیمیتر (پیک ‎-1 dB)؛ سقف بلندی جمعیت ۱.۵ دسی‌بل زیر خواننده.')]),

  h('۶. کنترل کیفیت و نقش دانشجو'),
  p('هوش مصنوعی صدا را نمی‌شنود. برای همین دو مسیر کنترل استفاده شد. (۱) بررسی خودکار (build/qa.py): کلیپ، تیک، شروع یا قطع ناگهانی، صدای تیز و سازگاری تک‌کاناله. هر مورد مشکوک با مقایسه با نسخه‌ی بدون جمعیت، دستی تأیید یا رد شد. (۲) گوش دانشجو: ایرادهایی مثل «حمومی بودن»، «رباتی بودن»، «ناهم‌زمانی در ۰:۲۹» و «قطع ناگهانی در ۱:۲۳» را دانشجو پیدا کرد. برای هر کدام علت فنی پیدا و اصلاح شد (مثلاً ریورب دوباره، نرم‌کننده‌ای که شروع صدا را حذف می‌کرد، یا جوابِ مردم که با خط اول اشتباه هم‌تراز شده بود).'),

  h('۷. جمع‌بندی و محدودیت‌ها'),
  p('نتیجه‌ی اصلی: بهترین همخوانی جمعیت از صدای واقعی آدم‌ها به دست می‌آید (نسخه لایو واقعی، صدای خود دانشجو، Kits.ai). شبیه‌سازی با کپی صدای خواننده، هر قدر هم تنظیم شود، اکو و صدای رباتی می‌دهد. محدودیت‌ها: Suno به‌خاطر تکرار زیاد شعر را نپذیرفت؛ دانلود مدل‌های تبدیل صدا در محیط کار مسدود بود؛ و تبدیل Kits.ai در برخی جمله‌ها رباتی بود و حذف شد. همه‌ی کدها، ۲۹ نسخه، پرامپت‌ها و گزارش کامل در مخزن گیت‌هاب ثبت شده است (REPORT.md، پوشه‌های build و output).'),
];

const doc = new Document({
  styles: { default: { document: { run: { font: { ascii: LATIN, hAnsi: LATIN, cs: FONT, eastAsia: FONT }, size: SZ, sizeComplexScript: SZ, rightToLeft: true } } } },
  numbering: { config: [{ reference: 'b', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.START,
    style: { paragraph: { indent: { start: 300, hanging: 200 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 850, bottom: 850, left: 1000, right: 1000 } } },
    children }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(process.argv[2], b); console.log('written'); });
