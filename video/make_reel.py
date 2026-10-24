"""北新地 大嵓埜 Instagram リール (1080x1920 / 30fps / 30秒) を生成する。

使い方: python3 video/make_reel.py
必要: Pillow, numpy, ffmpeg。フォント (Shippori Mincho / Cormorant Garamond, OFL) は
video/fonts に無ければ Google Fonts のリポジトリから取得する。
"""
import glob
import math
import os
import subprocess
import unicodedata
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images")
FONT_DIR = os.path.join(ROOT, "video", "fonts")
OUT = os.path.join(ROOT, "video", "okurano_reel_30s.mp4")

W, H, FPS = 1080, 1920, 30
TOTAL = 30.0
XFADE = 0.7

CREAM = (246, 238, 222)
GOLD = (201, 169, 110)

FONTS = {
    "mincho": ("ShipporiMincho-Medium.ttf",
               "https://raw.githubusercontent.com/google/fonts/main/ofl/shipporimincho/ShipporiMincho-Medium.ttf"),
    "mincho_r": ("ShipporiMincho-Regular.ttf",
                 "https://raw.githubusercontent.com/google/fonts/main/ofl/shipporimincho/ShipporiMincho-Regular.ttf"),
    "serif_en": ("CormorantGaramond.ttf",
                 "https://raw.githubusercontent.com/google/fonts/main/ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf"),
}


def font(name, size):
    fname, url = FONTS[name]
    path = os.path.join(FONT_DIR, fname)
    if not os.path.exists(path):
        os.makedirs(FONT_DIR, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return ImageFont.truetype(path, size)


def find_image(key):
    for p in glob.glob(os.path.join(IMG_DIR, "*.jpg")):
        if unicodedata.normalize("NFC", os.path.basename(p)) == key:
            return p
    raise FileNotFoundError(key)


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(x)
    return 3 * x * x - 2 * x * x * x


# ---------------------------------------------------------------- slides
class Slide:
    """写真は切らずに全体を表示 (横幅に合わせる)。背景はぼかした同じ写真。"""

    def __init__(self, key, dur, transition, texts, dim=0.0, shade=None):
        self.key, self.dur, self.transition, self.texts, self.dim = key, dur, transition, texts, dim
        self.shade = shade
        im = ImageOps.exif_transpose(Image.open(find_image(key))).convert("RGB")
        # 背景: カバー + 強いぼかし + 暗く
        s = max(W / im.width, H / im.height) * 1.1
        bg = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        bg = bg.crop(((bg.width - W) // 2, (bg.height - H) // 2, (bg.width - W) // 2 + W, (bg.height - H) // 2 + H))
        bg = bg.filter(ImageFilter.GaussianBlur(48))
        self.bg = Image.eval(bg, lambda v: int(v * 0.32))
        # 前景: 横幅いっぱい (縦長でも横長でも写真は切れない)
        s = W / im.width
        if im.height * s > H:
            s = H / im.height
        self.fg = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        if shade:
            # テロップの下だけ写真を少し暗くして読みやすく (y0 -> y1 にかけて濃くなる)
            y0, y1, k = shade
            top = (H - self.fg.height) // 2
            g = np.ones(self.fg.height, dtype=np.float32)
            yy = np.arange(self.fg.height) + top
            g -= k * np.clip((yy - y0) / (y1 - y0), 0, 1) ** 1.3
            self.fg = Image.fromarray((np.asarray(self.fg, dtype=np.float32) * g[:, None, None]).astype(np.uint8))
        # 写真の上下の縁を背景になじませる
        fe = 70
        a = np.ones(self.fg.height, dtype=np.float32)
        ramp = np.linspace(0, 1, fe, dtype=np.float32) ** 1.5
        a[:fe] = ramp
        a[-fe:] = ramp[::-1]
        self.mask = Image.fromarray((np.broadcast_to(a[:, None], (self.fg.height, self.fg.width)) * 255).astype(np.uint8))

    def render(self, t, extra_scale=1.0):
        """t: スライド内の経過秒。前景は 0.965 -> 1.0 にゆっくり寄る (枠からはみ出さない)。"""
        p = clamp(t / self.dur)
        scale = (0.965 + 0.035 * ease_in_out(p)) * extra_scale
        frame = self.bg.copy()
        fw, fh = round(self.fg.width * scale), round(self.fg.height * scale)
        fg = self.fg.resize((fw, fh), Image.BILINEAR)
        frame.paste(fg, ((W - fw) // 2, (H - fh) // 2), self.mask.resize((fw, fh), Image.BILINEAR))
        if self.dim:
            d = self.dim(t) if callable(self.dim) else self.dim
            if d > 0:
                frame = Image.blend(frame, Image.new("RGB", (W, H), (8, 6, 5)), d)
        return frame

    def full(self, t, extra_scale=1.0):
        """写真 + テロップ。トランジションでもテロップが写真と一緒に動く。"""
        canvas = self.render(t, extra_scale).convert("RGBA")
        for tx in self.texts:
            tx.draw(canvas, t)
        return canvas.convert("RGB")


# ---------------------------------------------------------------- text
_sprite_cache = {}


def char_sprite(ch, fnt_name, size, color, glow=True):
    key = (ch, fnt_name, size, color, glow)
    if key in _sprite_cache:
        return _sprite_cache[key]
    f = font(fnt_name, size)
    pad = size // 2
    box = size * 2 + pad * 2
    layer = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((pad, pad), ch, font=f, fill=color + (255,))
    if glow:
        a = layer.split()[3].filter(ImageFilter.GaussianBlur(size * 0.18))
        a = a.point(lambda v: int(min(255, v * 1.6)))
        shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        shadow.putalpha(a.point(lambda v: int(v * 0.75)))
        layer = Image.alpha_composite(shadow, layer)
    _sprite_cache[key] = (layer, pad)
    return _sprite_cache[key]


VERT_PUNCT = set("、。，．")


class Text:
    """縦書き (columns, 右から左) または横書き (lines) のテロップ。1文字ずつふわっと現れる。"""

    def __init__(self, lines, x, y, size, start, end, vertical=True, fnt="mincho",
                 color=CREAM, align="center", stagger=0.075, tracking=1.0, line_gap=1.55, glow=True):
        self.lines, self.x, self.y, self.size = lines, x, y, size
        self.start, self.end, self.vertical = start, end, vertical
        self.fnt, self.color, self.align = fnt, color, align
        self.stagger, self.tracking, self.line_gap, self.glow = stagger, tracking, line_gap, glow
        self.chars = self._layout()

    def _layout(self):
        f = font(self.fnt, self.size)
        out, i = [], 0
        for li, line in enumerate(self.lines):
            if self.vertical:
                cx = self.x - li * self.size * self.line_gap
                cy = self.y
                for ch in line:
                    dx = dy = 0
                    if ch in VERT_PUNCT:
                        dx, dy = self.size * 0.6, -self.size * 0.62
                    out.append((ch, cx + dx, cy + dy, i))
                    cy += self.size * self.tracking
                    i += 1
            else:
                widths = [f.getlength(ch) * self.tracking for ch in line]
                total = sum(widths)
                cx = {"center": self.x - total / 2, "left": self.x, "right": self.x - total}[self.align]
                cy = self.y + li * self.size * self.line_gap
                for ch, w in zip(line, widths):
                    out.append((ch, cx, cy, i))
                    cx += w
                    i += 1
        return out

    def draw(self, canvas, t):
        if t < self.start or t > self.end + 0.01:
            return
        fade_out = clamp((self.end - t) / 0.6)
        for ch, cx, cy, i in self.chars:
            if ch == " ":
                continue
            p = ease_out((t - self.start - i * self.stagger) / 0.7)
            a = p * fade_out
            if a <= 0.003:
                continue
            spr, pad = char_sprite(ch, self.fnt, self.size, self.color, self.glow)
            if a < 0.999:
                spr = spr.copy()
                spr.putalpha(spr.split()[3].point(lambda v, a=a: int(v * a)))
            off = (1 - p) * self.size * 0.35
            if self.vertical:
                pos = (int(cx - pad), int(cy - pad + off))
            else:
                pos = (int(cx - pad), int(cy - pad + off))
            canvas.alpha_composite(spr, dest=(max(pos[0], 0), max(pos[1], 0)),
                                   source=(max(-pos[0], 0), max(-pos[1], 0)))


class Line:
    """細い金のライン (中心から左右に伸びる)。"""

    def __init__(self, cx, cy, length, start, end, vertical=False, width=2):
        self.cx, self.cy, self.length, self.start, self.end = cx, cy, length, start, end
        self.vertical, self.width = vertical, width

    def draw(self, canvas, t):
        if t < self.start or t > self.end:
            return
        p = ease_out((t - self.start) / 0.9) * clamp((self.end - t) / 0.6)
        half = self.length / 2 * ease_out((t - self.start) / 0.9)
        d = ImageDraw.Draw(canvas)
        col = GOLD + (int(255 * clamp(p)),)
        if self.vertical:
            d.line([(self.cx, self.cy - half), (self.cx, self.cy + half)], fill=col, width=self.width)
        else:
            d.line([(self.cx - half, self.cy), (self.cx + half, self.cy)], fill=col, width=self.width)


# ---------------------------------------------------------------- transitions
def soft_mask(p, kind, feather=0.18):
    """0..1 の進行度から B (次の写真) の不透明度マスク (H, W) を返す。"""
    if kind == "wipe_lr" or kind == "wipe_rl":
        x = np.linspace(0, 1, W, dtype=np.float32)
        if kind == "wipe_rl":
            x = 1 - x
        edge = p * (1 + feather) - feather
        m = np.clip((edge + feather - x) / feather, 0, 1)
        return np.broadcast_to(m[None, :], (H, W))
    if kind == "iris":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / math.hypot(W / 2, H / 2)
        edge = p * (1 + feather)
        return np.clip((edge - r) / feather, 0, 1)
    raise ValueError(kind)


def gold_edge(frame, p, kind):
    """ふすまが開くようなワイプの境目に、細い金の光を入れる。"""
    if kind not in ("wipe_lr", "wipe_rl"):
        return frame
    feather = 0.18
    edge = p * (1 + feather) - feather / 2
    x = edge * W if kind == "wipe_lr" else (1 - edge) * W
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    a = int(200 * math.sin(math.pi * clamp(p)))
    d.line([(x, 0), (x, H)], fill=GOLD + (a,), width=3)
    glow = Image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(10)), glow)
    out = frame.convert("RGBA")
    out.alpha_composite(glow)
    return out.convert("RGB")


def transition(a_slide, ta, b_slide, tb, p, kind):
    if kind == "blur":
        A = a_slide.full(ta).filter(ImageFilter.GaussianBlur(18 * ease_in_out(p)))
        B = b_slide.full(tb).filter(ImageFilter.GaussianBlur(18 * (1 - ease_in_out(p))))
        return Image.blend(A, B, ease_in_out(p))
    if kind == "black":
        if p < 0.5:
            return Image.blend(a_slide.full(ta), Image.new("RGB", (W, H)), ease_in_out(p * 2))
        return Image.blend(Image.new("RGB", (W, H)), b_slide.full(tb), ease_in_out(p * 2 - 1))
    if kind == "zoom":
        e = ease_in_out(p)
        A = a_slide.full(ta, extra_scale=1 + 0.25 * e)
        B = b_slide.full(tb, extra_scale=1.12 - 0.12 * e)
        return Image.blend(A, B, e)
    A = np.asarray(a_slide.full(ta), dtype=np.float32)
    B = np.asarray(b_slide.full(tb), dtype=np.float32)
    m = soft_mask(ease_in_out(p), kind)[..., None]
    out = Image.fromarray((A * (1 - m) + B * m).astype(np.uint8))
    return gold_edge(out, ease_in_out(p), kind)


# ---------------------------------------------------------------- vignette
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
VIGNETTE = np.clip(1.08 - 0.32 * _r ** 2, 0.55, 1.0)[..., None]


# ---------------------------------------------------------------- storyboard
def build():
    d = 3.9
    end_d = TOTAL - 7 * d + 7 * XFADE  # 7.6s
    slides = [
        Slide("イメージ_個室_全景2.jpg", d, "blur", [
            Text(["KITASHINCHI  OKURANO"], W / 2, 92, 30, 0.2, d, vertical=False, fnt="serif_en",
                 color=GOLD, tracking=1.25, stagger=0.03, glow=False),
            Text(["お届けしたいのは、", "お料理だけではありません。"], 930, 300, 64, 0.5, d - 0.1, stagger=0.07),
        ]),
        Slide("イメージ_個室_テーブル2.jpg", d, "wipe_lr", [
            Text(["この日のために、", "しつらえる一席。"], 220, 290, 60, XFADE + 0.1, d - 0.05),
            Text(["SPACE"], 175, 820, 28, XFADE + 0.4, d - 0.05, vertical=False, fnt="serif_en",
                 color=GOLD, tracking=1.6, glow=True),
        ]),
        Slide("イメージ_接客0127.jpg", d, "black", shade=(1150, 1770, 0.7), texts=[
            Text(["OMOTENASHI"], W / 2, 1460, 30, XFADE + 0.2, d - 0.05, vertical=False, fnt="serif_en",
                 color=GOLD, tracking=1.6),
            Text(["目の前で、心を尽くす。"], W / 2, 1520, 58, XFADE + 0.35, d - 0.05, vertical=False),
        ]),
        Slide("イメージ_接客0034修.jpg", d, "iris", [
            Text(["SERVICE"], W / 2, 470, 30, XFADE + 0.2, d - 0.05, vertical=False, fnt="serif_en",
                 color=GOLD, tracking=1.6),
            Line(W / 2, 1420, 120, XFADE + 0.3, d - 0.05),
            Text(["ひと皿ごとに、真心を添えて。"], W / 2, 1470, 54, XFADE + 0.4, d - 0.05, vertical=False),
        ]),
        Slide("イメージ_酒器_全景.jpg", d, "zoom", shade=(1250, 1770, 0.5), texts=[
            Text(["一献を、ゆるりと。"], W / 2, 1530, 60, XFADE + 0.1, d - 0.05, vertical=False),
        ]),
        Slide("イメージ_酒器_アップ.jpg", d, "wipe_rl", [
            Text(["器を選ぶ、愉しみも。"], 950, 260, 58, XFADE * 0.6, d - 0.05),
        ]),
        Slide("イメージ_個室3_全景1.jpg", d, "black", [
            Text(["大切な方との、", "特別な一日に。"], 930, 290, 62, XFADE + 0.1, d - 0.05),
        ]),
        Slide("イメージ_個室2_全景.jpg", end_d, None, [
            # メッセージ
            Text(["空間、おもてなし、お料理。"], W / 2, 760, 54, XFADE + 0.2, 4.0, vertical=False, stagger=0.06),
            Text(["そのすべてで、", "素敵なひとときを。"], W / 2, 880, 62, XFADE + 1.0, 4.0, vertical=False,
                 stagger=0.07, line_gap=1.6),
            # ロゴ
            Text(["御料理"], W / 2, 620, 40, 4.3, end_d, vertical=False, color=GOLD, tracking=1.5, stagger=0.05),
            Text(["大嵓埜"], W / 2, 690, 132, 4.4, end_d, vertical=False, tracking=1.35, stagger=0.12),
            Text(["お お く ら の"], W / 2, 885, 30, 4.7, end_d, vertical=False, fnt="mincho_r", stagger=0.03),
            Line(W / 2, 950, 260, 4.8, end_d),
            Text(["北新地　懐石料理"], W / 2, 985, 40, 4.9, end_d, vertical=False, tracking=1.2, stagger=0.04),
            Text(["MICHELIN GUIDE KYOTO OSAKA 2024 / 2025 / 2026"], W / 2, 1095, 27, 5.2, end_d,
                 vertical=False, fnt="serif_en", color=GOLD, tracking=1.1, stagger=0.012),
            Text(["3年連続 セレクテッドレストラン掲載"], W / 2, 1140, 30, 5.3, end_d, vertical=False,
                 fnt="mincho_r", stagger=0.02),
            Text(["JR北新地駅より徒歩2分　ご予約はプロフィールから"], W / 2, 1270, 28, 5.6, end_d,
                 vertical=False, fnt="mincho_r", stagger=0.015),
        ], dim=lambda t: 0.25 + 0.4 * ease_in_out((t - 0.6) / 1.2)),
    ]
    return slides


def timeline(slides):
    starts, t0 = [], 0.0
    for s in slides:
        starts.append(t0)
        t0 += s.dur - XFADE
    return starts


def frame_at(slides, starts, t):
    i = max(k for k, s0 in enumerate(starts) if s0 <= t)
    s, ts = slides[i], t - starts[i]
    if i > 0 and ts < XFADE:
        # 前のスライドとの重なり = トランジション中
        prev = slides[i - 1]
        frame = transition(prev, t - starts[i - 1], s, ts, ts / XFADE, prev.transition)
    else:
        frame = s.full(ts)
    rgb = np.asarray(frame, dtype=np.float32) * VIGNETTE
    # 最初と最後はふわっと
    fade = min(ease_out(t / 0.5), ease_in_out((TOTAL - t) / 0.6))
    return (rgb * fade).astype(np.uint8)


def main():
    slides = build()
    starts = timeline(slides)
    n = int(round(TOTAL * FPS))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-movflags", "+faststart", OUT,
    ], stdin=subprocess.PIPE)
    for fi in range(n):
        t = fi / FPS
        ff.stdin.write(frame_at(slides, starts, t).tobytes())
        if fi % 90 == 0:
            print(f"{t:5.1f}s / {TOTAL:.0f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
