"""Add a 'training only: knowledge distillation' band to the AFC architecture
figure (user's image), matching its style. Output: arch_with_kd.png"""
from PIL import Image, ImageDraw, ImageFont

SRC = "/tmp/claude-1000/-media-vcl-DATA-gyuz/c4d06c0c-732d-4c96-b90b-a31ea0aec5f7/images/12.png"
OUT = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "outputs/figures/arch_with_kd.png")
S = 3                                     # draw at 3x for crisp lines
_rgba = Image.open(SRC).convert("RGBA")
src = Image.new("RGB", _rgba.size, "white")
src.paste(_rgba, mask=_rgba.split()[3])          # transparent -> white
W0, H0 = src.size
EXT_R, EXT_B = 40, 165
img = Image.new("RGB", ((W0 + EXT_R) * S, (H0 + EXT_B) * S), "white")
img.paste(src.resize((W0 * S, H0 * S), Image.LANCZOS), (0, 0))
d = ImageDraw.Draw(img)

SERIF_B = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
SERIF_I = "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"


def F(path, pt):
    return ImageFont.truetype(path, int(pt * S))


BLUE, BLUE_E = (190, 225, 240), (120, 170, 200)
PINK, PINK_E = (245, 200, 195), (215, 140, 130)
ORANGE, ORANGE_E = (252, 228, 190), (225, 160, 70)
GREEN = (225, 238, 215)
GRAY = (120, 120, 120)
RED = (200, 40, 40)
INK = (30, 30, 30)


def P(*xy):
    return [v * S for v in xy]


def box(x0, y0, x1, y1, fill, edge, r=6):
    d.rounded_rectangle(P(x0, y0, x1, y1), radius=r * S, fill=fill, outline=edge, width=S)


def text(x, y, s, font, fill=INK, anchor="mm"):
    d.multiline_text((x * S, y * S), s, font=font, fill=fill, anchor=anchor, align="center",
                     spacing=2 * S)


def arrow(pts, color=INK, width=1.2, dash=False, head=6):
    pts = [(x * S, y * S) for x, y in pts]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if dash:
            n = int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / (6 * S))
            for k in range(0, n, 2):
                a, b = k / max(n, 1), min((k + 1) / max(n, 1), 1)
                d.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a),
                        (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)], fill=color, width=int(width * S))
        else:
            d.line([(x0, y0), (x1, y1)], fill=color, width=int(width * S))
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    import math
    ang = math.atan2(yb - ya, xb - xa)
    h = head * S
    d.polygon([(xb, yb), (xb - h * math.cos(ang - 0.4), yb - h * math.sin(ang - 0.4)),
               (xb - h * math.cos(ang + 0.4), yb - h * math.sin(ang + 0.4))], fill=color)


def trapezoid(x0, y0, x1, y1, fill, edge, narrow_right=True, inset=8):
    if narrow_right:
        poly = [(x0, y0), (x1, y0 + inset), (x1, y1 - inset), (x0, y1)]
    else:
        poly = [(x0, y0 + inset), (x1, y0), (x1, y1), (x0, y1 - inset)]
    d.polygon([(x * S, y * S) for x, y in poly], fill=fill, outline=edge)


def tokens(x, y, n, w=11, fill=BLUE, edge=BLUE_E, dots=True):
    for i in range(n):
        d.rectangle(P(x + i * (w + 3), y, x + i * (w + 3) + w, y + w), fill=fill, outline=edge,
                    width=S)
    if dots:
        text(x + n * (w + 3) + 8, y + w / 2, "…", F(SERIF, 9))


def bars(x, y_base, hs, color, w=7):
    for i, h in enumerate(hs):
        d.rectangle(P(x + i * (w + 2), y_base - h, x + i * (w + 2) + w, y_base), fill=color)
    d.line(P(x - 3, y_base, x + len(hs) * (w + 2) + 2, y_base), fill=INK, width=S)


# ---------------------------------------------------------------- KD band
Y0, Y1 = H0 + 12, H0 + EXT_B - 10
d.rounded_rectangle(P(130, Y0, W0 + EXT_R - 8, Y1), radius=10 * S, outline=(150, 150, 150),
                    width=S, fill=(250, 250, 250))
text(142, Y0 + 13, "Training only: knowledge distillation from the uncompressed model",
     F(SERIF_B, 10.5), fill=(40, 60, 100), anchor="lm")
yc = Y0 + 66

# teacher branch from the vision encoder (AFC bypassed)
arrow([(10, 358), (4, 358), (4, yc), (148, yc)], color=GRAY, dash=True)
text(70, yc - 10, "576 tokens\n(AFC bypassed)", F(SERIF_I, 8.5), fill=GRAY)
tokens(152, yc - 6, 6)
text(196, yc + 20, "Uncompressed\nvisual tokens", F(SERIF, 8.5))
arrow([(250, yc), (282, yc)])
trapezoid(284, yc - 26, 352, yc + 26, BLUE, BLUE_E)
text(316, yc, "Multimodal\nProjector", F(SERIF_B, 8.5))
arrow([(354, yc), (386, yc)])
box(388, yc - 26, 492, yc + 26, BLUE, BLUE_E)
text(440, yc - 6, "LLaVA-1.5", F(SERIF_B, 11))
text(440, yc + 13, "(teacher, same frozen weights)", F(SERIF_I, 6.5))
arrow([(494, yc), (520, yc)])
trapezoid(522, yc - 24, 590, yc + 24, BLUE, BLUE_E, narrow_right=False)
text(556, yc, "LM Head", F(SERIF_B, 8.5))
arrow([(592, yc), (618, yc)])
bars(624, yc + 14, [10, 26, 18, 8, 5], (110, 110, 110))
text(645, yc + 28, "Teacher p", F(SERIF_I, 9))
text(667, yc + 31, "T", F(SERIF_I, 6.5))

# loss box
LX0, LX1 = 715, W0 + EXT_R - 18
box(LX0, yc - 46, LX1, yc + 46, ORANGE, ORANGE_E)
text((LX0 + LX1) / 2, yc - 30, "Training objective", F(SERIF_B, 10))
text((LX0 + LX1) / 2, yc - 10, "L = L_CE(y, p_S) + λ · L_KD", F(SERIF, 9.5))
text((LX0 + LX1) / 2, yc + 8, "L_KD = T² · KL( p_T || p_S )", F(SERIF, 9.5))
text((LX0 + LX1) / 2, yc + 28, "answer tokens only · λ = 1, T = 2", F(SERIF_I, 8))
arrow([(672, yc), (LX0 - 2, yc)])

# student distribution -> loss (right margin)
text(925, 398, "Student p", F(SERIF_I, 9), fill=RED)
text(947, 401, "S", F(SERIF_I, 6.5), fill=RED)
arrow([(906, 410), (W0 + 22, 410), (W0 + 22, yc - 48)], color=RED)


# gradient note
text(142, Y1 - 14, "Gradients update only the Adaptive Fourier Compressor (≈1.2M parameters); "
     "vision encoder, projector, LLaVA-1.5 and LM head stay frozen.",
     F(SERIF_I, 8.5), fill=RED, anchor="lm")

img = img.resize((img.width // S * 2, img.height // S * 2), Image.LANCZOS)
img.save(OUT)
print(OUT, img.size)
