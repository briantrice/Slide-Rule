#!/usr/bin/env python3

"""
Most Recent Update (11/5/20)
Please check the readme for details!
--------------------------------------------------------

Slide Rule Scale Generator 2.0 by Javier Lopez 2020
Available Scales: A B C D K R1 R2 CI DI CF DF CIF L S T ST

Table of Contents
   1. Setup
   2. Fundamental Functions
   3. Scale Generating Function
   4. Line Drawing Functions
   5. Action!
   6. Stickers
   7. Extras
"""

import math
import time

from PIL import Image, ImageFont, ImageDraw

REG = 0  # font_style regular
ITALIC = 1  # font_style italic

DIR_UP = 'up'
DIR_DOWN = 'down'

# ----------------------1. Setup----------------------------

WHITE = 'white'
BLACK = 'black'
RGB_BLACK = (0, 0, 0)
RGB_BLUE = (0, 0, 255)  # (0,0,255 = blue)
RED = 'red'
BG = WHITE
"""background color white"""
FG = BLACK
"""foreground color black"""
CUT_COLOR = RGB_BLUE  # color which indicates CUT

oX = 100  # x margins
oY = 100  # y margins
total_width = 8000 + 2 * oX
sliderule_height = 1600 * 2 + 3 * oY

sliderule_img = Image.new('RGB', (total_width, sliderule_height), BG)
renderer = ImageDraw.Draw(sliderule_img)

SH = 160
"""scale height"""

UPPER = 'upper'  # Lower alignment
LOWER = 'lower'  # Upper Alignment
SL = 5600  # scale length
li = round(total_width / 2 - SL / 2)  # left index offset from left edge

# Ticks, Labels, are referenced from li as to be consistent
STH = 70  # standard tick height
STT = 4  # standard tick thickness

# tick height size factors
XS = 0.5
SM = 0.85
MED = 1
MXL = 1.15
XL = 1.3

FRONT_SIDE = 'front'
REAR_SIDE = 'rear'
# y_off = 100  # No longer global


# ----------------------2. Fundamental Functions----------------------------


def draw_tick(y_off, x, height, thickness, al):
    """
    Places an individual tick
    :param y_off: y pos
    :param x: offset of left edge from *left index*
    :param height: height of tickmark (measured from baseline or upper line)
    :param thickness: thickness of tickmark (measured from left edge)
    :param str al: alignment (UPPER or LOWER)
    """

    global renderer
    x0 = x + li - 2
    y0 = y1 = y_off
    if al == UPPER:
        y0 = y_off
        y1 = y_off + height
    if al == LOWER:
        y0 = y_off + SH - height
        y1 = y_off + SH
    renderer.rectangle((x0, y0, x0 + thickness, y1), fill=FG)


X_INDEX_SCALE = 100
X_LEFT_INDEX = X_INDEX_SCALE
X_RIGHT_INDEX = X_INDEX_SCALE * 10 + 1


def pat(y_off, sc, tick_height_scaler, iI, iF, a, b, is_exclusion, a0, b0, al, shift_adj=0):
    """
    Place ticks in a pattern
    a+bN (N ∈ Z) defines the patterning (which ticks to place)
    a0+b0N (N ∈ Z) defines the exclusion patterning (which ticks not to place)

    :param y_off: y pos
    :param Scale sc:
    :param float tick_height_scaler: height modifier (input height scalar like xs, sm, med, lg)
    :param iI: starting index point (100 = left index as X_LEFT_INDEX)
    :param iF: ending index point (1001 = right index as X_RIGHT_INDEX)
    :param a: offset from iI
    :param b: multiple value
    :param is_exclusion: exclusion pattern? 1 = yes, 0 = no
    :param a0: offset from iI
    :param b0: multiple value; put placeholders like 1 & 1 in here if e == 0
    :param str al: alignment (UPPER or LOWER)
    :param float shift_adj: how much to adjust the shift from the scale
    """

    h = round(tick_height_scaler * STH)
    for x in range(iI, iF):
        x_scaled = sc.scale_to(x / X_INDEX_SCALE, shift_adj=shift_adj)
        if is_exclusion == 1:
            if x % b - a == 0 and x % b0 - a0 != 0:
                draw_tick(y_off, x_scaled, h, STT, al)
        elif is_exclusion == 0:
            if x % b - a == 0:
                draw_tick(y_off, x_scaled, h, STT, al)


def font_for_family(font_style, font_size):
    """
    :param int font_style: font style (REG or ITALIC)
    :param int font_size: font size
    :return: FreeTypeFont
    """
    font_name = "cmunit.ttf" if font_style == ITALIC else "cmuntt.ttf"
    # font_name = "cmunrm.ttf" # mythical latex edition
    return ImageFont.truetype(font_name, font_size)


def get_size(symbol, font_size, font_style=REG):
    """
    Gets the size dimensions (width, height) of the input text
    :param str symbol: the text
    :param int font_size: font size
    :param int font_style: font style (REG or ITALIC)
    :return: Tuple[int, int]
    """
    font = font_for_family(font_style, font_size)
    if True:
        width, height = font.font.getsize(str(symbol))[0]
        return width, height
    else:
        (x1, y1, x2, y2) = font.getbbox(str(symbol))
        return x2 - x1, y2 - y1


def get_width(s, font_size, font_style):
    """
    Gets the width of the input s
    :param s: symbol (string)
    :param int font_size: font size
    :param int font_style: font style (REG or ITALIC)
    :return: int
    """
    w, h = get_size(s, font_size, font_style)
    return w


def get_height(s, font_size, font_style):
    """
    :param str s: symbol
    :param int font_size: font size
    :param int font_style: font style (REG or ITALIC)
    :return: int
    """
    w, h = get_size(s, font_size, font_style)
    return h


def draw_symbol(color, y_off, symbol, x, y, font_size, font_style, al):
    """
    :param str color: color name that PIL recognizes
    :param y_off: y pos
    :param str|int symbol: content (text or number)
    :param x: offset of centerline from left index (li)
    :param y: offset of base from baseline (LOWER) or top from upper line (UPPER)
    :param int font_size: font size
    :param int font_style: font style (REG or ITALIC)
    :param str al: alignment (UPPER or LOWER)
    """

    if color == 'green':  # Override PIL for green for slide rule symbol conventions
        color = '#228B1E'

    font = font_for_family(font_style, font_size)
    w, h = get_size(symbol, font_size, font_style)

    global renderer
    y0 = y_off
    if al == UPPER:
        y0 += y
    if al == LOWER:
        y0 += SH - 1 - y - h * 1.2
        y0 -= 16  # FIXME hack to lower-align text better
    renderer.text((x + li - round(w / 2) + round(STT / 2), y0), str(symbol), font=font, fill=color)


def extend(image, y, direction, A):
    """
    Used to create bleed for sticker cutouts
    :param Image.Image image: e.g. img, img2, etc.
    :param int y: y pixel row to duplicate
    :param str direction: direction ('up','down')
    :param int A: Amplitude (# of pixels to extend)
    """

    for x in range(0, total_width):
        r, g, b = image.getpixel((x, y))
        bleed_color = (r, g, b)

        if direction == DIR_UP:
            for yi in range(y - A, y):
                image.putpixel((x, yi), bleed_color)

        elif direction == DIR_DOWN:
            for yi in range(y, y + A):
                image.putpixel((x, yi), bleed_color)


# ----------------------3. Scale Generating Function----------------------------


DIGITS = 10


def scale_linear(x):
    return math.log10(x)


def scale_square(x):
    return math.log10(x) / 2


def scale_sqrt(x):
    return math.log10(x / DIGITS) * 2


def scale_sqrt_ten(x):
    return math.log10(x) * 2  # TODO


def scale_cube(x):
    return math.log10(x) / 3


def scale_inverse_linear(x):
    return 1 - math.log10(x)


def scale_inverse_linear_pi_folded(x):
    return 1 - math.log10(math.pi) - math.log10(x)


def scale_log(x):
    return x / DIGITS


def scale_sin(x):
    return math.log10(DIGITS * math.sin(math.radians(x)))


def scale_tan(x):
    return math.log10(DIGITS * math.tan(math.radians(x)))


def scale_tan1(x):
    return None  # TODO


def scale_tan2(x):
    return None  # TODO


def scale_sin_tan(x):
    return math.log10(DIGITS * DIGITS * (math.sin(math.radians(x)) + math.tan(math.radians(x))) / 2)


def scale_pythagorean(x):
    return math.log10(math.sqrt(1 - x ** 2))


def scale_log_log1(x):
    return None  # TODO


def scale_log_log2(x):
    return None  # TODO


def scale_log_log3(x):
    return None  # TODO


def angle_opp(x):
    """The opposite angle in degrees across a right triangle."""
    return 90 - x


class Scale:

    def __init__(self, left_sym: str, right_sym: str, gen_fn: callable, shift: float = 0, col=FG, key=None):
        self.left_sym = left_sym
        """left scale symbol"""
        self.right_sym = right_sym
        """right scale symbol"""
        self.gen_fn = gen_fn
        """generating function (producing a fraction of output width)"""
        self.shift = shift
        """scale shift from left index (as a fraction of output width)"""
        self.col = col
        """symbol color"""
        self.key = key or left_sym

    def scale_fraction(self, x, shift_adj=0):
        """
        Generating Function for the Scales
        :param Number x:
        :param Number shift_adj:
        :return: int
        """
        return self.shift + shift_adj + self.gen_fn(x)

    def scale_to(self, x, shift_adj=0, scale_width=SL):
        """
        Generating Function for the Scales
        :param Number x:
        :param Number shift_adj:
        :param int scale_width:
        :return: int
        """
        return round(scale_width * self.scale_fraction(x, shift_adj=shift_adj))


SCALE_CONFIGS: dict[str, Scale] = {
    'A': Scale('A', 'x²', scale_square),
    'B': Scale('B', 'x²', scale_square),
    'C': Scale('C', 'x', scale_linear),
    'D': Scale('D', 'x', scale_linear),
    'K': Scale('K', 'x³', scale_cube),
    'R1': Scale('R', '√x', scale_sqrt, key='R1'),
    'R2': Scale('R', '√x', scale_sqrt, shift=-1, key='R2'),
    'CI': Scale('CI', '1/x', scale_inverse_linear, col=RED),
    'DI': Scale('DI', '1/x', scale_inverse_linear, col=RED),
    'CF': Scale('CF', 'πx', scale_linear, shift=1 - math.log10(math.pi)),
    'DF': Scale('DF', 'πx', scale_linear, shift=1 - math.log10(math.pi)),
    'CIF': Scale('CIF', '1/πx', scale_inverse_linear_pi_folded, col=RED),
    'L': Scale('L', 'log x', scale_log),
    'S': Scale('S', 'sin x', scale_sin),
    'T': Scale('T', 'tan x', scale_tan),
    'T1': Scale('T₁', 'tan θ > 45°', scale_tan1, key='T1'),
    'T2': Scale('T₂', 'tan θ < 45°', scale_tan2, key='T2'),
    'ST': Scale('ST', 'θ<5.7°', scale_sin_tan),
    'P': Scale('P', '√1-x²', scale_pythagorean),
    'LL1': Scale('LL₁', '√10x', scale_log_log1),
    'LL2': Scale('LL₂', '√10x', scale_log_log1),
    'W1': Scale('W₁', '√10x', scale_sqrt_ten),
    'W2': Scale('W₂', '√10x', scale_sqrt_ten),
}


class GaugeMark:
    def __init__(self, sym, value, comment=None):
        self.sym = sym
        self.value = value
        self.comment = comment


GAUGE_MARKS: dict[str, GaugeMark] = {
    'e': GaugeMark('e', math.e, comment='base of natural logarithms'),
    'pi': GaugeMark('π', math.pi, comment='ratio of circle circumference to diameter'),
    'R': GaugeMark('R', 180/math.pi, comment='degrees per radian'),
    'rho': GaugeMark('ρ', math.pi/180, comment='radians per degree'),
    'rho_prime': GaugeMark('ρ′', 60 * math.pi/180, comment='radians per minute'),
    'rho_double_prime': GaugeMark('ρ″', 60 * 60 * math.pi/180, comment='radians per second'),
    'M': GaugeMark('M', 1/math.pi, comment='reciprocal of π'),
    'N': GaugeMark('N', 1.341022, comment='mechanical horsepower per kW'),
    'L': GaugeMark('L', 1/math.log10(math.e), comment='ratio of natural log to log base 10'),
    'c': GaugeMark('c', math.pow(10, 1/3), comment='cube root of 10'),
}


def draw_gauge_mark(y_off, gm_key, sc, font_size, al):
    """
    :param int y_off: y pos
    :param Scale sc:
    :param int font_size: font size
    :param str al: alignment (UPPER or LOWER)
    """
    gm = GAUGE_MARKS[gm_key]
    x = sc.scale_to(gm.value)
    h = round(STH)
    draw_tick(y_off, x, h, STT, al)
    draw_symbol(FG, y_off, gm.sym, x, h, font_size, REG, al)


def gen_scale(y_off, sc, al):
    """
    :param int y_off: y pos
    :param Scale sc:
    :param str al: alignment (UPPER or LOWER)
    """

    # Place Index Symbols (Left and Right)
    font_size = 90
    (w2, h2) = get_size(sc.right_sym, font_size, REG)
    draw_symbol(sc.col, y_off, sc.right_sym, 102 / 100 * SL + 0.5 * w2, (SH - h2) / 2, font_size, REG, al)
    (w1, h1) = get_size(sc.left_sym, font_size, REG)
    draw_symbol(sc.col, y_off, sc.left_sym, -2 / 100 * SL - 0.5 * w1, (SH - h1) / 2, font_size, REG, al)

    sc_key = sc.key
    # Exceptions / Special Symbols for R1, R2, S, and T
    if sc_key == 'R1':
        if al == LOWER:
            draw_symbol(FG, y_off, 1, -2 / 100 * SL + 0.5 * w1,
                        SH - 1.3 * ((SH - h1) / 2 + 0.75 * h1), 60, 0, al)
        if al == UPPER:
            draw_symbol(FG, y_off, 1, -2 / 100 * SL + 0.5 * w1,
                        (SH - get_height(sc.left_sym, font_size, REG)) / 2 + 0.75 * h1, 60, 0, al)
    if sc_key == 'R2':
        if al == LOWER:
            draw_symbol(FG, y_off, 2, -2 / 100 * SL + 0.5 * w1,
                        SH - 1.3 * ((SH - h1) / 2 + 0.75 * h1), 60, 0, al)
        if al == UPPER:
            draw_symbol(FG, y_off, 2, -2 / 100 * SL + 0.5 * w1,
                        (SH - h1) / 2 + 0.75 * h1, 60, 0, al)
    if sc_key == 'S':
        draw_symbol(RED, y_off, 'C', -2 / 100 * SL - 0.5 * w1 - get_width('_S', font_size, REG),
                    (SH - h2) / 2, font_size, REG, al)
    if sc_key == 'T':
        draw_symbol(RED, y_off, 'T', -2 / 100 * SL - 0.5 * w1 - get_width('_T', font_size, REG),
                    (SH - h2) / 2, font_size, REG, al)

    # Tick Placement (the bulk!)
    if sc_key == 'C' or sc_key == 'D' or sc_key == 'CI' or sc_key == 'DI':

        # Ticks
        pat(y_off, sc, MED, 100, 1001, 0, 100, 0, 1, 1, al)
        pat(y_off, sc, XL, 100, 1001, 50, 100, 1, 150, 1000, al)
        pat(y_off, sc, SM, 100, 1001, 0, 10, 1, 150, 100, al)
        pat(y_off, sc, SM, 100, 200, 5, 10, 0, 1, 1, al)
        pat(y_off, sc, XS, 100, 200, 0, 1, 1, 0, 5, al)
        pat(y_off, sc, XS, 200, 400, 0, 2, 1, 0, 10, al)
        pat(y_off, sc, XS, 400, 1001, 0, 5, 1, 0, 10, al)

        # 1-10 Labels
        for x in range(1, 11):
            if x == 10:
                draw_symbol(sc.col, y_off, 1, sc.scale_to(x), STH, font_size, REG, al)
            else:
                draw_symbol(sc.col, y_off, x, sc.scale_to(x), STH, font_size, REG, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            draw_symbol(sc.col, y_off, x - 10, sc.scale_to(x / 10), round(STH * 0.85), 60, REG, al)

        # Gauge Points
        draw_tick(y_off, sc.scale_to(math.pi), round(STH), STT, al)
        draw_symbol(sc.col, y_off, 'π', sc.scale_to(math.pi), round(STH), font_size, REG, al)

    if sc_key == "C" or sc_key == "D":
        if y_off < 1600 + oY:
            # r Gauge Point
            draw_tick(y_off, sc.scale_to(18 / math.pi), round(STH), STT, al)
            draw_symbol(FG, y_off, 'r', sc.scale_to(18 / math.pi), round(STH), font_size, REG, al)

    if sc_key == "A" or sc_key == "B":

        # Ticks
        pat(y_off, sc, MED, 100, 1001, 0, 100, 0, 1, 1, al)
        pat(y_off, sc, MED, 1000, 10001, 0, 1000, 0, 1, 1, al)
        pat(y_off, sc, SM, 100, 501, 0, 10, 1, 50, 100, al)
        pat(y_off, sc, SM, 1000, 5001, 0, 100, 1, 500, 1000, al)
        pat(y_off, sc, XL, 100, 1001, 50, 100, 0, 1, 1, al)
        pat(y_off, sc, XL, 1000, 10001, 500, 1000, 0, 1, 1, al)
        pat(y_off, sc, XS, 100, 200, 0, 2, 0, 1, 1, al)
        pat(y_off, sc, XS, 1000, 2000, 0, 20, 0, 1, 1, al)
        pat(y_off, sc, XS, 200, 500, 5, 10, 0, 1, 1, al)
        pat(y_off, sc, XS, 2000, 5000, 50, 100, 0, 1, 1, al)
        pat(y_off, sc, XS, 500, 1001, 0, 10, 1, 0, 50, al)
        pat(y_off, sc, XS, 5000, 10001, 0, 100, 1, 0, 500, al)

        # 1-10 Labels
        for x in range(1, 11):
            if x == 10:
                draw_symbol(FG, y_off, 1, sc.scale_to(x), STH, font_size, REG, al)
                draw_symbol(FG, y_off, 1, sc.scale_to(x * 10), STH, font_size, REG, al)
            else:
                draw_symbol(FG, y_off, x, sc.scale_to(x), STH, font_size, REG, al)
                draw_symbol(FG, y_off, x, sc.scale_to(x * 10), STH, font_size, REG, al)

        # Gauge Points
        draw_gauge_mark(y_off, 'pi', sc, font_size, al)

    if sc_key == "K":
        for b in range(0, 3):
            # Ticks
            ten_to_b = (10 ** b)
            pat(y_off, sc, MED, 100 * ten_to_b, 1000 * ten_to_b + 1, 0, 100 * ten_to_b, 0, 1, 1, al)
            pat(y_off, sc, XL, 100 * ten_to_b, 600 * ten_to_b + 1, 50 * ten_to_b, 100 * ten_to_b, 0, 1, 1, al)
            pat(y_off, sc, SM, 100 * ten_to_b, 300 * ten_to_b + 1, 0, 10 * ten_to_b, 0, 1, 1, al)
            pat(y_off, sc, XS, 100 * ten_to_b, 300 * ten_to_b + 1, 5 * ten_to_b, 10 * ten_to_b, 0, 1, 1, al)
            pat(y_off, sc, XS, 300 * ten_to_b, 600 * ten_to_b + 1, 0, 10 * ten_to_b, 0, 1, 1, al)
            pat(y_off, sc, XS, 600 * ten_to_b, 1000 * ten_to_b + 1, 0, 20 * ten_to_b, 0, 1, 1, al)

        # 1-10 Labels
        f = 75
        for x in range(1, 11):
            if x == 10:
                draw_symbol(FG, y_off, 1, sc.scale_to(x), STH, f, 0, al)
                draw_symbol(FG, y_off, 1, sc.scale_to(x * 10), STH, f, 0, al)
                draw_symbol(FG, y_off, 1, sc.scale_to(x * 100), STH, f, 0, al)
            else:
                draw_symbol(FG, y_off, x, sc.scale_to(x), STH, f, 0, al)
                draw_symbol(FG, y_off, x, sc.scale_to(x * 10), STH, f, 0, al)
                draw_symbol(FG, y_off, x, sc.scale_to(x * 100), STH, f, 0, al)

    if sc_key == 'R1':

        # Ticks
        pat(y_off, sc, MED, 1000, 3200, 0, 100, 0, 1, 1, al)
        pat(y_off, sc, XL, 1000, 2000, 0, 50, 1, 0, 100, al)
        pat(y_off, sc, SM, 2000, 3200, 0, 50, 0, 0, 1000, al)
        pat(y_off, sc, SM, 1000, 2000, 0, 10, 1, 0, 50, al)
        pat(y_off, sc, XS, 1000, 2000, 5, 10, 0, 1, 1, al)
        pat(y_off, sc, XS, 2000, 3180, 0, 10, 1, 0, 50, al)

        # 1-10 Labels
        for x in range(1, 4):
            draw_symbol(FG, y_off, x, sc.scale_to(10 * x), STH, font_size, REG, al)

        # 0.1-3.1 Labels
        for x in range(11, 20):
            draw_symbol(FG, y_off, x - 10, sc.scale_to(x), STH, 60, 0, al)
        for x in range(21, 30):
            draw_symbol(FG, y_off, x - 20, sc.scale_to(x), STH, 60, 0, al)
        draw_symbol(FG, y_off, 1, sc.scale_to(31), STH, 60, 0, al)

        # draw_tick(y_off,sl,round(sth),stt)

    if sc_key == 'R2':

        # Ticks
        pat(y_off, sc, MED, 4000, 10001, 0, 1000, 0, 1, 1, al)
        pat(y_off, sc, XL, 5000, 10000, 500, 1000, 0, 1, 1, al)
        pat(y_off, sc, SM, 3200, 10000, 0, 100, 1, 0, 1000, al)
        pat(y_off, sc, SM, 3200, 5000, 0, 50, 0, 1, 1, al)
        pat(y_off, sc, XS, 3160, 5000, 0, 10, 1, 0, 50, al)
        pat(y_off, sc, XS, 5000, 10000, 0, 20, 1, 0, 100, al)

        # 1-10 Labels
        for x in range(4, 10):
            draw_symbol(FG, y_off, x, sc.scale_to(10 * x), STH, font_size, REG, al)
        draw_symbol(FG, y_off, 1, SL, STH, font_size, REG, al)

        # 0.1-3.1 Labels
        for x in range(32, 40):
            draw_symbol(FG, y_off, x % 10, sc.scale_to(x), STH, 60, REG, al)
        for x in range(41, 50):
            draw_symbol(FG, y_off, x % 10, sc.scale_to(x), STH, 60, REG, al)

    if sc_key == 'CF' or sc_key == 'DF':

        # Ticks
        pat(y_off, sc, MED, 100, 301, 0, 100, 0, 1, 1, al)
        pat(y_off, sc, MED, 400, 1001, 0, 100, 0, 1, 1, al, shift_adj=-1)
        pat(y_off, sc, XL, 200, 301, 50, 100, 0, 1, 1, al)
        pat(y_off, sc, SM, 100, 201, 0, 5, 0, 1, 1, al)
        pat(y_off, sc, SM, 200, 311, 0, 10, 0, 1, 1, al)
        pat(y_off, sc, XL, 320, 1001, 50, 100, 0, 150, 1000, al, shift_adj=-1)
        pat(y_off, sc, SM, 320, 1001, 0, 10, 1, 150, 100, al, shift_adj=-1)
        pat(y_off, sc, XS, 100, 201, 0, 1, 1, 0, 5, al)
        pat(y_off, sc, XS, 200, 314, 0, 2, 1, 0, 10, al)
        pat(y_off, sc, XS, 316, 401, 0, 2, 1, 0, 10, al, shift_adj=-1)
        pat(y_off, sc, XS, 400, 1001, 0, 5, 1, 0, 10, al, shift_adj=-1)

        # 1-10 Labels
        for x in range(1, 4):
            draw_symbol(FG, y_off, x, sc.scale_to(x), STH, font_size, REG, al)
        for x in range(4, 10):
            draw_symbol(FG, y_off, x, sc.scale_to(x) - SL, STH, font_size, REG, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            draw_symbol(FG, y_off, x - 10, sc.scale_to(x / 10), round(STH * 0.85), 60, REG, al)

        # Gauge Points
        draw_tick(y_off, sc.scale_to(math.pi), round(STH), STT, al)
        draw_symbol(FG, y_off, 'π', sc.scale_to(math.pi), round(STH), font_size, REG, al)
        draw_tick(y_off, sc.scale_to(math.pi) - SL, round(STH), STT, al)
        draw_symbol(FG, y_off, 'π', sc.scale_to(math.pi) - SL, round(STH), font_size, REG, al)

    if sc_key == 'CIF':

        # Ticks
        pat(y_off, sc, MED, 100, 301, 0, 100, 0, 1, 1, al)
        pat(y_off, sc, MED, 400, 1001, 0, 100, 0, 1, 1, al, shift_adj=1)

        pat(y_off, sc, XL, 200, 301, 50, 100, 0, 1, 1, al)
        pat(y_off, sc, SM, 100, 201, 0, 5, 0, 1, 1, al)
        pat(y_off, sc, SM, 200, 321, 0, 10, 0, 1, 1, al)
        pat(y_off, sc, XL, 320, 1001, 50, 100, 0, 150, 1000, al, shift_adj=1)
        pat(y_off, sc, SM, 310, 1001, 0, 10, 1, 150, 100, al, shift_adj=1)
        pat(y_off, sc, XS, 100, 201, 0, 1, 1, 0, 5, al)
        pat(y_off, sc, XS, 200, 321, 0, 2, 1, 0, 10, al)
        pat(y_off, sc, XS, 310, 401, 0, 2, 1, 0, 10, al, shift_adj=1)
        pat(y_off, sc, XS, 400, 1001, 0, 5, 1, 0, 10, al, shift_adj=1)

        # 1-10 Labels
        for x in range(4, 10):
            draw_symbol(RED, y_off, x, sc.scale_to(x) + SL, STH, font_size, REG, al)
        for x in range(1, 4):
            draw_symbol(RED, y_off, x, sc.scale_to(x), STH, font_size, REG, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            draw_symbol(RED, y_off, x - 10, sc.scale_to(x / 10), round(STH * 0.85), 60, REG, al)

    if sc_key == 'L':

        # Ticks
        pat(y_off, sc, MED, 0, 1001, 0, 10, 1, 50, 50, al)
        pat(y_off, sc, XL, 1, 1001, 50, 100, 0, 1, 1, al)
        pat(y_off, sc, MXL, 0, 1001, 0, 100, 0, 1, 1, al)
        pat(y_off, sc, XS, 1, 1001, 0, 2, 1, 0, 50, al)

        # Labels
        for x in range(0, 11):
            if x == 0:
                draw_symbol(FG, y_off, 0, sc.scale_to(x), STH, font_size, REG, al)
            if x == 10:
                draw_symbol(FG, y_off, 1, sc.scale_to(x), STH, font_size, REG, al)
            elif x in range(1, 10):
                draw_symbol(FG, y_off, '.' + str(x), sc.scale_to(x), STH, font_size, REG, al)

    if sc_key == 'S':

        # Ticks
        pat(y_off, sc, XL, 1000, 7001, 0, 1000, 0, 1, 1, al)
        pat(y_off, sc, MED, 7000, 10001, 0, 1000, 0, 1, 1, al)
        pat(y_off, sc, XL, 600, 2001, 0, 100, 0, 1, 1, al)
        pat(y_off, sc, SM, 600, 2000, 50, 100, 1, 0, 100, al)
        pat(y_off, sc, XL, 2000, 6000, 500, 1000, 1, 0, 1000, al)
        pat(y_off, sc, SM, 2000, 6000, 0, 100, 1, 0, 500, al)
        pat(y_off, sc, XS, 570, 2000, 0, 10, 1, 0, 50, al)
        pat(y_off, sc, XS, 2000, 3000, 0, 20, 1, 0, 100, al)
        pat(y_off, sc, XS, 3000, 6000, 0, 50, 1, 0, 100, al)
        pat(y_off, sc, SM, 6000, 8501, 500, 1000, 0, 1, 1, al)
        pat(y_off, sc, XS, 6000, 8000, 0, 100, 0, 1, 1, al)

        # Degree Labels

        for x in range(6, 16):
            xi = angle_opp(x)
            draw_symbol(FG, y_off, str(x), sc.scale_to(x) + 1.2 / 2 * get_width(x, 50, ITALIC), STH, 50, REG, al)
            draw_symbol(RED, y_off, str(xi), sc.scale_to(x) - 1.4 / 2 * get_width(xi, 50, ITALIC),
                        STH, 50, ITALIC, al)

        for x in range(16, 20):
            draw_symbol(FG, y_off, str(x), sc.scale_to(x) + 1.2 / 2 * get_width(x, 55, ITALIC), STH, 55, REG, al)

        for x in range(20, 71, 5):
            if (x % 5 == 0 and x < 40) or x % 10 == 0:
                draw_symbol(FG, y_off, str(x), sc.scale_to(x) + 1.2 / 2 * get_width(x, 55, ITALIC), STH, 55, REG,
                            al)
                if x != 20:
                    xi = angle_opp(x)
                    if xi != 40:
                        draw_symbol(RED, y_off, str(xi),
                                    sc.scale_to(x) - 1.4 / 2 * get_width(xi, 55, ITALIC), STH, 55, ITALIC, al)
                    if xi == 40:
                        draw_symbol(RED, y_off + 11, str(40),
                                    sc.scale_to(x) - 1.4 / 2 * get_width(xi, 55, ITALIC), STH, 55, ITALIC, al)

        draw_symbol(FG, y_off, font_size, SL, STH, 60, 0, al)

    if sc_key == 'T':

        # Ticks
        pat(y_off, sc, XL, 600, 2501, 0, 100, 0, 1, 1, al)
        pat(y_off, sc, XL, 600, 1001, 50, 100, 0, 1, 1, al)
        pat(y_off, sc, XL, 2500, 4501, 0, 500, 0, 1, 1, al)
        pat(y_off, sc, MED, 2500, 4501, 0, 100, 0, 1, 1, al)
        draw_tick(y_off, SL, round(STH), STT, al)
        pat(y_off, sc, MED, 600, 951, 50, 100, 0, 1, 1, al)
        pat(y_off, sc, SM, 570, 1001, 0, 10, 1, 0, 50, al)
        pat(y_off, sc, SM, 1000, 2500, 50, 100, 0, 1, 1, al)
        pat(y_off, sc, XS, 570, 1001, 5, 10, 1, 0, 10, al)
        pat(y_off, sc, XS, 1000, 2500, 0, 10, 1, 0, 50, al)
        pat(y_off, sc, XS, 2500, 4501, 0, 20, 1, 0, 100, al)

        # Degree Labels
        f = 1.1 * STH
        for x in range(6, 16):
            draw_symbol(FG, y_off, str(x), sc.scale_to(x) + 1.2 / 2 * get_width(x, 50, ITALIC), f, 50, REG,
                        al)
            xi = angle_opp(x)
            draw_symbol(RED, y_off, str(xi), sc.scale_to(x) - 1.4 / 2 * get_width(xi, 50, ITALIC),
                        f, 50, ITALIC, al)

        for x in range(16, 21):
            draw_symbol(FG, y_off, str(x), sc.scale_to(x) + 1.2 / 2 * get_width(x, 55, ITALIC), f, 55, REG,
                        al)

        for x in range(25, 41, 5):
            if x % 5 == 0:
                draw_symbol(FG, y_off, str(x), sc.scale_to(x) + 1.2 / 2 * get_width(x, 55, ITALIC), f, 55,
                            REG, al)
                xi = angle_opp(x)
                draw_symbol(RED, y_off, str(xi), sc.scale_to(x) - 1.4 / 2 * get_width(xi, 55, ITALIC),
                            f, 55, ITALIC, al)

        draw_symbol(FG, y_off, 45, SL, f * STH, 60, REG, al)

    if sc_key == 'ST':

        # Ticks
        pat(y_off, sc, MED, 100, 551, 0, 50, 0, 1, 1, al)
        pat(y_off, sc, 1.2, 60, 100, 0, 10, 0, 1, 1, al)
        pat(y_off, sc, XL, 60, 100, 5, 10, 0, 1, 1, al)
        pat(y_off, sc, MED, 100, 200, 0, 10, 1, 0, 50, al)
        pat(y_off, sc, SM, 200, 590, 0, 10, 0, 1, 1, al)
        pat(y_off, sc, SM, 57, 100, 0, 1, 0, 1, 1, al)
        pat(y_off, sc, SM, 100, 200, 0, 5, 0, 1, 1, al)
        pat(y_off, sc, XS, 100, 200, 0, 1, 1, 0, 5, al)
        pat(y_off, sc, XS, 200, 400, 0, 2, 1, 0, 10, al)
        pat(y_off, sc, XS, 400, 585, 5, 10, 0, 1, 1, al)

        for x in range(570, 1000):
            if x % 5 == 0 and x % 10 - 0 != 0:
                draw_tick(y_off, sc.scale_to(x / 1000), round(XS * STH), STT, al)

        # Degree Labels
        draw_symbol(FG, y_off, '1°', sc.scale_to(1), STH, font_size, REG, al)
        for x in range(6, 10):
            draw_symbol(FG, y_off, "." + str(x), sc.scale_to(x / 10), STH, font_size, REG, al)
        for x in range(1, 4):
            draw_symbol(FG, y_off, str(x + 0.5), sc.scale_to(x + 0.5), STH, font_size, REG, al)
        for x in range(2, 6):
            draw_symbol(FG, y_off, str(x), sc.scale_to(x), STH, font_size, REG, al)


# ----------------------4. Line Drawing Functions----------------------------

# These functions are unfortunately difficult to modify,
# since I built them with specific numbers rather than variables

def draw_borders(y0, side):  # Place initial borders around scales y0 = vertical offset

    # Main Frame
    horizontals = [y0, 479 + y0, 1119 + y0, 1598 + y0]

    for start in horizontals:
        for x in range(oX, total_width - oX):
            for y in range(start, start + 2):
                sliderule_img.putpixel((x, y), RGB_BLACK)
    verticals = [oX, total_width - oX]
    for start in verticals:
        for x in range(start, start + 2):
            for y in range(y0, 1600 + y0):
                sliderule_img.putpixel((x, y), RGB_BLACK)

    # Top Stator Cut-outs
    verticals = [240 + oX, (total_width - 240) - oX]

    # if side == FRONT_SIDE:
    y_start = y0
    if side == REAR_SIDE:
        y_start = y_start + 1120
    y_end = 480 + y_start
    for start in verticals:
        for x in range(start, start + 2):
            for y in range(y_start, y_end):
                sliderule_img.putpixel((x, y), RGB_BLACK)


def draw_metal_cutoffs(y0, side):
    """
    Use to temporarily view the metal bracket locations
    :param int y0: vertical offset
    :param str side: one of (FRONT,REAR)
    """
    b = 30  # offset of metal from boundary

    # Initial Boundary verticals
    verticals = [480 + oX, total_width - 480 - oX]
    for i in range(0, 2):
        start = verticals[i]
        for x in range(start - 1, start + 1):
            for y in range(y0, 1600 + y0):
                sliderule_img.putpixel((x, y), (230, 230, 230))

        # ~Cute~little~visualization~
        #
        #   0    240   480
        #   |     |     |
        #            1       -0
        #          -----
        #          |   |
        #          |   |
        #       4> |   | <6
        #          |   |
        #          |   |
        #       2  |   |    -1120
        #      -----   |
        #   5> |       |
        #      |       |
        #      ---------
        #       3           -1600
        #   |     |     |

    # Create the left piece using coords format: (x1,x2,y1,y2)
    coords = [[240 + b + oX, 480 - b + oX, b + y0, b + y0],  # 1
              [b + oX, 240 + b + oX, 1120 + b + y0, 1120 + b + y0],  # 2
              [b + oX, 480 - b + oX, 1600 - b + y0, 1600 - b + y0],  # 3
              [240 + b + oX, 240 + b + oX, b + y0, 1120 + b + y0],  # 4
              [b + oX, b + oX, 1120 + b + y0, 1600 - b + y0],  # 5
              [480 - b + oX, 480 - b + oX, b + y0, 1600 - b + y0]]  # 6

    # Symmetrically create the right piece
    for i in range(0, 6):
        (x1, x2, y1, y2) = coords[i]
        coords.append([total_width - x2, total_width - x1, y1, y2])

    # Transfer coords to points for printing (yeah I know it's dumb)
    points = coords
    # If backside, first apply a vertical reflection
    if side == REAR_SIDE:
        points = []
        for i in range(0, 12):
            (x1, x2, y1, y2) = coords[i]
            points.append([x1, x2,
                           2 * y0 + 1600 - y2,
                           2 * y0 + 1600 - y1])
    for i in range(0, 12):
        (x1, x2, y1, y2) = points[i]
        for x in range(x1 - 1, x2 + 1):
            for y in range(y1 - 1, y2 + 1):
                sliderule_img.putpixel((x, y), (234, 36, 98))


# User Prompt Section

RENDER_MODE = 'render'
DIAGNOSTIC_MODE = 'diagnostic'
STICKERPRINT_MODE = 'stickerprint'
VALID_MODES = [RENDER_MODE, DIAGNOSTIC_MODE, STICKERPRINT_MODE]


def prompt_for_mode():
    print("Type render, diagnostic, or stickerprint to set the desired mode")
    print("Each one does something different, so play around with it!")
    mode_accepted = False
    mode = None
    while not mode_accepted:
        mode = input("Mode selection: ")
        if mode in VALID_MODES:
            mode_accepted = True
            continue
        else:
            print("Check your spelling, and try again")
    return mode


SCALE_NAMES = ['A', 'B', 'C', 'D',
               'K', 'R1', 'R2', 'CI',
               'DI', 'CF', 'DF', 'CIF', 'L',
               'S', 'T', 'ST']


# ---------------------- 6. Stickers -----------------------------


should_delineate: bool = True


def draw_box(img_renderer, x0, y0, dx, dy):
    """
    :param ImageDraw.ImageDraw img_renderer:
    :param int x0: First corner of box
    :param int y0: First corner of box
    :param int dx: width extension of box in positive direction
    :param int dy: height extension of box in positive direction
    :return:
    """
    img_renderer.rectangle((x0, y0, x0 + dx, y0 + dy), fill=None, outline=CUT_COLOR)


def draw_corners(img_renderer, x1, y1, x2, y2, arm_width=20):
    """
    :type img_renderer: ImageDraw.ImageDraw
    :param int x1: First corner of box
    :param int y1: First corner of box
    :param int x2: Second corner of box
    :param int y2: Second corner of box
    :param int arm_width: width of extension cross arms
    """

    # horizontal cross arms at 4 corners:
    img_renderer.line((x1 - arm_width, y1, x1 + arm_width, y1), CUT_COLOR)
    img_renderer.line((x1 - arm_width, y2, x1 + arm_width, y2), CUT_COLOR)
    img_renderer.line((x2 - arm_width, y1, x2 + arm_width, y1), CUT_COLOR)
    img_renderer.line((x2 - arm_width, y2, x2 + arm_width, y2), CUT_COLOR)
    # vertical cross arms at 4 corners:
    img_renderer.line((x1, y1 - arm_width, x1, y1 + arm_width), CUT_COLOR)
    img_renderer.line((x1, y2 - arm_width, x1, y2 + arm_width), CUT_COLOR)
    img_renderer.line((x2, y1 - arm_width, x2, y1 + arm_width), CUT_COLOR)
    img_renderer.line((x2, y2 - arm_width, x2, y2 + arm_width), CUT_COLOR)


def transcribe(src_img, dest_img, src_x, src_y, size_x, size_y, target_x, target_y):
    """
    (x0,y0) First corner of SOURCE (rendering)
    (dx,dy) Width and Length of SOURCE chunk to transcribe
    (xT,yT) Target corner of DESTINATION; where to in-plop (into stickerprint)

    Note to self: this is such a bad way to do this, instead of
    transcribing over literally thousands of pixels I should have
    just generated the scales in the place where they are needed

    :param src_img: SOURCE of pixels
    :param dest_img: DESTINATION of pixels
    :param src_x: First corner of SOURCE (rendering)
    :param src_y: First corner of SOURCE (rendering)
    :param size_x: Width of SOURCE chunk to transcribe
    :param size_y: Length of SOURCE chunk to transcribe
    :param target_x: Target corner of DESTINATION; where to in-plop (into stickerprint)
    :param target_y: Target corner of DESTINATION; where to in-plop (into stickerprint)
    :return:
    """

    src_box = src_img.crop((src_x, src_y, src_x + size_x, src_y + size_y))
    dest_img.paste(src_box, (target_x, target_y))


def save_png(img_to_save, basename, output_suffix=None):
    output_filename = f"{basename}{'.' + output_suffix if output_suffix else ''}.png"
    img_to_save.save(output_filename, 'PNG')
    print(f"The result has been saved to {output_filename}")


def main():
    import argparse
    global oX, oY
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--mode',
                             choices=VALID_MODES,
                             help='What to render')
    args_parser.add_argument('--suffix',
                             help='Output filename suffix for variations')
    args_parser.add_argument('--test',
                             action='store_true',
                             help='Output filename for test comparisons')
    args_parser.add_argument('--cutoffs',
                             action='store_true',
                             help='Render the metal cutoffs')
    cli_args = args_parser.parse_args()
    render_mode = cli_args.mode or prompt_for_mode()
    output_suffix = cli_args.suffix or ('test' if cli_args.test else None)
    render_cutoffs = cli_args.cutoffs

    start_time = time.time()

    scale_width = 6500

    if render_mode == RENDER_MODE or render_mode == STICKERPRINT_MODE:
        y_front_end = 1600 + 2 * oY
        if render_mode == RENDER_MODE:
            draw_borders(oY, FRONT_SIDE)
            if render_cutoffs:
                draw_metal_cutoffs(oY, FRONT_SIDE)
            draw_borders(y_front_end, REAR_SIDE)
            if render_cutoffs:
                draw_metal_cutoffs(y_front_end, REAR_SIDE)

        # Front Scale
        gen_scale(110 + oY, SCALE_CONFIGS['L'], LOWER)
        gen_scale(320 + oY, SCALE_CONFIGS['DF'], LOWER)
        gen_scale(800 + oY, SCALE_CONFIGS['CI'], LOWER)
        gen_scale(960 + oY, SCALE_CONFIGS['C'], LOWER)

        gen_scale(480 + oY, SCALE_CONFIGS['CF'], UPPER)
        gen_scale(640 + oY, SCALE_CONFIGS['CIF'], UPPER)
        gen_scale(1120 + oY, SCALE_CONFIGS['D'], UPPER)
        gen_scale(1280 + oY, SCALE_CONFIGS['R1'], UPPER)
        gen_scale(1435 + oY, SCALE_CONFIGS['R2'], UPPER)

        # These are my weirdo alternative universe "brand names", "model name", etc.
        # Feel free to comment them out
        global total_width, li
        draw_symbol(RED, 25 + oY, 'BOGELEX 1000', (total_width - 2 * oX) * 1 / 4 - li, 0, 90, REG, UPPER)
        draw_symbol(RED, 25 + oY, 'LEFT HANDED LIMAÇON 2020', (total_width - 2 * oX) * 2 / 4 - li + oX, 0, 90, REG,
                    UPPER)
        draw_symbol(RED, 25 + oY, 'KWENA & TOOR CO.', (total_width - 2 * oX) * 3 / 4 - li, 0, 90, REG, UPPER)

        # Back Scale
        gen_scale(110 + y_front_end, SCALE_CONFIGS['K'], LOWER)
        gen_scale(320 + y_front_end, SCALE_CONFIGS['A'], LOWER)
        gen_scale(640 + y_front_end, SCALE_CONFIGS['T'], LOWER)
        gen_scale(800 + y_front_end, SCALE_CONFIGS['ST'], LOWER)
        gen_scale(960 + y_front_end, SCALE_CONFIGS['S'], LOWER)

        gen_scale(480 + y_front_end, SCALE_CONFIGS['B'], UPPER)
        gen_scale(1120 + y_front_end, SCALE_CONFIGS['D'], UPPER)
        gen_scale(1360 + y_front_end, SCALE_CONFIGS['DI'], UPPER)

    if render_mode == RENDER_MODE:
        save_png(sliderule_img, 'SlideRuleScales', output_suffix)

    if render_mode == DIAGNOSTIC_MODE:
        global renderer
        # If you're reading this, you're a real one
        # +5 brownie points to you

        oX = 0  # x dir margins
        oY = 0  # y dir margins
        total_width = scale_width + 250 * 2
        total_height = 160 * 24
        li = round(total_width / 2 - SL / 2)  # update left index
        diagnostic_img = Image.new('RGB', (total_width, total_height), BG)
        renderer = ImageDraw.Draw(diagnostic_img)

        x_offset_cl = total_width / 2 - li
        draw_symbol(FG, 50, 'Diagnostic Test Print of Available Scales', x_offset_cl, 0, 140, REG, UPPER)
        draw_symbol(FG, 200, ' '.join(SCALE_NAMES), x_offset_cl, 0, 120, REG, UPPER)
        k = 120 + SH

        for n, sc in enumerate(SCALE_NAMES):
            gen_scale(k + (n + 1) * 200, SCALE_CONFIGS[sc], LOWER)

        save_png(diagnostic_img, 'Diagnostic', output_suffix)

    if render_mode == STICKERPRINT_MODE:
        # Disclaimer, this section also suffers from lack of generality
        # since I built them with specific numbers rather than variables

        # Code Names
        # (fs) | UL,UM,UR [ ML,MM,MR ] LL,LM,LR |
        # (bs) | UL,UM,UR [ ML,MM,MR ] LL,LM,LR |
        # Front Scale, Back Scale
        # Upper Middle Lower, Left Middle Right
        # (18 total stickers)

        oX2 = 50  # x dir margins
        oY2 = 50  # y dir margins
        oA = 50  # overhang amount
        ext = 20  # extension amount
        total_width = scale_width + 2 * oX2
        total_height = 5075

        stickerprint_img = Image.new('RGB', (total_width, total_height), BG)
        renderer = ImageDraw.Draw(stickerprint_img)

        # fsUM,MM,LM:
        l = 0

        l += oY2 + oA
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY, scale_width, 480, oX2, l)
        extend(stickerprint_img, l + 480 - 1, DIR_DOWN, ext)
        if should_delineate:
            draw_corners(renderer, oX2, l - oA, oX2 + scale_width, l + 480)

        l += 480 + oA
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 481, scale_width, 640, oX2, l)
        extend(stickerprint_img, l + 1, DIR_UP, ext)
        extend(stickerprint_img, l + 640 - 1, DIR_DOWN, ext)
        if should_delineate:
            draw_corners(renderer, oX2, l, oX2 + scale_width, l + 640)

        l += 640 + oA
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 1120, scale_width, 480, oX2, l)
        extend(stickerprint_img, l + 1, DIR_UP, ext)
        extend(stickerprint_img, l + 480 - 1, DIR_DOWN, ext)
        if should_delineate:
            draw_corners(renderer, oX2, l, oX2 + scale_width, l + 480 + oA)

        # bsUM,MM,LM:

        l += 480 + oA + oA + oA

        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 1600 + oY, scale_width, 480, oX2, l)
        extend(stickerprint_img, l + 480 - 1, DIR_DOWN, ext)
        if should_delineate:
            draw_corners(renderer, oX2, l - oA, oX2 + scale_width, l + 480)

        l += 480 + oA
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 1600 + oY + 481 - 3, scale_width, 640, oX2, l)
        extend(stickerprint_img, l + 1, DIR_UP, ext)
        extend(stickerprint_img, l + 640 - 1, DIR_DOWN, ext)
        if should_delineate:
            draw_corners(renderer, oX2, l, oX2 + scale_width, l + 640)

        l += 640 + oA
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 1600 + oY + 1120, scale_width, 480, oX2, l)
        extend(stickerprint_img, l + 1, DIR_UP, ext)
        extend(stickerprint_img, l + 480 - 1, DIR_DOWN, ext)
        if should_delineate:
            draw_corners(renderer, oX2, l, oX2 + scale_width, l + 480 + oA)

        yB = 3720

        box = [
            [oA, yB,
             510 + oA, 480 + oA],
            [510 + 3 * oA, yB,
             750 + oA, 640],
            [510 + 750 + 5 * oA, yB,
             750 + oA, 480 + oA]
        ]

        for i, box_i in enumerate(box):
            if should_delineate:
                draw_box(renderer, box_i[0], box_i[1], box_i[2], box_i[3])
                draw_box(renderer, box_i[0], box_i[1] + 640 + oA, box_i[2], box_i[3])

            box_i[0] = round(2 * (6.5 * oA + 510 + 2 * 750) - box_i[0] - box_i[2])

            if should_delineate:
                draw_box(renderer, box_i[0], box_i[1], box_i[2], box_i[3])
                draw_box(renderer, box_i[0], box_i[1] + 640 + oA, box_i[2], box_i[3])

        points = [
            [2 * oA + 120, yB + oA + 160],
            [6 * oA + 510 + 750 + 2 * 160, yB + 160],
            [6 * oA + 510 + 750 + 160, yB + 2 * 160],

            [2 * oA + 120, yB + 640 + oA + 160],
            [6 * oA + 510 + 750 + 160, yB + 640 + oA + oA + 2 * 160],
            [6 * oA + 510 + 750 + 2 * 160, yB + 640 + oA + oA + 160]
        ]

        r = 34  # (2.5mm diameter screw holes)

        for i in range(0, 6):
            (p_x, p_y) = points[i]
            renderer.ellipse((p_x - r, p_y - r,
                              p_x + r, p_y + r),
                             fill=BG,
                             outline=CUT_COLOR)

            p_x = round(2 * (6.5 * oA + 510 + 2 * 750) - p_x)

            renderer.ellipse((p_x - r, p_y - r,
                              p_x + r, p_y + r),
                             fill=BG,
                             outline=CUT_COLOR)

        save_png(stickerprint_img, 'StickerCut', output_suffix)

    print("The program took", round(time.time() - start_time, 2), "seconds to run")


# --------------------------7. Extras----------------------------

# A B C D K R1 R2 CI DI CF DF L S T ST

# Layout:
# |  K,  A  [ B, T, ST, S ] D,  DI    |
# |  L,  DF [ CF,CIF,CI,C ] D, R1, R2 |

# MODEL 1000 -- LEFT HANDED LIMACON 2020 -- KWENA & TOOR CO.S


if __name__ == '__main__':
    main()
