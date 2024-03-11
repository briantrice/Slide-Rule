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
from enum import Enum
from functools import cache

from PIL import Image, ImageFont, ImageDraw


class FontStyle(Enum):
    REG = 0  # font_style regular
    ITALIC = 1  # font_style italic


class Dir(Enum):
    UP = 'up'
    DOWN = 'down'

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
r_global = ImageDraw.Draw(sliderule_img)

SH = 160
"""scale height"""

SL = 5600  # scale length
li = round(total_width / 2 - SL / 2)  # left index offset from left edge

# Ticks, Labels, are referenced from li as to be consistent
STH = 70  # standard tick height
STT = 4  # standard tick thickness

# tick height size factors (h_mod in pat)
XS = 0.5
SM = 0.85
MED = 1
MXL = 1.15
XL = 1.3


class Align(Enum):
    """Scale Alignment (ticks and labels against upper or lower bounds)"""
    UPPER = 'upper'  # Lower alignment
    LOWER = 'lower'  # Upper Alignment


class Side(Enum):
    """Side of the slide (front or rear)"""
    FRONT = 'front'
    REAR = 'rear'

# y_off = 100  # No longer global


# ----------------------2. Fundamental Functions----------------------------


def draw_tick(r, y_off, x, height, thickness, al):
    """
    Places an individual tick
    :param ImageDraw.Draw r:
    :param y_off: y pos
    :param x: offset of left edge from *left index*
    :param height: height of tickmark (measured from baseline or upper line)
    :param thickness: thickness of tickmark (measured from left edge)
    :param Align al: alignment
    """

    x0 = x + li - 2
    y0 = y1 = y_off
    if al == Align.UPPER:
        y0 = y_off
        y1 = y_off + height
    elif al == Align.LOWER:
        y0 = y_off + SH - height
        y1 = y_off + SH
    r.rectangle((x0, y0, x0 + thickness, y1), fill=FG)


INDEX_PER_TENTH = 100
LEFT_INDEX = INDEX_PER_TENTH
RIGHT_INDEX = INDEX_PER_TENTH * 10 + 1


def i_range(first: int, last: int, include_last: bool):
    return range(first, last + (1 if include_last else 0))


def i_range_tenths(first_tenth: int, last_tenth: int, include_last=True) -> range:
    return i_range(first_tenth * INDEX_PER_TENTH, last_tenth * INDEX_PER_TENTH, include_last)


def pat(r, y_off, sc, h_mod, index_range, a, b, is_exclusion, a0, b0, al, shift_adj=0):
    """
    Place ticks in a pattern
    a+bN (N ∈ Z) defines the patterning (which ticks to place)
    a0+b0N (N ∈ Z) defines the exclusion patterning (which ticks not to place)

    :param ImageDraw.Draw r:
    :param y_off: y pos
    :param Scale sc:
    :param float h_mod: height modifier (input height scalar like xs, sm, med, lg)
    :param range index_range: index point range (X_LEFT_INDEX to X_RIGHT_INDEX at widest)
    :param int a: offset from i_i
    :param int b: tick iteration offset
    :param bool is_exclusion: has an exclusion pattern
    :param int a0: exclusion offset from i_i
    :param int b0: exclusion tick iteration offset; put placeholders like 1 & 1 in here if e == 0
    :param Align al: alignment
    :param float shift_adj: how much to adjust the shift from the scale
    """

    h = round(h_mod * STH)
    for x in index_range:
        if x % b - a == 0:
            x_scaled = sc.scale_to(x / INDEX_PER_TENTH, shift_adj=shift_adj, scale_width=SL)
            if is_exclusion:
                if x % b0 - a0 != 0:
                    draw_tick(r, y_off, x_scaled, h, STT, al)
            else:
                draw_tick(r, y_off, x_scaled, h, STT, al)


LATEX_REG_FONT = "cmunrm.ttf"  # mythical LaTeX edition
REG_FONT = "cmuntt.ttf"
ITALIC_FONT = "cmunit.ttf"


@cache
def font_for_family(font_style, font_size):
    """
    :param FontStyle font_style: font style
    :param int font_size: font size
    :return: FreeTypeFont
    """
    font_name = ITALIC_FONT if font_style == FontStyle.ITALIC else REG_FONT
    return ImageFont.truetype(font_name, font_size)


@cache
def get_size(symbol, font_size, font_style=FontStyle.REG):
    """
    Gets the size dimensions (width, height) of the input text
    :param str symbol: the text
    :param int font_size: font size
    :param FontStyle font_style: font style
    :return: Tuple[int, int]
    """
    font = font_for_family(font_style, font_size)
    (x1, y1, x2, y2) = font.getbbox(str(symbol))
    return x2 - x1, y2 - y1 + 19


@cache
def get_width(s, font_size, font_style):
    """
    Gets the width of the input s
    :param s: symbol (string)
    :param int font_size: font size
    :param FontStyle font_style: font style
    :return: int
    """
    w, h = get_size(s, font_size, font_style)
    return w


@cache
def get_height(s, font_size, font_style):
    """
    :param str s: symbol
    :param int font_size: font size
    :param FontStyle font_style: font style
    :return: int
    """
    w, h = get_size(s, font_size, font_style)
    return h


def draw_symbol(color, y_off, symbol, x, y, font_size, font_style, al):
    """
    :param str color: color name that PIL recognizes
    :param y_off: y pos
    :param str symbol: content (text or number)
    :param x: offset of centerline from left index (li)
    :param y: offset of base from baseline (LOWER) or top from upper line (UPPER)
    :param int font_size: font size
    :param FontStyle font_style: font style
    :param Align al: alignment
    """

    if color == 'green':  # Override PIL for green for slide rule symbol conventions
        color = '#228B1E'

    font = font_for_family(font_style, font_size)
    w, h = get_size(symbol, font_size, font_style)

    global r_global
    y0 = y_off
    if al == Align.UPPER:
        y0 += y
    elif al == Align.LOWER:
        y0 += SH - 1 - y - h * 1.2
    r_global.text((x + li - round(w / 2) + round(STT / 2), y0), str(symbol), font=font, fill=color)


def extend(image, y, direction, amplitude):
    """
    Used to create bleed for sticker cutouts
    :param Image.Image image: e.g. img, img2, etc.
    :param int y: y pixel row to duplicate
    :param Dir direction: direction
    :param int amplitude: number of pixels to extend
    """

    for x in range(0, total_width):
        bleed_color = image.getpixel((x, y))

        if direction == Dir.UP:
            for yi in range(y - amplitude, y):
                image.putpixel((x, yi), bleed_color)

        elif direction == Dir.DOWN:
            for yi in range(y, y + amplitude):
                image.putpixel((x, yi), bleed_color)


# ----------------------3. Scale Generating Function----------------------------


DIGITS = 10


def scale_linear(x):
    return math.log10(x)


def scale_square(x):
    return math.log10(x) / 2


def scale_sqrt(x):
    x_frac = x / DIGITS
    return math.log10(x_frac) * 2


def scale_sqrt_ten(x):
    return math.log10(x * 10) * 2


def scale_cube(x):
    return math.log10(x) / 3


def scale_inverse(x):
    return 1 - math.log10(x)


pi_fold_shift = scale_inverse(math.pi)


def scale_inverse_pi_folded(x):
    return pi_fold_shift - math.log10(x)


def scale_log(x):
    return x / DIGITS


def scale_sin(x):
    x_rad = math.radians(x)
    return math.log10(DIGITS * math.sin(x_rad))


def scale_tan(x):
    x_rad = math.radians(x)
    return math.log10(DIGITS * math.tan(x_rad))


def scale_tan_tenth(x):
    x_rad = math.radians(x) / 10
    return math.log10(DIGITS * math.tan(x_rad))


def scale_sin_tan(x):
    x_rad = math.radians(x)
    return math.log10(DIGITS * DIGITS * (math.sin(x_rad) + math.tan(x_rad)) / 2)


def scale_pythagorean(x):
    assert 0 <= x < 1
    return math.log10(math.sqrt(1 - (x ** 2))) + 1


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

    def frac_pos_of(self, x, shift_adj=0):
        """
        Generating Function for the Scales
        :param Number x: the dependent variable
        :param Number shift_adj: how much the scale is shifted, as a fraction of the scale
        :return: float scaled so 0 and 1 are the left and right of the scale
        """
        return self.shift + shift_adj + self.gen_fn(x)

    def pos_of(self, x, scale_width):
        return round(scale_width * self.frac_pos_of(x))

    def scale_to(self, x, shift_adj=0, scale_width=SL):
        """
        Generating Function for the Scales
        :param Number x: the dependent variable
        :param Number shift_adj: how much the scale is shifted, as a fraction of the scale
        :param int scale_width: number of pixels of scale width
        :return: int number of pixels across to the result position
        """
        return round(scale_width * self.frac_pos_of(x, shift_adj=shift_adj))


Scale.A = Scale('A', 'x²', scale_square)
Scale.B = Scale('B', 'x²', scale_square)
Scale.C = Scale('C', 'x', scale_linear)
Scale.CF = Scale('CF', 'πx', scale_linear, key='CF', shift=pi_fold_shift)
Scale.CI = Scale('CI', '1/x', scale_inverse, col=RED)
Scale.CIF = Scale('CIF', '1/πx', scale_inverse_pi_folded, col=RED)
Scale.D = Scale('D', 'x', scale_linear)
Scale.DF = Scale('DF', 'πx', scale_linear, key='DF', shift=pi_fold_shift)
Scale.DI = Scale('DI', '1/x', scale_inverse, col=RED)
Scale.K = Scale('K', 'x³', scale_cube)
Scale.L = Scale('L', 'log x', scale_log)
Scale.LL1 = Scale('LL₁', '√10x', scale_log_log1)
Scale.LL2 = Scale('LL₂', '√10x', scale_log_log2)
Scale.P = Scale('P', '√1-x²', scale_pythagorean, key='P', shift=1)
Scale.R1 = Scale('R', '√x', scale_sqrt, key='R1')
Scale.R2 = Scale('R', '√x', scale_sqrt, key='R2', shift=-1)
Scale.S = Scale('S', 'sin x', scale_sin)
Scale.ST = Scale('ST', 'θ<5.7°', scale_sin_tan)
Scale.T = Scale('T', 'tan x', scale_tan)
Scale.T1 = Scale('T₁', 'tan θ > 45°', scale_tan, key='T1', shift=-0.5)
Scale.T2 = Scale('T₂', 'tan θ < 45°', scale_tan, key='T2', shift=0.5)
Scale.W1 = Scale('W₁', '√10x', scale_sqrt_ten)
Scale.W2 = Scale('W₂', '√10x', scale_sqrt_ten)


class GaugeMark:
    def __init__(self, sym, value, comment=None):
        self.sym = sym
        self.value = value
        self.comment = comment


GAUGE_MARKS: dict[str, GaugeMark] = {
    'e': GaugeMark('e', math.e, comment='base of natural logarithms'),
    'pi': GaugeMark('π', math.pi, comment='ratio of circle circumference to diameter'),
    'R': GaugeMark('r', 180/math.pi/10, comment='degrees per radian'),
    'rho': GaugeMark('ρ', math.pi/180, comment='radians per degree'),
    'rho_prime': GaugeMark('ρ′', 60 * math.pi/180, comment='radians per minute'),
    'rho_double_prime': GaugeMark('ρ″', 60 * 60 * math.pi/180, comment='radians per second'),
    'M': GaugeMark('M', 1/math.pi, comment='reciprocal of π'),
    'N': GaugeMark('N', 1.341022, comment='mechanical horsepower per kW'),
    'L': GaugeMark('L', 1/math.log10(math.e), comment='ratio of natural log to log base 10'),
    'c': GaugeMark('c', math.pow(10, 1/3), comment='cube root of 10'),
}


def draw_gauge_mark(r, y_off, gm_key, sc, font_size, al, col=FG, shift_adj=0):
    """
    :param ImageDraw.Draw r:
    :param int y_off: y pos
    :param str gm_key:
    :param Scale sc:
    :param int font_size: font size
    :param Align al: alignment
    :param str col: color
    :param int shift_adj:
    """
    gm = GAUGE_MARKS[gm_key]
    x = sc.scale_to(gm.value, shift_adj=shift_adj, scale_width=SL)
    h = round(STH)
    draw_tick(r, y_off, x, h, STT, al)
    draw_symbol(col, y_off, gm.sym, x, h * 1.4, font_size, FontStyle.REG, al)


def gen_scale(r, y_off, sc, al):
    """
    :param ImageDraw.Draw r:
    :param int y_off: y pos
    :param Scale sc:
    :param Align al: alignment
    """

    # Place Index Symbols (Left and Right)
    font_size = 90
    reg = FontStyle.REG
    (w2, h2) = get_size(sc.right_sym, font_size, reg)
    draw_symbol(sc.col, y_off, sc.right_sym, 102 / 100 * SL + 0.5 * w2, (SH - h2) / 2, font_size, reg, al)
    (w1, h1) = get_size(sc.left_sym, font_size, reg)
    draw_symbol(sc.col, y_off, sc.left_sym, -2 / 100 * SL - 0.5 * w1, (SH - h1) / 2, font_size, reg, al)

    sc_key = sc.key
    # Exceptions / Special Symbols for R1, R2, S, and T
    if sc_key == 'R1':
        sym = str(1)
        if al == Align.LOWER:
            draw_symbol(FG, y_off, sym, -2 / 100 * SL + 0.5 * w1,
                        SH - 1.3 * ((SH - h1) / 2 + 0.75 * h1), 60, reg, al)
        if al == Align.UPPER:
            draw_symbol(FG, y_off, sym, -2 / 100 * SL + 0.5 * w1,
                        (SH - get_height(sc.left_sym, font_size, reg)) / 2 + 0.75 * h1, 60, reg, al)
    elif sc_key == 'R2':
        sym = str(2)
        if al == Align.LOWER:
            draw_symbol(FG, y_off, sym, -2 / 100 * SL + 0.5 * w1,
                        SH - 1.3 * ((SH - h1) / 2 + 0.75 * h1), 60, reg, al)
        if al == Align.UPPER:
            draw_symbol(FG, y_off, sym, -2 / 100 * SL + 0.5 * w1,
                        (SH - h1) / 2 + 0.75 * h1, 60, reg, al)
    elif sc_key == 'S':
        draw_symbol(RED, y_off, 'C', -2 / 100 * SL - 0.5 * w1 - get_width('_S', font_size, reg),
                    (SH - h2) / 2, font_size, reg, al)
    elif sc_key == 'T':
        draw_symbol(RED, y_off, 'T', -2 / 100 * SL - 0.5 * w1 - get_width('_T', font_size, reg),
                    (SH - h2) / 2, font_size, reg, al)

    full_range = i_range_tenths(1, 10)

    # Tick Placement (the bulk!)
    if sc_key == 'C' or sc_key == 'D' or sc_key == 'CI' or sc_key == 'DI':

        # Ticks
        pat(r, y_off, sc, MED, full_range, 0, 100, False, 1, 1, al)
        pat(r, y_off, sc, XL, full_range, 50, 100, True, 150, 1000, al)
        pat(r, y_off, sc, SM, full_range, 0, 10, True, 150, 100, al)
        range_1to2 = i_range_tenths(1, 2, False)
        pat(r, y_off, sc, SM, range_1to2, 5, 10, False, 1, 1, al)
        pat(r, y_off, sc, XS, range_1to2, 0, 1, True, 0, 5, al)
        pat(r, y_off, sc, XS, i_range_tenths(2, 4, False), 0, 2, True, 0, 10, al)
        pat(r, y_off, sc, XS, i_range_tenths(4, 10), 0, 5, True, 0, 10, al)

        # 1-10 Labels
        for x in range(1, 11):
            sym = leading_digit_of(x)
            draw_symbol(sc.col, y_off, sym, sc.pos_of(x, SL), STH, font_size, reg, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            sym = str(x - 10)
            draw_symbol(sc.col, y_off, sym, sc.pos_of(x / 10, SL), round(STH * 0.85), 60, reg, al)

        # Gauge Points
        draw_gauge_mark(r, y_off, 'pi', sc, font_size, al, col=RED if sc_key == 'CI' or sc_key == 'DI' else FG)

    italic = FontStyle.ITALIC
    if sc_key == 'C' or sc_key == 'D':
        if y_off < 1600 + oY:
            # r Gauge Point
            draw_gauge_mark(r, y_off, 'R', sc, font_size, al)

    elif sc_key == 'A' or sc_key == 'B':

        # Ticks
        pat(r, y_off, sc, MED, full_range, 0, 100, False, 1, 1, al)
        pat(r, y_off, sc, MED, i_range(1000, 10001, True), 0, 1000, False, 1, 1, al)
        pat(r, y_off, sc, SM, i_range_tenths(1, 5), 0, 10, True, 50, 100, al)
        pat(r, y_off, sc, SM, i_range(1000, 5001, True), 0, 100, True, 500, 1000, al)
        pat(r, y_off, sc, XL, full_range, 50, 100, False, 1, 1, al)
        pat(r, y_off, sc, XL, i_range(1000, 10001, True), 500, 1000, False, 1, 1, al)
        pat(r, y_off, sc, XS, i_range_tenths(1, 2), 0, 2, False, 1, 1, al)
        pat(r, y_off, sc, XS, i_range(1000, 2000, True), 0, 20, False, 1, 1, al)
        pat(r, y_off, sc, XS, i_range_tenths(2, 5, False), 5, 10, False, 1, 1, al)
        pat(r, y_off, sc, XS, i_range(2000, 5000, True), 50, 100, False, 1, 1, al)
        pat(r, y_off, sc, XS, i_range_tenths(5, 10), 0, 10, True, 0, 50, al)
        pat(r, y_off, sc, XS, i_range(5000, 10001, True), 0, 100, True, 0, 500, al)

        # 1-10 Labels
        for x in range(1, 11):
            sym = leading_digit_of(x)
            draw_symbol(FG, y_off, sym, sc.pos_of(x, SL), STH, font_size, reg, al)
            draw_symbol(FG, y_off, sym, sc.pos_of(x * 10, SL), STH, font_size, reg, al)

        # Gauge Points
        draw_gauge_mark(r, y_off, 'pi', sc, font_size, al)

    elif sc_key == 'K':
        # Ticks per power of 10
        for b in [10 ** foo for foo in range(0, 3)]:
            pat(r, y_off, sc, MED, i_range_tenths(1 * b, 10 * b, True), 0, 100 * b, False, 1, 1, al)
            pat(r, y_off, sc, XL, i_range_tenths(1 * b, 6 * b, True), 50 * b, 100 * b, False, 1, 1, al)
            pat(r, y_off, sc, SM, i_range_tenths(1 * b, 3 * b, True), 0, 10 * b, False, 1, 1, al)
            pat(r, y_off, sc, XS, i_range_tenths(1 * b, 3 * b, True), 5 * b, 10 * b, False, 1, 1, al)
            pat(r, y_off, sc, XS, i_range_tenths(3 * b, 6 * b, True), 0, 10 * b, False, 1, 1, al)
            pat(r, y_off, sc, XS, i_range_tenths(6 * b, 10 * b, True), 0, 20 * b, False, 1, 1, al)

        # 1-10 Labels
        f = 75
        for x in range(1, 11):
            sym = leading_digit_of(x)
            draw_symbol(FG, y_off, sym, sc.pos_of(x, SL), STH, f, reg, al)
            draw_symbol(FG, y_off, sym, sc.pos_of(x * 10, SL), STH, f, reg, al)
            draw_symbol(FG, y_off, sym, sc.pos_of(x * 100, SL), STH, f, reg, al)

    elif sc_key == 'R1':

        # Ticks
        pat(r, y_off, sc, MED, i_range(1000, 3200, True), 0, 100, False, 1, 1, al)
        pat(r, y_off, sc, XL, i_range(1000, 2000, True), 0, 50, True, 0, 100, al)
        pat(r, y_off, sc, SM, i_range(2000, 3200, True), 0, 50, False, 0, 1000, al)
        pat(r, y_off, sc, SM, i_range(1000, 2000, True), 0, 10, True, 0, 50, al)
        pat(r, y_off, sc, XS, i_range(1000, 2000, True), 5, 10, False, 1, 1, al)
        pat(r, y_off, sc, XS, i_range(2000, 3180, True), 0, 10, True, 0, 50, al)

        # 1-10 Labels
        for x in range(1, 4):
            draw_symbol(FG, y_off, str(x), sc.pos_of(10 * x, SL), STH, font_size, reg, al)

        # 0.1-3.1 Labels
        for x in range(11, 20):
            draw_symbol(FG, y_off, str(x - 10), sc.pos_of(x, SL), STH, 60, reg, al)
        for x in range(21, 30):
            draw_symbol(FG, y_off, str(x - 20), sc.pos_of(x, SL), STH, 60, reg, al)
        draw_symbol(FG, y_off, '1', sc.pos_of(31, SL), STH, 60, reg, al)

        # draw_tick(y_off,sl,round(sth),stt)

    elif sc_key == 'R2':

        # Ticks
        pat(r, y_off, sc, MED, i_range(4000, 10000, True), 0, 1000, False, 1, 1, al)
        pat(r, y_off, sc, XL, i_range(5000, 10000, False), 500, 1000, False, 1, 1, al)
        pat(r, y_off, sc, SM, i_range(3200, 10000, False), 0, 100, True, 0, 1000, al)
        pat(r, y_off, sc, SM, i_range(3200, 5000, False), 0, 50, False, 1, 1, al)
        pat(r, y_off, sc, XS, i_range(3160, 5000, False), 0, 10, True, 0, 50, al)
        pat(r, y_off, sc, XS, i_range(5000, 10000, False), 0, 20, True, 0, 100, al)

        # 1-10 Labels
        for x in range(4, 10):
            draw_symbol(FG, y_off, str(x), sc.pos_of(10 * x, SL), STH, font_size, reg, al)
        draw_symbol(FG, y_off, '1', SL, STH, font_size, reg, al)

        # 0.1-3.1 Labels
        for x in range(32, 40):
            draw_symbol(FG, y_off, str(x % 10), sc.pos_of(x, SL), STH, 60, reg, al)
        for x in range(41, 50):
            draw_symbol(FG, y_off, str(x % 10), sc.pos_of(x, SL), STH, 60, reg, al)

    elif sc_key == 'CF' or sc_key == 'DF':

        # Ticks
        pat(r, y_off, sc, MED, i_range_tenths(1, 3, True), 0, 100, False, 1, 1, al)
        pat(r, y_off, sc, MED, i_range_tenths(4, 10, True), 0, 100, False, 1, 1, al, shift_adj=-1)
        pat(r, y_off, sc, XL, i_range_tenths(2, 3, True), 50, 100, False, 1, 1, al)
        pat(r, y_off, sc, SM, i_range_tenths(1, 2, True), 0, 5, False, 1, 1, al)
        pat(r, y_off, sc, SM, i_range(200, 311, False), 0, 10, False, 1, 1, al)
        pat(r, y_off, sc, XL, i_range(320, RIGHT_INDEX, False), 50, 100, False, 150, 1000, al, shift_adj=-1)
        pat(r, y_off, sc, SM, i_range(320, RIGHT_INDEX, False), 0, 10, True, 150, 100, al, shift_adj=-1)
        pat(r, y_off, sc, XS, i_range(LEFT_INDEX, 201, False), 0, 1, True, 0, 5, al)
        pat(r, y_off, sc, XS, i_range(200, 314, False), 0, 2, True, 0, 10, al)
        pat(r, y_off, sc, XS, i_range(316, 401, False), 0, 2, True, 0, 10, al, shift_adj=-1)
        pat(r, y_off, sc, XS, i_range(400, RIGHT_INDEX, False), 0, 5, True, 0, 10, al, shift_adj=-1)

        # 1-10 Labels
        for x in range(1, 4):
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), STH, font_size, reg, al)
        for x in range(4, 10):
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL) - SL, STH, font_size, reg, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            draw_symbol(FG, y_off, str(x - 10), sc.pos_of(x / 10, SL), round(STH * 0.85), 60, reg, al)

        # Gauge Points
        draw_gauge_mark(r, y_off, 'pi', sc, font_size, al)
        draw_gauge_mark(r, y_off, 'pi', sc, font_size, al, shift_adj=-1)

    elif sc_key == 'CIF':

        # Ticks
        pat(r, y_off, sc, MED, i_range(LEFT_INDEX, 301, False), 0, 100, False, 1, 1, al)
        pat(r, y_off, sc, MED, i_range(400, RIGHT_INDEX, False), 0, 100, False, 1, 1, al, shift_adj=1)

        pat(r, y_off, sc, XL, i_range(200, 301, False), 50, 100, False, 1, 1, al)
        pat(r, y_off, sc, SM, i_range(LEFT_INDEX, 201, False), 0, 5, False, 1, 1, al)
        pat(r, y_off, sc, SM, i_range(200, 321, False), 0, 10, False, 1, 1, al)
        pat(r, y_off, sc, XL, i_range(320, RIGHT_INDEX, False), 50, 100, False, 150, 1000, al, shift_adj=1)
        pat(r, y_off, sc, SM, i_range(310, RIGHT_INDEX, False), 0, 10, True, 150, 100, al, shift_adj=1)
        pat(r, y_off, sc, XS, i_range(LEFT_INDEX, 201, False), 0, 1, True, 0, 5, al)
        pat(r, y_off, sc, XS, i_range(200, 321, False), 0, 2, True, 0, 10, al)
        pat(r, y_off, sc, XS, i_range(310, 401, False), 0, 2, True, 0, 10, al, shift_adj=1)
        pat(r, y_off, sc, XS, i_range(400, RIGHT_INDEX, False), 0, 5, True, 0, 10, al, shift_adj=1)

        # 1-10 Labels
        for x in range(4, 10):
            draw_symbol(RED, y_off, str(x), sc.pos_of(x, SL) + SL, STH, font_size, reg, al)
        for x in range(1, 4):
            draw_symbol(RED, y_off, str(x), sc.pos_of(x, SL), STH, font_size, reg, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            draw_symbol(RED, y_off, str(x - 10), sc.pos_of(x / 10, SL), round(STH * 0.85), 60, reg, al)

    elif sc_key == 'L':

        # Ticks
        range1 = i_range(0, RIGHT_INDEX, True)
        range2 = i_range(1, RIGHT_INDEX, True)
        pat(r, y_off, sc, MED, range1, 0, 10, True, 50, 50, al)
        pat(r, y_off, sc, XL, range2, 50, 100, False, 1, 1, al)
        pat(r, y_off, sc, MXL, range1, 0, 100, False, 1, 1, al)
        pat(r, y_off, sc, XS, range2, 0, 2, True, 0, 50, al)

        # Labels
        for x in range(0, 11):
            if x == 0:
                draw_symbol(FG, y_off, '0', sc.pos_of(x, SL), STH, font_size, reg, al)
            if x == 10:
                draw_symbol(FG, y_off, '1', sc.pos_of(x, SL), STH, font_size, reg, al)
            elif x in range(1, 10):
                draw_symbol(FG, y_off, '.' + str(x), sc.pos_of(x, SL), STH, font_size, reg, al)

    elif sc_key == 'S':

        # Ticks
        pat(r, y_off, sc, XL, i_range(1000, 7001, False), 0, 1000, False, 1, 1, al)
        pat(r, y_off, sc, MED, i_range(7000, 10001, False), 0, 1000, False, 1, 1, al)
        pat(r, y_off, sc, XL, i_range(600, 2001, False), 0, 100, False, 1, 1, al)
        pat(r, y_off, sc, SM, i_range(600, 2000, False), 50, 100, True, 0, 100, al)
        pat(r, y_off, sc, XL, i_range(2000, 6000, False), 500, 1000, True, 0, 1000, al)
        pat(r, y_off, sc, SM, i_range(2000, 6000, False), 0, 100, True, 0, 500, al)
        pat(r, y_off, sc, XS, i_range(570, 2000, False), 0, 10, True, 0, 50, al)
        pat(r, y_off, sc, XS, i_range(2000, 3000, False), 0, 20, True, 0, 100, al)
        pat(r, y_off, sc, XS, i_range(3000, 6000, False), 0, 50, True, 0, 100, al)
        pat(r, y_off, sc, SM, i_range(6000, 8501, False), 500, 1000, False, 1, 1, al)
        pat(r, y_off, sc, XS, i_range(6000, 8000, False), 0, 100, False, 1, 1, al)

        # Degree Labels

        for x in range(6, 16):
            xi = angle_opp(x)
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL) + 1.2 / 2 * get_width(x, 50, italic), STH, 50, reg, al)
            draw_symbol(RED, y_off, str(xi), sc.pos_of(x, SL) - 1.4 / 2 * get_width(xi, 50, italic),
                        STH, 50, italic, al)

        for x in range(16, 20):
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL) + 1.2 / 2 * get_width(x, 55, italic), STH, 55, reg, al)

        for x in range(20, 71, 5):
            if (x % 5 == 0 and x < 40) or x % 10 == 0:
                draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL) + 1.2 / 2 * get_width(x, 55, italic), STH, 55, reg,
                            al)
                if x != 20:
                    xi = angle_opp(x)
                    if xi != 40:
                        draw_symbol(RED, y_off, str(xi),
                                    sc.pos_of(x, SL) - 1.4 / 2 * get_width(xi, 55, italic), STH, 55, italic, al)
                    elif xi == 40:
                        draw_symbol(RED, y_off + 11, str(40),
                                    sc.pos_of(x, SL) - 1.4 / 2 * get_width(xi, 55, italic), STH, 55, italic, al)

        draw_symbol(FG, y_off, '90', SL, STH, 60, reg, al)

    elif sc_key == 'T':

        # Ticks
        pat(r, y_off, sc, XL, i_range(600, 2501, False), 0, 100, False, 1, 1, al)
        pat(r, y_off, sc, XL, i_range(600, RIGHT_INDEX, False), 50, 100, False, 1, 1, al)
        pat(r, y_off, sc, XL, i_range(2500, 4501, False), 0, 500, False, 1, 1, al)
        pat(r, y_off, sc, MED, i_range(2500, 4501, False), 0, 100, False, 1, 1, al)
        draw_tick(r, y_off, SL, round(STH), STT, al)
        pat(r, y_off, sc, MED, i_range(600, 951, False), 50, 100, False, 1, 1, al)
        pat(r, y_off, sc, SM, i_range(570, RIGHT_INDEX, False), 0, 10, True, 0, 50, al)
        pat(r, y_off, sc, SM, i_range(1000, 2500, False), 50, 100, False, 1, 1, al)
        pat(r, y_off, sc, XS, i_range(570, RIGHT_INDEX, False), 5, 10, True, 0, 10, al)
        pat(r, y_off, sc, XS, i_range(1000, 2500, False), 0, 10, True, 0, 50, al)
        pat(r, y_off, sc, XS, i_range(2500, 4501, False), 0, 20, True, 0, 100, al)

        # Degree Labels
        f = 1.1 * STH
        for x in range(6, 16):
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL) + 1.2 / 2 * get_width(x, 50, italic), f, 50, reg,
                        al)
            xi = angle_opp(x)
            draw_symbol(RED, y_off, str(xi), sc.pos_of(x, SL) - 1.4 / 2 * get_width(xi, 50, italic),
                        f, 50, italic, al)

        for x in range(16, 21):
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL) + 1.2 / 2 * get_width(x, 55, italic), f, 55, reg,
                        al)

        for x in range(25, 41, 5):
            if x % 5 == 0:
                draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL) + 1.2 / 2 * get_width(x, 55, italic), f, 55,
                            reg, al)
                xi = angle_opp(x)
                draw_symbol(RED, y_off, str(xi), sc.pos_of(x, SL) - 1.4 / 2 * get_width(xi, 55, italic),
                            f, 55, italic, al)

        draw_symbol(FG, y_off, '45', SL, f, 60, reg, al)

    elif sc_key == 'ST':

        # Ticks
        pat(r, y_off, sc, MED, i_range(LEFT_INDEX, 551, False), 0, 50, False, 1, 1, al)
        pat(r, y_off, sc, 1.2, i_range(60, 100, False), 0, 10, False, 1, 1, al)
        pat(r, y_off, sc, XL, i_range(60, 100, False), 5, 10, False, 1, 1, al)
        pat(r, y_off, sc, MED, i_range(LEFT_INDEX, 200, False), 0, 10, True, 0, 50, al)
        pat(r, y_off, sc, SM, i_range(200, 590, False), 0, 10, False, 1, 1, al)
        pat(r, y_off, sc, SM, i_range(57, 100, False), 0, 1, False, 1, 1, al)
        pat(r, y_off, sc, SM, i_range(LEFT_INDEX, 200, False), 0, 5, False, 1, 1, al)
        pat(r, y_off, sc, XS, i_range(LEFT_INDEX, 200, False), 0, 1, True, 0, 5, al)
        pat(r, y_off, sc, XS, i_range(200, 400, False), 0, 2, True, 0, 10, al)
        pat(r, y_off, sc, XS, i_range(400, 585, False), 5, 10, False, 1, 1, al)

        for x in range(570, 1000):
            if x % 5 == 0 and x % 10 - 0 != 0:
                draw_tick(r, y_off, sc.pos_of(x / 1000, SL), round(XS * STH), STT, al)

        # Degree Labels
        draw_symbol(FG, y_off, '1°', sc.pos_of(1, SL), STH, font_size, reg, al)
        for x in range(6, 10):
            draw_symbol(FG, y_off, '.' + str(x), sc.pos_of(x / 10, SL), STH, font_size, reg, al)
        for x in range(1, 4):
            draw_symbol(FG, y_off, str(x + 0.5), sc.pos_of(x + 0.5, SL), STH, font_size, reg, al)
        for x in range(2, 6):
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), STH, font_size, reg, al)

    elif sc_key == 'P':
        pat(r, y_off, sc, XS, i_range(10, 90, True), 0, 10, False, 1, 1, al)
        # Labels
        label_h = STH * 0.5
        font_s = 45
        for value in [0.995]:
            draw_symbol(FG, y_off, str(value), sc.pos_of(value, SL), label_h, font_s, reg, al)
        for x in range(99, 90, -1):
            x_value = x / 100
            draw_symbol(FG, y_off, str(x_value), sc.pos_of(x_value, SL), label_h, font_s, reg, al)
        for x in range(9, 2, -1):
            x_value = x / 10
            draw_symbol(FG, y_off, str(x_value), sc.pos_of(x_value, SL), label_h, font_s, reg, al)


def leading_digit_of(x: int) -> str:
    return '1' if x == 10 else str(x)


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

    # if side == SlideSide.FRONT:
    y_start = y0
    if side == Side.REAR:
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
    :param Side side:
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
    if side == Side.REAR:
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

    reg = FontStyle.REG
    upper = Align.UPPER
    lower = Align.LOWER
    global r_global
    r = r_global
    if render_mode == RENDER_MODE or render_mode == STICKERPRINT_MODE:
        y_front_end = 1600 + 2 * oY
        if render_mode == RENDER_MODE:
            draw_borders(oY, Side.FRONT)
            if render_cutoffs:
                draw_metal_cutoffs(oY, Side.FRONT)
            draw_borders(y_front_end, Side.REAR)
            if render_cutoffs:
                draw_metal_cutoffs(y_front_end, Side.REAR)

        # Front Scale
        gen_scale(r, 110 + oY, Scale.L, lower)
        gen_scale(r, 320 + oY, Scale.DF, lower)
        gen_scale(r, 800 + oY, Scale.CI, lower)
        gen_scale(r, 960 + oY, Scale.C, lower)

        gen_scale(r, 480 + oY, Scale.CF, upper)
        gen_scale(r, 640 + oY, Scale.CIF, upper)
        gen_scale(r, 1120 + oY, Scale.D, upper)
        gen_scale(r, 1280 + oY, Scale.R1, upper)
        gen_scale(r, 1435 + oY, Scale.R2, upper)

        # These are my weirdo alternative universe "brand names", "model name", etc.
        # Feel free to comment them out
        global total_width, li
        draw_symbol(RED, 25 + oY, 'BOGELEX 1000', (total_width - 2 * oX) * 1 / 4 - li, 0, 90, reg, upper)
        draw_symbol(RED, 25 + oY, 'LEFT HANDED LIMAÇON 2020', (total_width - 2 * oX) * 2 / 4 - li + oX, 0, 90, reg,
                    upper)
        draw_symbol(RED, 25 + oY, 'KWENA & TOOR CO.', (total_width - 2 * oX) * 3 / 4 - li, 0, 90, reg, upper)

        # Back Scale
        gen_scale(r, 110 + y_front_end, Scale.K, lower)
        gen_scale(r, 320 + y_front_end, Scale.A, lower)
        gen_scale(r, 640 + y_front_end, Scale.T, lower)
        gen_scale(r, 800 + y_front_end, Scale.ST, lower)
        gen_scale(r, 960 + y_front_end, Scale.S, lower)

        gen_scale(r, 480 + y_front_end, Scale.B, upper)
        gen_scale(r, 1120 + y_front_end, Scale.D, upper)
        gen_scale(r, 1360 + y_front_end, Scale.DI, upper)

    if render_mode == RENDER_MODE:
        save_png(sliderule_img, 'SlideRuleScales', output_suffix)

    if render_mode == DIAGNOSTIC_MODE:
        # If you're reading this, you're a real one
        # +5 brownie points to you

        oX = 0  # x dir margins
        oY = 0  # y dir margins
        total_width = scale_width + 250 * 2
        total_height = 160 * 24
        li = round(total_width / 2 - SL / 2)  # update left index
        diagnostic_img = Image.new('RGB', (total_width, total_height), BG)
        r_global = ImageDraw.Draw(diagnostic_img)

        x_offset_cl = total_width / 2 - li
        draw_symbol(FG, 50, 'Diagnostic Test Print of Available Scales', x_offset_cl, 0, 140, reg, upper)
        draw_symbol(FG, 200, ' '.join(SCALE_NAMES), x_offset_cl, 0, 120, reg, upper)
        k = 120 + SH

        for n, sc in enumerate(SCALE_NAMES):
            gen_scale(r_global, k + (n + 1) * 200, getattr(Scale, sc), lower)

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

        o_x2 = 50  # x dir margins
        o_y2 = 50  # y dir margins
        o_a = 50  # overhang amount
        ext = 20  # extension amount
        total_width = scale_width + 2 * o_x2
        total_height = 5075

        stickerprint_img = Image.new('RGB', (total_width, total_height), BG)
        r_global = ImageDraw.Draw(stickerprint_img)

        # fsUM,MM,LM:
        l = 0

        l += o_y2 + o_a
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY, scale_width, 480, o_x2, l)
        extend(stickerprint_img, l + 480 - 1, Dir.DOWN, ext)
        if should_delineate:
            draw_corners(r_global, o_x2, l - o_a, o_x2 + scale_width, l + 480)

        l += 480 + o_a
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 481, scale_width, 640, o_x2, l)
        extend(stickerprint_img, l + 1, Dir.UP, ext)
        extend(stickerprint_img, l + 640 - 1, Dir.DOWN, ext)
        if should_delineate:
            draw_corners(r_global, o_x2, l, o_x2 + scale_width, l + 640)

        l += 640 + o_a
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 1120, scale_width, 480, o_x2, l)
        extend(stickerprint_img, l + 1, Dir.UP, ext)
        extend(stickerprint_img, l + 480 - 1, Dir.DOWN, ext)
        if should_delineate:
            draw_corners(r_global, o_x2, l, o_x2 + scale_width, l + 480 + o_a)

        # bsUM,MM,LM:

        l += 480 + o_a + o_a + o_a

        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 1600 + oY, scale_width, 480, o_x2, l)
        extend(stickerprint_img, l + 480 - 1, Dir.DOWN, ext)
        if should_delineate:
            draw_corners(r_global, o_x2, l - o_a, o_x2 + scale_width, l + 480)

        l += 480 + o_a
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 1600 + oY + 481 - 3, scale_width, 640, o_x2, l)
        extend(stickerprint_img, l + 1, Dir.UP, ext)
        extend(stickerprint_img, l + 640 - 1, Dir.DOWN, ext)
        if should_delineate:
            draw_corners(r_global, o_x2, l, o_x2 + scale_width, l + 640)

        l += 640 + o_a
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 1600 + oY + 1120, scale_width, 480, o_x2, l)
        extend(stickerprint_img, l + 1, Dir.UP, ext)
        extend(stickerprint_img, l + 480 - 1, Dir.DOWN, ext)
        if should_delineate:
            draw_corners(r_global, o_x2, l, o_x2 + scale_width, l + 480 + o_a)

        y_b = 3720

        boxes = [
            [o_a, y_b,
             510 + o_a, 480 + o_a],
            [510 + 3 * o_a, y_b,
             750 + o_a, 640],
            [510 + 750 + 5 * o_a, y_b,
             750 + o_a, 480 + o_a]
        ]

        for box in boxes:
            (x0, y0, dx, dy) = box
            if should_delineate:
                draw_box(r_global, x0, y0, dx, dy)
                draw_box(r_global, x0, y0 + 640 + o_a, dx, dy)

            x0 = round(2 * (6.5 * o_a + 510 + 2 * 750) - x0 - dx)

            if should_delineate:
                draw_box(r_global, x0, y0, dx, dy)
                draw_box(r_global, x0, y0 + 640 + o_a, dx, dy)

        points = [
            [2 * o_a + 120, y_b + o_a + 160],
            [6 * o_a + 510 + 750 + 2 * 160, y_b + 160],
            [6 * o_a + 510 + 750 + 160, y_b + 2 * 160],

            [2 * o_a + 120, y_b + 640 + o_a + 160],
            [6 * o_a + 510 + 750 + 160, y_b + 640 + o_a + o_a + 2 * 160],
            [6 * o_a + 510 + 750 + 2 * 160, y_b + 640 + o_a + o_a + 160]
        ]

        hole_radius = 34  # (2.5mm diameter screw holes)

        for point in points:
            (p_x, p_y) = point
            r_global.ellipse((p_x - hole_radius, p_y - hole_radius,
                              p_x + hole_radius, p_y + hole_radius),
                             fill=BG,
                             outline=CUT_COLOR)

            p_x = round(2 * (6.5 * o_a + 510 + 2 * 750) - p_x)

            r_global.ellipse((p_x - hole_radius, p_y - hole_radius,
                              p_x + hole_radius, p_y + hole_radius),
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
