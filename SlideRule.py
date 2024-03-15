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
import re
import time
from enum import Enum
from functools import cache
from unicodedata import digit, category

from PIL import Image, ImageFont, ImageDraw

DEG_RIGHT_ANGLE = 90


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
STS = STT * 3  # minimum tick horizontal separation

# tick height size factors (h_mod in pat)
DOT = 0.25
XS = 0.5
SM = 0.85
MED = 1
ML = 1.15
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


def range_div(first: int, last: int, scale_factor: int, include_last: bool):
    return (x / scale_factor for x in i_range(first, last, include_last))


def range_mul(first: int, last: int, scale_factor: int, include_last: bool):
    return (x * scale_factor for x in i_range(first, last, include_last))


def i_range_tenths(first: int, last: int, include_last=True) -> range:
    return i_range(first * INDEX_PER_TENTH, last * INDEX_PER_TENTH, include_last)


def pat(r, y_off, sc, h_mod, index_range, base_pat, excl_pat, al, scale_factor=100, shift_adj=0, scale_width=SL):
    """
    Place ticks in a pattern
    a+bN (N ∈ Z) defines the patterning (which ticks to place)
    a0+b0N (N ∈ Z) defines the exclusion patterning (which ticks not to place)

    :param ImageDraw.Draw r:
    :param y_off: y pos
    :param Scale sc:
    :param float h_mod: height modifier (input height scalar like xs, sm, med, lg)
    :param Iterable index_range: index point range (X_LEFT_INDEX to X_RIGHT_INDEX at widest)
    :param (int, int) base_pat: the base pattern; a=offset from i_i, b=tick iteration offset
    :param (int, int)|None excl_pat: an exclusion pattern; a0=offset from i_i, b0=tick iteration offset
    :param Align al: alignment
    :param int|None scale_factor: how much to divide the inputs by before scaling (to generate fine decimals)
    :param float shift_adj: how much to adjust the shift from the scale
    """

    h = round(h_mod * STH)
    (a, b) = base_pat
    (a0, b0) = excl_pat or (None, None)
    for x in index_range:
        if x % b - a == 0:
            x_scaled = sc.scale_to(x / scale_factor if scale_factor else x, shift_adj=shift_adj, scale_width=scale_width)
            if excl_pat:
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
    :param int y_off: y pos
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
    (base_sym, exponent) = symbol_with_expon(symbol)
    w, h = get_size(base_sym, font_size, font_style)

    global r_global
    y0 = y_off
    if al == Align.UPPER:
        y0 += y
    elif al == Align.LOWER:
        y0 += SH - 1 - y - h * 1.2
    x0 = x + li - round(w / 2) + round(STT / 2)
    r_global.text((x0, y0), base_sym, font=font, fill=color)
    if exponent:
        expon_font = font
        w_e, h_e = get_size(exponent, font_size, font_style)
        r_global.text((x0 + (w + w_e) / 2, y0 - h / 2), exponent, font=expon_font, fill=color)


RE_EXPON_CARET = r'^(.+)\^([-0-9.a-z]+)$'
RE_EXPON_UNICODE = r'^([^⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)$'


def num_char_convert(char):
    if char == '⁻':
        return '-'
    return digit(char)


def unicode_sub_convert(symbol: str):
    return ''.join(map(str, map(num_char_convert, symbol)))


def symbol_with_expon(symbol: str):
    base_sym = symbol
    expon_sym = None
    matches = re.match(RE_EXPON_CARET, symbol)
    if matches:
        base_sym = matches.group(1)
        expon_sym = matches.group(2)
    else:
        matches = re.match(RE_EXPON_UNICODE, symbol)
        if matches:
            base_sym = matches.group(1)
            expon_sym = unicode_sub_convert(matches.group(2))
    return base_sym, expon_sym


def draw_symbol_expon(color, y_off, exponent, x_base, y_base, sub_font_size, reg, al):
    if len(exponent) == 1 and category(exponent) == 'No':
        exponent = str(digit(exponent))
    (w_e, h_e) = get_size(exponent, sub_font_size, reg)
    y_expon = y_base
    if al == Align.LOWER:
        y_expon = SH - 1.3 * y_expon
    draw_symbol(color, y_off, exponent, x_base + w_e / 2,
                y_expon, sub_font_size, reg, al)


def symbol_with_subscript(symbol: str):
    matches = re.match(r'^([A-Z]+)([0-9₀₁₂₃]+)$', symbol)
    if matches:
        return matches.group(1), ''.join(map(str, map(digit, matches.group(2))))
    return symbol, None


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


def scale_base(x):
    return math.log10(x)


def scale_square(x):
    return math.log10(x) / 2


def scale_sqrt(x):
    x_frac = x / DIGITS
    return math.log10(x_frac) * 2


def scale_sqrt_ten(x):
    return math.log10(x * DIGITS) * 2


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


def scale_log_log(x):
    return math.log10(math.log(x))


def scale_log_log1(x):
    return scale_log_log(x) + 2


def scale_log_log2(x):
    return scale_log_log(x) + 1


def scale_log_log3(x):
    return scale_log_log(x)


def scale_log_log01(x):
    return math.log10(-math.log(x)) + 2


def scale_log_log02(x):
    return math.log10(-math.log(x)) + 1


def scale_log_log03(x):
    return math.log10(-math.log(x))


def angle_opp(x):
    """The opposite angle in degrees across a right triangle."""
    return DEG_RIGHT_ANGLE - x


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


class Scales:
    A = Scale('A', 'x²', scale_square)
    B = Scale('B', 'x²', scale_square)
    C = Scale('C', 'x', scale_base)
    CF = Scale('CF', 'πx', scale_base, key='CF', shift=pi_fold_shift)
    CI = Scale('CI', '1/x', scale_inverse, col=RED)
    CIF = Scale('CIF', '1/πx', scale_inverse_pi_folded, col=RED)
    D = Scale('D', 'x', scale_base)
    DF = Scale('DF', 'πx', scale_base, key='DF', shift=pi_fold_shift)
    DI = Scale('DI', '1/x', scale_inverse, col=RED)
    K = Scale('K', 'x³', scale_cube)
    L = Scale('L', 'log x', scale_log)
    LL1 = Scale('LL₁', 'e^0.01x', scale_log_log1)
    LL2 = Scale('LL₂', 'e^0.1x', scale_log_log2)
    LL3 = Scale('LL₃', 'e^x', scale_log_log3)
    LL01 = Scale('LL₀₁', 'e^-0.01x', scale_log_log01)
    LL02 = Scale('LL₀₂', 'e^-0.1x', scale_log_log02)
    LL03 = Scale('LL₀₃', 'e^-x', scale_log_log03)
    P = Scale('P', '√1-x²', scale_pythagorean, key='P')
    R1 = Scale('R₁', '√x', scale_sqrt, key='R1')
    R2 = Scale('R₂', '√x', scale_sqrt, key='R2', shift=-1)
    S = Scale('S', 'sin x', scale_sin)
    ST = Scale('ST', 'θ<5.7°', scale_sin_tan)
    T = Scale('T', 'tan x', scale_tan)
    T1 = Scale('T₁', 'tan θ > 45°', scale_tan, key='T1', shift=-0.5)
    T2 = Scale('T₂', 'tan θ < 45°', scale_tan, key='T2', shift=0.5)
    W1 = Scale('W₁', '√x', scale_sqrt)
    W2 = Scale('W₂', '√10x', scale_sqrt_ten)


class GaugeMark:
    def __init__(self, sym, value, comment=None):
        self.sym = sym
        self.value = value
        self.comment = comment

    def draw(self, r, y_off, sc, font_size, al, col=FG, shift_adj=0):
        """
        :param ImageDraw.Draw r:
        :param int y_off: y pos
        :param Scale sc:
        :param int font_size: font size
        :param Align al: alignment
        :param str col: color
        :param int shift_adj:
        """
        x = sc.scale_to(self.value, shift_adj=shift_adj, scale_width=SL)
        h = round(STH)
        draw_tick(r, y_off, x, h, STT, al)
        draw_symbol(col, y_off, self.sym, x, h * 1.4, font_size, FontStyle.REG, al)


GaugeMark.e = GaugeMark('e', math.e, comment='base of natural logarithms')
GaugeMark.inv_e = GaugeMark('1/e', 1/math.e, comment='base of natural logarithms')
GaugeMark.pi = GaugeMark('π', math.pi, comment='ratio of circle circumference to diameter')
GaugeMark.inv_pi = GaugeMark('M', 1/math.pi, comment='reciprocal of π')

GaugeMark.deg_per_rad = GaugeMark('r', 180/math.pi/DIGITS, comment='degrees per radian')
GaugeMark.rad_per_deg = GaugeMark('ρ', math.pi/180, comment='radians per degree')
GaugeMark.rad_per_min = GaugeMark('ρ′', 60 * math.pi/180, comment='radians per minute')
GaugeMark.rad_per_sec = GaugeMark('ρ″', 60 * 60 * math.pi/180, comment='radians per second')

GaugeMark.ln_over_log10 = GaugeMark('L', 1/math.log10(math.e), comment='ratio of natural log to log base 10')

GaugeMark.cube_root_ten = GaugeMark('c', math.pow(10, 1 / 3), comment='cube root of 10')

GaugeMark.hp_per_kw = GaugeMark('N', 1.341022, comment='mechanical horsepower per kW')


def gen_scale(r, y_off, sc, al):
    """
    :param ImageDraw.Draw r:
    :param int y_off: y pos
    :param Scale sc:
    :param Align al: alignment
    """

    # Place Index Symbols (Left and Right)
    font_size = 90
    sub_font_size = 60
    reg = FontStyle.REG

    # Right
    (right_sym, exponent) = symbol_with_expon(sc.right_sym)
    (w2, h2) = get_size(right_sym, font_size, reg)
    y2 = (SH - h2) / 2
    x_right = 102 / 100 * SL + 0.5 * w2
    draw_symbol(sc.col, y_off, right_sym, x_right, y2, font_size, reg, al)
    if exponent:
        draw_symbol_expon(sc.col, y_off, exponent, x_right + w2 / 2, y2, sub_font_size, reg, al)

    # Left
    (left_sym, subscript) = symbol_with_subscript(sc.key)
    (w1, h1) = get_size(left_sym, font_size, reg)
    y1 = (SH - h1) / 2
    x_left = -2 / 100 * SL - 0.5 * w1
    draw_symbol(sc.col, y_off, left_sym, x_left, y1, font_size, reg, al)
    if subscript:
        draw_symbol_subscript(sc.col, y_off, subscript, h1, x_left + w1 / 2, y1, sub_font_size, reg, al)

    # Special Symbols for  S, and T
    if sc.gen_fn == scale_sin:
        draw_symbol(RED, y_off, 'C', x_left - get_width('_S', font_size, reg),
                    y2, font_size, reg, al)
    elif sc == Scales.T:
        draw_symbol(RED, y_off, 'T', x_left - get_width('_T', font_size, reg),
                    y2, font_size, reg, al)

    full_range = i_range_tenths(1, 10)

    is_cd = sc.gen_fn == scale_base and sc.shift == 0  # C/D

    # Tick Placement (the bulk!)
    if is_cd or sc.gen_fn == scale_inverse:

        # Ticks
        pat(r, y_off, sc, MED, full_range, (0, 100), None, al)
        pat(r, y_off, sc, XL, full_range, (50, 100), (150, 1000), al)
        pat(r, y_off, sc, SM, full_range, (0, 10), (150, 100), al)
        range_1to2 = i_range_tenths(1, 2, False)
        pat(r, y_off, sc, SM, range_1to2, (5, 10), None, al)
        pat(r, y_off, sc, XS, range_1to2, (0, 1), (0, 5), al)
        pat(r, y_off, sc, XS, i_range_tenths(2, 4, False), (0, 2), (0, 10), al)
        pat(r, y_off, sc, XS, i_range_tenths(4, 10), (0, 5), (0, 10), al)

        # 1-10 Labels
        for x in range(1, 11):
            sym = leading_digit_of(x)
            draw_symbol(sc.col, y_off, sym, sc.pos_of(x, SL), STH, font_size, reg, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            sym = str(x - 10)
            draw_symbol(sc.col, y_off, sym, sc.pos_of(x / 10, SL), round(STH * 0.85), 60, reg, al)

        # Gauge Points
        mark_color = RED if sc.gen_fn == scale_inverse else FG
        GaugeMark.pi.draw(r, y_off, sc, font_size, al, col=mark_color)

    italic = FontStyle.ITALIC

    if is_cd:
        if y_off < 1600 + oY:
            # r Gauge Point
            GaugeMark.deg_per_rad.draw(r, y_off, sc, font_size, al)

    elif sc.gen_fn == scale_square:

        # Ticks
        pat(r, y_off, sc, MED, full_range, (0, 100), None, al)
        pat(r, y_off, sc, MED, i_range(1000, 10001, True), (0, 1000), None, al)
        pat(r, y_off, sc, SM, i_range_tenths(1, 5), (0, 10), (50, 100), al)
        pat(r, y_off, sc, SM, i_range(1000, 5001, True), (0, 100), (500, 1000), al)
        pat(r, y_off, sc, XL, full_range, (50, 100), None, al)
        pat(r, y_off, sc, XL, i_range(1000, 10001, True), (500, 1000), None, al)
        pat(r, y_off, sc, XS, i_range_tenths(1, 2), (0, 2), None, al)
        pat(r, y_off, sc, XS, i_range(1000, 2000, True), (0, 20), None, al)
        pat(r, y_off, sc, XS, i_range_tenths(2, 5, False), (5, 10), None, al)
        pat(r, y_off, sc, XS, i_range(2000, 5000, True), (50, 100), None, al)
        pat(r, y_off, sc, XS, i_range_tenths(5, 10), (0, 10), (0, 50), al)
        pat(r, y_off, sc, XS, i_range(5000, 10001, True), (0, 100), (0, 500), al)

        # 1-10 Labels
        for x in range(1, 11):
            sym = leading_digit_of(x)
            draw_symbol(FG, y_off, sym, sc.pos_of(x, SL), STH, font_size, reg, al)
            draw_symbol(FG, y_off, sym, sc.pos_of(x * 10, SL), STH, font_size, reg, al)

        # Gauge Points
        GaugeMark.pi.draw(r, y_off, sc, font_size, al)

    elif sc == Scales.K:
        # Ticks per power of 10
        for b in [10 ** foo for foo in range(0, 3)]:
            pat(r, y_off, sc, MED, i_range_tenths(1 * b, 10 * b, True), (0, 100 * b), None, al)
            pat(r, y_off, sc, XL, i_range_tenths(1 * b, 6 * b, True), (50 * b, 100 * b), None, al)
            pat(r, y_off, sc, SM, i_range_tenths(1 * b, 3 * b, True), (0, 10 * b), None, al)
            pat(r, y_off, sc, XS, i_range_tenths(1 * b, 3 * b, True), (5 * b, 10 * b), None, al)
            pat(r, y_off, sc, XS, i_range_tenths(3 * b, 6 * b, True), (0, 10 * b), None, al)
            pat(r, y_off, sc, XS, i_range_tenths(6 * b, 10 * b, True), (0, 20 * b), None, al)

        # 1-10 Labels
        f = 75
        for x in range(1, 11):
            sym = leading_digit_of(x)
            draw_symbol(FG, y_off, sym, sc.pos_of(x, SL), STH, f, reg, al)
            draw_symbol(FG, y_off, sym, sc.pos_of(x * 10, SL), STH, f, reg, al)
            draw_symbol(FG, y_off, sym, sc.pos_of(x * 100, SL), STH, f, reg, al)

    elif sc == Scales.R1 or sc == Scales.W1:

        # Ticks
        pat(r, y_off, sc, MED, i_range(1000, 3200, True), (0, 100), None, al)
        pat(r, y_off, sc, XL, i_range(1000, 2000, True), (0, 50), (0, 100), al)
        pat(r, y_off, sc, SM, i_range(2000, 3200, True), (0, 50), None, al)
        pat(r, y_off, sc, SM, i_range(1000, 2000, True), (0, 10), (0, 50), al)
        pat(r, y_off, sc, XS, i_range(1000, 2000, True), (5, 10), None, al)
        pat(r, y_off, sc, XS, i_range(2000, 3180, True), (0, 10), (0, 50), al)

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

    elif sc == Scales.R2:

        # Ticks
        pat(r, y_off, sc, MED, i_range(4000, 10000, True), (0, 1000), None, al)
        pat(r, y_off, sc, XL, i_range(5000, 10000, False), (500, 1000), None, al)
        pat(r, y_off, sc, SM, i_range(3200, 10000, False), (0, 100), (0, 1000), al)
        pat(r, y_off, sc, SM, i_range(3200, 5000, False), (0, 50), None, al)
        pat(r, y_off, sc, XS, i_range(3160, 5000, False), (0, 10), (0, 50), al)
        pat(r, y_off, sc, XS, i_range(5000, 10000, False), (0, 20), (0, 100), al)

        # 1-10 Labels
        for x in range(4, 10):
            draw_symbol(FG, y_off, str(x), sc.pos_of(10 * x, SL), STH, font_size, reg, al)
        draw_symbol(FG, y_off, '1', SL, STH, font_size, reg, al)

        # 0.1-3.1 Labels
        for x in range(32, 40):
            draw_symbol(FG, y_off, str(x % 10), sc.pos_of(x, SL), STH, 60, reg, al)
        for x in range(41, 50):
            draw_symbol(FG, y_off, str(x % 10), sc.pos_of(x, SL), STH, 60, reg, al)

    elif sc == Scales.W2:
        # Ticks
        pat(r, y_off, sc, MED, i_range(4000, 10000, True), (0, 1000), None, al)
        pat(r, y_off, sc, XL, i_range(5000, 10000, False), (500, 1000), None, al)
        pat(r, y_off, sc, SM, i_range(3200, 10000, False), (0, 100), (0, 1000), al)
        pat(r, y_off, sc, SM, i_range(3200, 5000, False), (0, 50), None, al)
        pat(r, y_off, sc, XS, i_range(3160, 5000, False), (0, 10), (0, 50), al)
        pat(r, y_off, sc, XS, i_range(5000, 10000, False), (0, 20), (0, 100), al)

        # 3-10 Labels
        for x in range(3, 11):
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), STH, font_size, reg, al)

        for x in [3.5, 4.5]:
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), STH, 60, reg, al)

    elif sc.gen_fn == scale_base and sc.shift == pi_fold_shift:  # CF/DF

        # Ticks
        pat(r, y_off, sc, MED, i_range_tenths(1, 3, True), (0, 100), None, al)
        pat(r, y_off, sc, MED, i_range_tenths(4, 10, True), (0, 100), None, al, shift_adj=-1)
        pat(r, y_off, sc, XL, i_range_tenths(2, 3, True), (50, 100), None, al)
        pat(r, y_off, sc, SM, i_range_tenths(1, 2, True), (0, 5), None, al)
        pat(r, y_off, sc, SM, i_range(200, 310, True), (0, 10), None, al)
        pat(r, y_off, sc, XL, i_range(320, RIGHT_INDEX, False), (50, 100), None, al, shift_adj=-1)
        pat(r, y_off, sc, SM, i_range(320, RIGHT_INDEX, False), (0, 10), (150, 100), al, shift_adj=-1)
        pat(r, y_off, sc, XS, i_range(LEFT_INDEX, 200, True), (0, 1), (0, 5), al)
        pat(r, y_off, sc, XS, i_range(200, 314, False), (0, 2), (0, 10), al)
        pat(r, y_off, sc, XS, i_range(316, 400, True), (0, 2), (0, 10), al, shift_adj=-1)
        pat(r, y_off, sc, XS, i_range(400, RIGHT_INDEX, False), (0, 5), (0, 10), al, shift_adj=-1)

        # 1-10 Labels
        for x in range(1, 4):
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), STH, font_size, reg, al)
        for x in range(4, 10):
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL) - SL, STH, font_size, reg, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            draw_symbol(FG, y_off, str(x - 10), sc.pos_of(x / 10, SL), round(STH * 0.85), 60, reg, al)

        # Gauge Points
        GaugeMark.pi.draw(r, y_off, sc, font_size, al)
        GaugeMark.pi.draw(r, y_off, sc, font_size, al, shift_adj=-1)

    elif sc == Scales.CIF:

        # Ticks
        pat(r, y_off, sc, MED, i_range(LEFT_INDEX, 300, True), (0, 100), None, al)
        pat(r, y_off, sc, MED, i_range(400, RIGHT_INDEX, False), (0, 100), None, al, shift_adj=1)

        pat(r, y_off, sc, XL, i_range(200, 300, True), (50, 100), None, al)
        pat(r, y_off, sc, SM, i_range(LEFT_INDEX, 200, True), (0, 5), None, al)
        pat(r, y_off, sc, SM, i_range(200, 320, True), (0, 10), None, al)
        pat(r, y_off, sc, XL, i_range(320, RIGHT_INDEX, False), (50, 100), None, al, shift_adj=1)
        pat(r, y_off, sc, SM, i_range(310, RIGHT_INDEX, False), (0, 10), (150, 100), al, shift_adj=1)
        pat(r, y_off, sc, XS, i_range(LEFT_INDEX, 200, True), (0, 1), (0, 5), al)
        pat(r, y_off, sc, XS, i_range(200, 320, True), (0, 2), (0, 10), al)
        pat(r, y_off, sc, XS, i_range(310, 400, True), (0, 2), (0, 10), al, shift_adj=1)
        pat(r, y_off, sc, XS, i_range(400, RIGHT_INDEX, False), (0, 5), (0, 10), al, shift_adj=1)

        # 1-10 Labels
        for x in range(4, 10):
            draw_symbol(RED, y_off, str(x), sc.pos_of(x, SL) + SL, STH, font_size, reg, al)
        for x in range(1, 4):
            draw_symbol(RED, y_off, str(x), sc.pos_of(x, SL), STH, font_size, reg, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            draw_symbol(RED, y_off, str(x - 10), sc.pos_of(x / 10, SL), round(STH * 0.85), 60, reg, al)

    elif sc == Scales.L:

        # Ticks
        range1 = i_range(0, RIGHT_INDEX, True)
        range2 = i_range(1, RIGHT_INDEX, True)
        pat(r, y_off, sc, MED, range1, (0, 10), (50, 50), al)
        pat(r, y_off, sc, XL, range2, (50, 100), None, al)
        pat(r, y_off, sc, ML, range1, (0, 100), None, al)
        pat(r, y_off, sc, XS, range2, (0, 2), (0, 50), al)

        # Labels
        for x in range(0, 11):
            if x == 0:
                draw_symbol(FG, y_off, '0', sc.pos_of(x, SL), STH, font_size, reg, al)
            if x == 10:
                draw_symbol(FG, y_off, '1', sc.pos_of(x, SL), STH, font_size, reg, al)
            elif x in range(1, 10):
                draw_symbol(FG, y_off, '.' + str(x), sc.pos_of(x, SL), STH, font_size, reg, al)

    elif sc.gen_fn == scale_sin:

        # Ticks
        pat(r, y_off, sc, XL, i_range(1000, 7000, True), (0, 1000), None, al)
        pat(r, y_off, sc, MED, i_range(7000, 10000, True), (0, 1000), None, al)
        pat(r, y_off, sc, XL, i_range(600, 2000, True), (0, 100), None, al)
        pat(r, y_off, sc, SM, i_range(600, 2000, False), (50, 100), (0, 100), al)
        pat(r, y_off, sc, XL, i_range(2000, 6000, False), (500, 1000), (0, 1000), al)
        pat(r, y_off, sc, SM, i_range(2000, 6000, False), (0, 100), (0, 500), al)
        pat(r, y_off, sc, XS, i_range(570, 2000, False), (0, 10), (0, 50), al)
        pat(r, y_off, sc, XS, i_range(2000, 3000, False), (0, 20), (0, 100), al)
        pat(r, y_off, sc, XS, i_range(3000, 6000, False), (0, 50), (0, 100), al)
        pat(r, y_off, sc, SM, i_range(6000, 8500, True), (500, 1000), None, al)
        pat(r, y_off, sc, XS, i_range(6000, 8000, False), (0, 100), None, al)

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

        draw_symbol(FG, y_off, str(DEG_RIGHT_ANGLE), SL, STH, 60, reg, al)

    elif sc == Scales.T:

        # Ticks
        pat(r, y_off, sc, XL, i_range(600, 2500, True), (0, 100), None, al)
        pat(r, y_off, sc, XL, i_range(600, RIGHT_INDEX, False), (50, 100), None, al)
        pat(r, y_off, sc, XL, i_range(2500, 4500, True), (0, 500), None, al)
        pat(r, y_off, sc, MED, i_range(2500, 4500, True), (0, 100), None, al)
        draw_tick(r, y_off, SL, round(STH), STT, al)
        pat(r, y_off, sc, MED, i_range(600, 950, True), (50, 100), None, al)
        pat(r, y_off, sc, SM, i_range(570, RIGHT_INDEX, False), (0, 10), (0, 50), al)
        pat(r, y_off, sc, SM, i_range(1000, 2500, False), (50, 100), None, al)
        pat(r, y_off, sc, XS, i_range(570, RIGHT_INDEX, False), (5, 10), (0, 10), al)
        pat(r, y_off, sc, XS, i_range(1000, 2500, False), (0, 10), (0, 50), al)
        pat(r, y_off, sc, XS, i_range(2500, 4500, True), (0, 20), (0, 100), al)

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

    elif sc == Scales.ST:

        # Ticks
        pat(r, y_off, sc, MED, i_range(LEFT_INDEX, 550, True), (0, 50), None, al)
        pat(r, y_off, sc, 1.2, i_range(60, 100, False), (0, 10), None, al)
        pat(r, y_off, sc, XL, i_range(60, 100, False), (5, 10), None, al)
        pat(r, y_off, sc, MED, i_range(LEFT_INDEX, 200, False), (0, 10), (0, 50), al)
        pat(r, y_off, sc, SM, i_range(200, 590, False), (0, 10), None, al)
        pat(r, y_off, sc, SM, i_range(57, 100, False), (0, 1), None, al)
        pat(r, y_off, sc, SM, i_range(LEFT_INDEX, 200, False), (0, 5), None, al)
        pat(r, y_off, sc, XS, i_range(LEFT_INDEX, 200, False), (0, 1), (0, 5), al)
        pat(r, y_off, sc, XS, i_range(200, 400, False), (0, 2), (0, 10), al)
        pat(r, y_off, sc, XS, i_range(400, 585, False), (5, 10), None, al)

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

    elif sc == Scales.P:
        sf = 100000
        fp1 = 10000
        fp2 = 60000
        fp3 = 90000
        fp4 = 98000
        fpe = 99500
        pat(r, y_off, sc, MED, i_range(fp1, fp3, True), (0, 10000), None, al, scale_factor=sf)
        pat(r, y_off, sc, XS, i_range(fp1, fp3, True), (0, 2000), (0, 10000), al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp2, fp3, True), (0, 400), (0, 2000), al, scale_factor=sf)
        pat(r, y_off, sc, MED, i_range(fp3, fpe, True), (0, 1000), None, al, scale_factor=sf)
        pat(r, y_off, sc, XS, i_range(fp3, fp4, True), (0, 200), (0, 1000), al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp3, fp4, True), (0, 100), (0, 200), al, scale_factor=sf)
        pat(r, y_off, sc, XS, i_range(fp4, fpe, True), (0, 100), (0, 1000), al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp4, fpe, True), (0, 20), (0, 1000), al, scale_factor=sf)
        # Labels
        label_h = MED * STH
        font_s = 45
        marks = [v / 100 for v in range(91, 100)] + [v / 10 for v in range(2, 10)] + [0.995]
        for x in marks:
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), label_h, font_s, reg, al)

    elif sc == Scales.LL1:
        # Ticks
        sf = 10000
        fp1 = 10100
        fp2 = 10200
        fp3 = 10500
        fpe = 11100
        pat(r, y_off, sc, MED, i_range(fp1, fp2, True), (0, 10), None, al, scale_factor=sf)
        pat(r, y_off, sc, XS, i_range(fp1, fp2, True), (0, 5), (0, 10), al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp1, fp2, True), (0, 1), (0, 5), al, scale_factor=sf)
        pat(r, y_off, sc, MED, i_range(fp2, fp3, True), (0, 100), None, al, scale_factor=sf)
        pat(r, y_off, sc, XS, i_range(fp2, fp3, True), (0, 10), (0, 100), al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp2, fp3, True), (0, 2), (0, 10), al, scale_factor=sf)
        pat(r, y_off, sc, MED, i_range(fp3, fpe, True), (0, 50), None, al, scale_factor=sf)
        pat(r, y_off, sc, XS, i_range(fp3, fpe, True), (0, 10), (0, 50), al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp3, fpe, True), (0, 50), (0, 10), al, scale_factor=sf)
        # Labels
        label_h = MED * STH
        font_s = 45
        for x in [1.01, 1.011, 1.015, 1.02, 1.025, 1.03, 1.035, 1.04, 1.045]:
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), label_h, font_s, reg, al)
        for x in [1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11]:
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), label_h, font_s, reg, al)

    elif sc == Scales.LL2:
        # Ticks
        sf = 100
        fp1 = 110
        fp2 = 250
        fpe = 270
        pat(r, y_off, sc, MED, i_range(fp1, fpe, True), (0, 10), None, al, scale_factor=sf)
        pat(r, y_off, sc, XS, i_range(fp1, fp2, True), (0, 5), (0, 10), al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp1, fp2, True), (0, 1), (0, 5), al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp2, fpe, True), (0, 2), (0, 10), al, scale_factor=sf)
        # Labels
        label_h = MED * STH
        font_s = 45
        for x in [1.1, 1.11, 1.12, 1.14, 1.16, 1.18, 1.2]:
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), label_h, font_s, reg, al)
        for x in [1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.9, 2, 2.5, 3]:
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), label_h, font_s, reg, al)
        GaugeMark.e.draw(r, y_off, sc, font_s, al, FG)

    elif sc == Scales.LL3:
        # Ticks
        d1 = i_range(25, 100, False)
        pat(r, y_off, sc, XS, d1, (0, 5), None, al, scale_factor=10)
        pat(r, y_off, sc, DOT, d1, (0, 1), (0, 10), al, scale_factor=10)
        d2 = i_range(10, 100, True)
        pat(r, y_off, sc, MED, d2, (0, 10), None, al, scale_factor=None)
        pat(r, y_off, sc, SM, i_range(d2.start, 50, True), (5, 10), None, al, scale_factor=None)
        pat(r, y_off, sc, DOT, i_range(d2.start, 50, True), (0, 1), (0, 5), al, scale_factor=None)
        pat(r, y_off, sc, DOT, i_range(50, d2.stop, True), (0, 2), (0, 10), al, scale_factor=None)
        pat(r, y_off, sc, MED, i_range(100, 1000, True), (0, 100), None, al, scale_factor=None)
        pat(r, y_off, sc, DOT, i_range(100, 500, True), (0, 20), (0, 100), al, scale_factor=None)
        pat(r, y_off, sc, DOT, i_range(500, 1000, True), (0, 50), (0, 100), al, scale_factor=None)
        pat(r, y_off, sc, MED, i_range(1000, 10000, True), (0, 1000), None, al, scale_factor=None)
        pat(r, y_off, sc, DOT, i_range(1000, 5000, True), (0, 200), (0, 1000), al, scale_factor=None)
        pat(r, y_off, sc, DOT, i_range(5000, 10000, True), (0, 500), (0, 1000), al, scale_factor=None)
        pat(r, y_off, sc, MED, i_range(10000, 30000, True), (0, 10000), None, al, scale_factor=None)
        pat(r, y_off, sc, DOT, i_range(10000, 30000, True), (0, 2000), (0, 10000), al, scale_factor=None)
        # Labels
        label_h = MED * STH
        font_s = 45
        for x in [2.5, 3, 3.5, 4, 4.5, 5, 5.5]:
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), label_h, font_s, reg, al)
        for x in [6, 7, 8, 9]:
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), label_h, font_s, reg, al)
        for x in [10, 15, 20, 25, 30, 40, 50, 100]:
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), label_h, font_s, reg, al)
        for x in range(200, 1000, 100):
            draw_symbol(FG, y_off, str(x)[:-2], sc.pos_of(x, SL), label_h, font_s, reg, al)
        draw_symbol(FG, y_off, '10³', sc.pos_of(1000, SL), label_h, font_s, reg, al)
        for x in range(2, 6):
            draw_symbol(FG, y_off, str(x), sc.pos_of(x * 1000, SL), label_h, font_s, reg, al)
        draw_symbol(FG, y_off, '10⁴', sc.pos_of(10000, SL), label_h, font_s, reg, al)
        for x in range(2, 7):
            draw_symbol(FG, y_off, str(x), sc.pos_of(x * 10000, SL), label_h, font_s, reg, al)
        GaugeMark.e.draw(r, y_off, sc, font_s, al, FG)

    elif sc == Scales.LL03:
        # Ticks
        sf = 100000
        fp1 = 100
        fp2 = 1000
        fp3 = 10000
        fpe = 39000
        pat(r, y_off, sc, SM, i_range(2, fp1, False), (0, 10), None, al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(2, 10, False), (0, 1), None, al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(10, fp1, False), (0, 5), None, al, scale_factor=sf)
        pat(r, y_off, sc, MED, i_range(fp1, fp2, False), (0, 100), None, al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp1, fp2, False), (0, 20), (0, 100), al, scale_factor=sf)
        pat(r, y_off, sc, MED, i_range(fp2, fp3, False), (0, 1000), None, al, scale_factor=sf)
        pat(r, y_off, sc, XS, i_range(fp2, fp3, False), (0, 200), (0, 1000), al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp2, fp3, False), (0, 100), (0, 200), al, scale_factor=sf)
        pat(r, y_off, sc, MED, i_range(fp3, fpe, False), (0, 5000), None, al, scale_factor=sf)
        pat(r, y_off, sc, XS, i_range(fp3, fpe, False), (0, 1000), (0, 5000), al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp3, fpe, False), (0, 200), (0, 1000), al, scale_factor=sf)
        # Labels
        label_h = STH * MED
        font_s = 45
        for x in [0.39, .35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), label_h, font_s, reg, al)
        for x in [0.01, 0.005]:
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), label_h, font_s, reg, al)
        draw_symbol(FG, y_off, '10⁻³', sc.pos_of(10**-3, SL), label_h, font_s, reg, al)
        draw_symbol(FG, y_off, '10⁻⁴', sc.pos_of(10**-4, SL), label_h, font_s, reg, al)
        GaugeMark.inv_e.draw(r, y_off, sc, font_s, al, FG)

    elif sc == Scales.LL02:
        # Ticks
        sf = 10000
        fp1 = 3500
        fp2 = 7500
        fpe = 9100
        pat(r, y_off, sc, MED, i_range(fp1, fpe, False), (0, 100), None, al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp1, fp2, False), (0, 20), (0, 100), al, scale_factor=sf)
        pat(r, y_off, sc, XS, i_range(fp2, fpe, False), (0, 50), (0, 100), al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp2, fpe, False), (0, 10), (0, 50), al, scale_factor=sf)
        # Labels
        label_h = STH * MED
        font_s = 45
        for x in [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35]:
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), label_h, font_s, reg, al)
        GaugeMark.inv_e.draw(r, y_off, sc, font_s, al, FG)

    elif sc == Scales.LL01:
        # Ticks
        sf = 10000
        fp1 = 9000
        fp2 = 9300
        fpe = 9900
        pat(r, y_off, sc, MED, i_range(fp1, fpe, True), (0, 100), None, al, scale_factor=sf)
        pat(r, y_off, sc, XS, i_range(fp2, fpe, True), (0, 20), (0, 100), al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp2, fpe, True), (0, 5), (0, 20), al, scale_factor=sf)
        pat(r, y_off, sc, XS, i_range(fp1, fp2, True), (0, 20), (0, 100), al, scale_factor=sf)
        pat(r, y_off, sc, DOT, i_range(fp1, fp2, True), (0, 10), (0, 20), al, scale_factor=sf)
        # Labels
        label_h = STH * MED
        font_s = 45
        for x in [0.99, 0.985, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]:
            draw_symbol(FG, y_off, str(x), sc.pos_of(x, SL), label_h, font_s, reg, al)


def draw_symbol_subscript(col, y_off, subscript, h1, x_left, y1, sub_font_size, reg, al):
    (w_s, h_s) = get_size(subscript, sub_font_size, reg)
    y_sub = y1 + 0.75 * h1
    if al == Align.LOWER:
        y_sub = SH - 1.3 * y_sub
    draw_symbol(col, y_off, subscript, x_left + w_s / 2,
                y_sub, sub_font_size, reg, al)


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

class Mode(Enum):
    RENDER = 'render'
    DIAGNOSTIC = 'diagnostic'
    STICKERPRINT = 'stickerprint'


VALID_MODES = [Mode.RENDER, Mode.DIAGNOSTIC, Mode.STICKERPRINT]


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
                             choices=[m.value for m in VALID_MODES],
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
    render_mode = next(mode for mode in Mode if mode.value == (cli_args.mode or prompt_for_mode()))
    output_suffix = cli_args.suffix or ('test' if cli_args.test else None)
    render_cutoffs = cli_args.cutoffs

    start_time = time.time()

    scale_width = 6500

    reg = FontStyle.REG
    upper = Align.UPPER
    lower = Align.LOWER
    global r_global
    r = r_global
    if render_mode == Mode.RENDER or render_mode == Mode.STICKERPRINT:
        y_front_end = 1600 + 2 * oY
        if render_mode == Mode.RENDER:
            draw_borders(oY, Side.FRONT)
            if render_cutoffs:
                draw_metal_cutoffs(oY, Side.FRONT)
            draw_borders(y_front_end, Side.REAR)
            if render_cutoffs:
                draw_metal_cutoffs(y_front_end, Side.REAR)

        # Front Scale
        gen_scale(r, 110 + oY, Scales.L, lower)
        gen_scale(r, 320 + oY, Scales.DF, lower)
        gen_scale(r, 800 + oY, Scales.CI, lower)
        gen_scale(r, 960 + oY, Scales.C, lower)

        gen_scale(r, 480 + oY, Scales.CF, upper)
        gen_scale(r, 640 + oY, Scales.CIF, upper)
        gen_scale(r, 1120 + oY, Scales.D, upper)
        gen_scale(r, 1280 + oY, Scales.R1, upper)
        gen_scale(r, 1435 + oY, Scales.R2, upper)

        # These are my weirdo alternative universe "brand names", "model name", etc.
        # Feel free to comment them out
        global total_width, li
        draw_symbol(RED, 25 + oY, 'BOGELEX 1000', (total_width - 2 * oX) * 1 / 4 - li, 0, 90, reg, upper)
        draw_symbol(RED, 25 + oY, 'LEFT HANDED LIMAÇON 2020', (total_width - 2 * oX) * 2 / 4 - li + oX, 0, 90, reg,
                    upper)
        draw_symbol(RED, 25 + oY, 'KWENA & TOOR CO.', (total_width - 2 * oX) * 3 / 4 - li, 0, 90, reg, upper)

        # Back Scale
        gen_scale(r, 110 + y_front_end, Scales.K, lower)
        gen_scale(r, 320 + y_front_end, Scales.A, lower)
        gen_scale(r, 640 + y_front_end, Scales.T, lower)
        gen_scale(r, 800 + y_front_end, Scales.ST, lower)
        gen_scale(r, 960 + y_front_end, Scales.S, lower)

        gen_scale(r, 480 + y_front_end, Scales.B, upper)
        gen_scale(r, 1120 + y_front_end, Scales.D, upper)
        gen_scale(r, 1360 + y_front_end, Scales.DI, upper)

    if render_mode == Mode.RENDER:
        save_png(sliderule_img, 'SlideRuleScales', output_suffix)

    if render_mode == Mode.DIAGNOSTIC:
        # If you're reading this, you're a real one
        # +5 brownie points to you

        oX = 0  # x dir margins
        oY = 0  # y dir margins
        total_width = scale_width + 250 * 2

        scale_names = ['A', 'B', 'C', 'D',
                       'K', 'R1', 'R2', 'CI',
                       'DI', 'CF', 'DF', 'CIF', 'L',
                       'S', 'T', 'ST', 'P',
                       'LL1', 'LL2', 'LL3',
                       'LL01', 'LL02', 'LL03',
                       'W1', 'W2']

        total_height = SH * 10 + len(scale_names) * SH
        li = round(total_width / 2 - SL / 2)  # update left index
        diagnostic_img = Image.new('RGB', (total_width, total_height), BG)
        r_global = ImageDraw.Draw(diagnostic_img)

        x_offset_cl = total_width / 2 - li
        draw_symbol(FG, 50, 'Diagnostic Test Print of Available Scales', x_offset_cl, 0, 140, reg, upper)
        draw_symbol(FG, 200, 'A B C D K R1 R2 CI DI CF DF CIF L S T ST', x_offset_cl, 0, 120, reg, upper)
        k = 120 + SH

        for n, sc in enumerate(scale_names):
            gen_scale(r_global, k + (n + 1) * 200, getattr(Scales, sc), lower)

        save_png(diagnostic_img, 'Diagnostic', output_suffix)

    if render_mode == Mode.STICKERPRINT:
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
