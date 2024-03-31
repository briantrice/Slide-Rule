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
import unicodedata

from PIL import Image, ImageFont, ImageDraw

TAU = math.tau
PI = math.pi
PI_HALF = PI / 2
DEG_FULL = 360
DEG_SEMI = 180
DEG_RIGHT_ANGLE = 90


class BleedDir(Enum):
    UP = 'up'
    DOWN = 'down'


# ----------------------1. Setup----------------------------


WHITE = 'white'
BLACK = 'black'
RGB_BLACK = (0, 0, 0)
RGB_BLUE = (0, 0, 255)  # (0,0,255 = blue)
RGB_CUTOFF = (230, 230, 230)
RGB_CUTOFF2 = (234, 36, 98)
RED = 'red'
GREEN = '#228B1E'  # Override PIL for green for slide rule symbol conventions


class ColorScheme:
    def __init__(self, fg_color: str, bg_color: str, dec_color: str = RED):
        self.fg = fg_color
        self.bg = bg_color
        self.decreasing = dec_color

    def color_of_scale(self, sc):
        """:type sc: Scale"""
        return self.fg if sc.increasing else self.decreasing


black_on_white = ColorScheme(BLACK, WHITE)

picket_eye_saver = ColorScheme(BLACK, 'yellow')

BG = WHITE
"""background color white"""
FG = BLACK
"""foreground color black"""
CUT_COLOR = RGB_BLUE  # color which indicates CUT


class HMod(Enum):
    """Tick height size factors (h_mod in pat)"""
    DOT = 0.25
    XS = 0.5
    SM = 0.85
    MED = 1
    LG = 1.15
    LG2 = 1.2
    XL = 1.3


class Geometry:
    """
    Slide Rule Geometric Parameters
    """
    oX: int = 100  # x margins
    oY: int = 100  # y margins
    side_width: int = 8000
    side_height: int = 1600

    SH: int = 160
    """scale height"""

    SL: int = 5600
    """scale length"""

    # Ticks, Labels, are referenced from li as to be consistent
    STH: int = 70
    """standard tick height"""
    STT: int = 4
    """standard tick thickness"""
    PixelsPerCM = 1600 / 6
    PixelsPerIN = PixelsPerCM * 2.54

    def __init__(self, side_wh: (int, int), margins_xy: (int, int), scale_wh: (int, int), tick_wh: (int, int)):
        (self.side_width, self.side_height) = side_wh
        (self.oX, self.oY) = margins_xy
        (self.SL, self.SH) = scale_wh
        (self.STT, self.STH) = tick_wh

    @property
    def total_width(self):
        return self.side_width + 2 * self.oX

    @property
    def midpoint_x(self):
        return self.total_width / 2

    @property
    def print_height(self):
        return self.side_height * 2 + 3 * self.oY

    @property
    def li(self):
        """left index offset from left edge"""
        return round((self.total_width - self.SL) / 2)

    @property
    def min_tick_offset(self):
        """minimum tick horizontal offset"""
        return self.STT * 2  # separate each tick by at least the space of its width

    def tick_h(self, h_mod: HMod = None) -> int:
        return round(self.STH * h_mod.value) if h_mod else self.STH


class FontSize(Enum):
    Title = 140
    Subtitle = 120
    ScaleLBL = 90
    NumXL = 75
    NumLG = 60
    NumMD = 55
    NumSM = 45


class Align(Enum):
    """Scale Alignment (ticks and labels against upper or lower bounds)"""
    UPPER = 'upper'  # Upper alignment
    LOWER = 'lower'  # Lower Alignment


class Side(Enum):
    """Side of the slide (front or rear)"""
    FRONT = 'front'
    REAR = 'rear'


# y_off = 100  # No longer global


# ----------------------2. Fundamental Functions----------------------------


def draw_tick(r, geom, col, y_off, x, height, al):
    """
    Places an individual tick
    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param str col: color
    :param y_off: y pos
    :param x: offset of left edge from *left index*
    :param height: height of tickmark (measured from baseline or upper line)
    :param Align al: alignment
    """

    x0 = x + geom.li - 2
    y0 = y1 = y_off
    if al == Align.UPPER:
        y0 = y_off
        y1 = y_off + height
    elif al == Align.LOWER:
        y0 = y_off + geom.SH - height
        y1 = y_off + geom.SH
    r.rectangle((x0, y0, x0 + geom.STT, y1), fill=col)


def i_range(first: int, last: int, include_last: bool):
    return range(first, last + (1 if include_last else 0))


def range_div(first: int, last: int, scale_factor: int, include_last: bool):
    return (x / scale_factor for x in i_range(first, last, include_last))


def range_mul(first: int, last: int, scale_factor: int, include_last: bool):
    return (x * scale_factor for x in i_range(first, last, include_last))


def i_range_tenths(first: int, last: int, include_last=True, sf=100) -> range:
    return i_range(first * sf, last * sf, include_last)


def pat(r, geom, y_off, sc, h_mod, index_range, base_pat, excl_pat, al, sf=100, shift_adj=0):
    """
    Place ticks in a pattern
    a+bN (N ∈ Z) defines the patterning (which ticks to place)
    a0+b0N (N ∈ Z) defines the exclusion patterning (which ticks not to place)

    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param y_off: y pos
    :param Scale sc:
    :param HMod h_mod: height modifier (input height scalar like xs, sm, med, lg)
    :param Iterable index_range: index point range (X_LEFT_INDEX to X_RIGHT_INDEX at widest)
    :param (int, int) base_pat: the base pattern; a=offset from i_i, b=tick iteration offset
    :param (int, int)|None excl_pat: an exclusion pattern; a0=offset from i_i, b0=tick iteration offset
    :param Align al: alignment
    :param int|None sf: scale factor - how much to divide the inputs by before scaling (to generate fine decimals)
    :param float shift_adj: how much to adjust the shift from the scale
    """

    h = geom.tick_h(h_mod)
    (a, b) = base_pat
    (a0, b0) = excl_pat or (None, None)
    tick_col = sc.col
    scale_width = geom.SL
    for x in index_range:
        if x % b - a == 0:
            x_scaled = sc.scale_to(x / sf if sf else x, scale_width, shift_adj=shift_adj)
            if excl_pat:
                if x % b0 - a0 != 0:
                    draw_tick(r, geom, tick_col, y_off, x_scaled, h, al)
            else:
                draw_tick(r, geom, tick_col, y_off, x_scaled, h, al)


def grad_pat(r, geom, y_off, sc, al, start_value=None, end_value=None):
    """
    Draw a graduated pattern of tick marks across the scale range.
    Determine the lowest digit tick mark spacing and work upwards from there.

    Tick Patterns:
    * 1-.5-.1-.05
    * 1-.5-.1-.02
    * 1-.5-.1-.01
    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param int y_off:
    :param Scale sc:
    :param Align al:
    :param Number start_value:
    :param Number end_value:
    """
    # Ticks
    if not start_value:
        start_value = sc.value_at_start()
    if not end_value:
        end_value = sc.value_at_end()
    if start_value > end_value:
        start_value, end_value = end_value, start_value
    min_tick_offset = geom.min_tick_offset
    log_diff = abs(math.log10(abs((end_value - start_value) / max(start_value, end_value))))
    num_digits = math.ceil(log_diff) + 3
    scale_width = geom.SL
    sf = 10 ** num_digits  # ensure enough precision for int ranges
    # Ensure between 6 and 15 numerals will display? Target log10 in 0.8..1.17
    frac_width = sc.offset_between(start_value, end_value, 1)
    step_numeral = 10 ** (math.floor(math.log10(abs(end_value - start_value)) - 0.5 * frac_width) + num_digits)
    step_half = step_numeral >> 1
    step_tenth = int(step_numeral / 10)  # second level
    tenth_tick_offset = sc.smallest_diff_size_for_delta(start_value, end_value, step_tenth / sf, scale_width)
    if tenth_tick_offset < min_tick_offset:
        step_tenth = step_numeral
    step_last = step_tenth  # last level
    for tick_div in [10, 5, 2]:
        v = step_tenth / tick_div / sf
        smallest_tick_offset = sc.smallest_diff_size_for_delta(start_value, end_value, v, scale_width)
        if smallest_tick_offset >= min_tick_offset:
            step_last = max(round(step_tenth / tick_div), 1)
            break
    sym_col = sc.col
    num_tick = HMod.MED
    half_tick = HMod.XL if step_tenth < step_numeral else HMod.XS
    tenth_tick = HMod.XS if step_last < step_tenth else HMod.DOT
    h = geom.tick_h(num_tick)
    # Ticks and Labels
    i_start = int(start_value * sf)
    i_offset = i_start % step_last
    if i_offset > 0:  # Align to first tick on or after start
        i_start = i_start - i_offset + step_last
    font_s = FontSize.NumLG
    font_style = FontStyle.REG
    numeral_tick_offset = sc.smallest_diff_size_for_delta(start_value, end_value, step_numeral / sf, scale_width)
    max_num_chars = numeral_tick_offset / get_width('_', font_s, font_style)
    if max_num_chars < 4:
        font_s = FontSize.NumSM
    single_digit = max_num_chars < 2
    for i in range(i_start, int(end_value * sf), step_last):
        num = i / sf
        x = sc.scale_to(num, scale_width)
        h_mod = HMod.DOT
        if i % step_numeral == 0:  # Numeral marks
            h_mod = num_tick
            if single_digit and not math.log10(num).is_integer():
                num_sym = str(num)
                if num_sym.endswith('0'):
                    num = int(num_sym[:1])
                elif num_sym.startswith('0.'):
                    num = int(num_sym[-1])
            draw_numeral(r, geom, sym_col, y_off, num, x, h, font_s, font_style, al)
        elif i % step_half == 0:  # Half marks
            h_mod = half_tick
        elif i % step_tenth == 0:  # Tenth marks
            h_mod = tenth_tick
        draw_tick(r, geom, sym_col, y_off, x, geom.tick_h(h_mod), al)


class FontStyle(Enum):
    REG = 0  # regular
    ITALIC = 1  # italic
    BOLD = 2  # bold
    BOLD_ITALIC = 3  # bold italic


# per https://cm-unicode.sourceforge.io/font_table.html:
font_families = {
    'CMUTypewriter': [
        'cmuntt.ttf',  # CMUTypewriter-Regular
        'cmunit.ttf',  # CMUTypewriter-Italic
        'cmuntb.ttf',  # CMUTypewriter-Bold
        'cmuntx.ttf',  # CMUTypewriter-BoldItalic
    ],
    'CMUSansSerif': [
        'cmunss.ttf',  # CMUSansSerif
        'cmunsi.ttf',  # CMUSansSerif-Oblique
        'cmunsx.ttf',  # CMUSansSerif-Bold
        'cmunso.ttf',  # CMUSansSerif-BoldOblique
    ],
    'CMUConcrete': [
        'cmunorm.ttf',  # CMUConcrete-Roman
        'cmunoti.ttf',  # CMUConcrete-Italic
        'cmunobx.ttf',  # CMUConcrete-Bold
        'cmunobi.ttf',  # CMUConcrete-BoldItalic
    ]
}


@cache
def font_for_family(font_style, font_size, font_family='CMUTypewriter'):
    """
    :param FontStyle font_style: font style
    :param FontSize font_size: font size
    :param str font_family: font family, key of font_families
    :return: FreeTypeFont
    """
    font_name = font_families[font_family][font_style.value]
    return ImageFont.truetype(font_name, font_size.value)


@cache
def get_font_size(symbol, font):
    """
    Gets the size dimensions (width, height) of the input text
    :param str symbol: the text
    :param FreeTypeFont font: font
    :return: Tuple[int, int]
    """
    (x1, y1, x2, y2) = font.getbbox(str(symbol))
    return x2 - x1, y2 - y1 + 20


@cache
def get_width(s, font_size, font_style):
    """
    Gets the width of the input s
    :param s: symbol (string)
    :param FontSize font_size: font size
    :param FontStyle font_style: font style
    :return: int
    """
    w, h = get_font_size(s, font_for_family(font_style, font_size))
    return w


DEBUG = False
DRAW_RADICALS = True


def draw_symbol_inner(r, symbol, color, x_left, y_top, font):
    """
    :param ImageDraw.Draw r:
    :param str symbol:
    :param str color:
    :param Number x_left:
    :param Number y_top:
    :param FreeTypeFont font:
    """
    if DEBUG:
        w, h = get_font_size(symbol, font)
        print(f'draw_symbol_inner: {symbol}\t{x_left} {y_top} {w} {h}')
        r.rectangle((x_left, y_top, x_left + w, y_top + h), outline='grey')
        r.rectangle((x_left, y_top, x_left + 10, y_top + 10), outline='navy', width=4)
    r.text((x_left, y_top), symbol, font=font, fill=color)
    if DRAW_RADICALS:
        radicals = re.search(r'[√∛∜]', symbol)
        if radicals:
            n_ch = radicals.start() + 1
            (w_ch, h_ch) = get_font_size('√', font)
            y_ch = y_top + round(h_ch / 4.4)
            w, _ = get_font_size(symbol, font)
            r.line((x_left + w_ch * n_ch - round(w_ch / 10), y_ch, x_left + w, y_ch), width=round(h_ch/14), fill=color)


def draw_symbol(r, geom, color, y_off, symbol, x, y, font_size, font_style, al):
    """
    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param str color: color name that PIL recognizes
    :param int y_off: y pos
    :param str symbol: content (text or number)
    :param x: offset of centerline from left index (li)
    :param y: offset of base from baseline (LOWER) or top from upper line (UPPER)
    :param FontSize font_size: font size
    :param FontStyle font_style: font style
    :param Align al: alignment
    """

    if not symbol:
        return
    font = font_for_family(font_style, font_size)
    (base_sym, exponent, subscript) = symbol_parts(symbol)
    w, h = get_font_size(base_sym, font)

    y_top = y_off
    if al == Align.UPPER:
        y_top += y
    elif al == Align.LOWER:
        y_top += geom.SH - 1 - y - h * 1.2
    x_left = x + geom.li - w / 2 + geom.STT / 2
    draw_symbol_inner(r, base_sym, color, round(x_left), y_top, font)

    if exponent or subscript:
        sub_font_size = FontSize.NumLG if font_size == FontSize.ScaleLBL else font_size
        sub_font = font_for_family(font_style, sub_font_size)
        x_right = round(x_left + w)
        if exponent:
            draw_symbol_sup(r, exponent, color, h, x_right, y_top, sub_font)
        if subscript:
            draw_symbol_sub(r, subscript, color, h, x_right, y_top, sub_font)


def draw_numeral(r, geom, color, y_off, num, x, y, font_size, font_style, al):
    """
    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param str color: color name that PIL recognizes
    :param int y_off: y pos
    :param Number num: number
    :param x: offset of centerline from left index (li)
    :param y: offset of base from baseline (LOWER) or top from upper line (UPPER)
    :param FontSize font_size: font size
    :param FontStyle font_style: font style
    :param Align al: alignment
    """
    if isinstance(num, int):
        num_sym = str(num)
    elif not num.is_integer():
        num_sym = str(num)
        if num_sym.startswith('0.'):
            expon = math.log10(num)
            if expon.is_integer() and abs(expon) > 2:
                num_sym = f'10^{int(expon)}'
            else:
                num_sym = num_sym[1:]  # Omit leading zero digit
    elif num == 0:
        num_sym = '0'
    else:
        expon = math.log10(num)
        if expon.is_integer() and abs(expon) > 2:
            num_sym = f'10^{int(expon)}'
        else:
            num_sym = str(int(num))

    draw_symbol(r, geom, color, y_off, num_sym, x, y, font_size, font_style, al)


RE_EXPON_CARET = re.compile(r'^(.+)\^([-0-9.a-z]+)$')
RE_SUB_UNDERSCORE = re.compile(r'^(.+)_([-0-9.a-z]+)$')
RE_EXPON_UNICODE = re.compile(r'^([^⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)$')
RE_SUB_UNICODE = re.compile(r'^([^₀₁₂₃]+)([₀₁₂₃]+)$')


def num_char_convert(char):
    if char == '⁻':
        return '-'
    return unicodedata.digit(char)


def unicode_sub_convert(symbol: str):
    return ''.join(map(str, map(num_char_convert, symbol)))


def split_symbol_by(symbol: str, text_re: re.Pattern, unicode_re: re.Pattern):
    base_sym = symbol
    subpart_sym = None
    matches = re.match(text_re, symbol)
    if matches:
        base_sym = matches.group(1)
        subpart_sym = matches.group(2)
    else:
        matches = re.match(unicode_re, symbol)
        if matches:
            base_sym = matches.group(1)
            subpart_sym = unicode_sub_convert(matches.group(2))
    return base_sym, subpart_sym


def symbol_with_expon(symbol: str):
    return split_symbol_by(symbol, RE_EXPON_CARET, RE_EXPON_UNICODE)


def symbol_with_subscript(symbol: str):
    return split_symbol_by(symbol, RE_SUB_UNDERSCORE, RE_SUB_UNICODE)


def symbol_parts(symbol: str):
    (base_sym, subscript) = symbol_with_subscript(symbol)
    (base_sym, expon) = symbol_with_expon(base_sym)
    return base_sym, expon, subscript


def draw_symbol_sup(r, sup_sym, color, h_base, x_left, y_base, font):
    if len(sup_sym) == 1 and unicodedata.category(sup_sym) == 'No':
        sup_sym = str(unicodedata.digit(sup_sym))
    draw_symbol_inner(r, sup_sym, color, x_left, y_base - h_base / 2, font)


def draw_symbol_sub(r, sub_sym, color, h_base, x_left, y_base, font):
    draw_symbol_inner(r, sub_sym, color, x_left, y_base + h_base / 2, font)


def extend(image, geom, y, direction, amplitude):
    """
    Used to create bleed for sticker cutouts
    :param Image.Image image: e.g. img, img2, etc.
    :param Geometry geom:
    :param int y: y pixel row to duplicate
    :param BleedDir direction: direction
    :param int amplitude: number of pixels to extend
    """

    for x in range(0, geom.total_width):
        bleed_color = image.getpixel((x, y))

        if direction == BleedDir.UP:
            for yi in range(y - amplitude, y):
                image.putpixel((x, yi), bleed_color)

        elif direction == BleedDir.DOWN:
            for yi in range(y, y + amplitude):
                image.putpixel((x, yi), bleed_color)


# ----------------------3. Scale Generating Function----------------------------


TEN = 10
HUNDRED = TEN * TEN


def gen_base(x): return math.log10(x)
def pos_base(p): return math.pow(TEN, p)
def scale_square(x): return gen_base(x) / 2
def scale_cube(x): return gen_base(x) / 3
def scale_sqrt(x): return gen_base(x) * 2
def scale_sqrt_ten(x): return gen_base(x) * 2
def scale_inverse(x): return 1 - gen_base(x)


pi_fold_shift = scale_inverse(PI)


def scale_log(x): return x / TEN
def scale_sin(x): return gen_base(TEN * math.sin(math.radians(x)))
def scale_cos(x): return gen_base(TEN * math.cos(math.radians(x)))
def scale_tan(x): return gen_base(TEN * math.tan(math.radians(x)))
def scale_cot(x): return gen_base(TEN * math.tan(math.radians(DEG_RIGHT_ANGLE - x)))


def scale_sin_tan_radians(x):
    return gen_base(HUNDRED * (math.sin(x) + math.tan(x)) / 2)


def scale_sin_tan(x):
    return scale_sin_tan_radians(math.radians(x))


def scale_sinh(x): return gen_base(math.sinh(x))
def scale_cosh(x): return gen_base(math.cosh(x))
def scale_tanh(x): return gen_base(math.tanh(x))


def scale_pythagorean(x):
    assert 0 <= x <= 1
    return gen_base(math.sqrt(1 - (x ** 2))) + 1


def scale_hyperbolic(x):
    assert x > 1
    # y = math.sqrt(1+x**2)
    return gen_base(math.sqrt((x ** 2) - 1))


def scale_log_log(x): return gen_base(math.log(x))


def scale_neg_log_log(x): return gen_base(-math.log(x))


def angle_opp(x):
    """The opposite angle in degrees across a right triangle."""
    return DEG_RIGHT_ANGLE - x


class RulePart(Enum):
    STOCK = 'stock'
    SLIDE = 'slide'


class InvertibleFn:
    def __init__(self, fn, inv_fn):
        self.fn = fn
        self.inverse = inv_fn

    def __call__(self, x):
        return self.fn(x)

    def inverted(self):
        return self.__class__(self.inverse, self.fn)

    def compose_with(self, another):
        """
        :param InvertibleFn another:
        :return: InvertibleFn
        """
        return self.__class__(
            lambda x: self.fn(another.fn(x)),
            lambda x: another.inverse(self.inverse(x))
        )

    def __repr__(self):
        return f'Scaler({self.fn}, {self.inverse})'

    def __add__(self, other):
        return self.__class__(
            lambda x: self.fn(x + other),
            lambda x: self.inverse(x) - other
        )

    def __sub__(self, other):
        return self.__class__(
            lambda x: self.fn(x - other),
            lambda x: self.inverse(x) + other
        )

    def __mul__(self, other):
        return self.__class__(
            lambda x: self.fn(x * other),
            lambda x: self.inverse(x) / other
        )


def unit(x): return x


class Invertibles:
    Unit = InvertibleFn(unit, unit)
    F_to_C = InvertibleFn(lambda f: (f - 32) * 5 / 9, lambda c: (c * 9 / 5) + 32)
    cm_to_in = InvertibleFn(lambda x_mm: x_mm / 2.54, lambda x_in: x_in * 2.54)
    mm_to_in = InvertibleFn(lambda x_mm: x_mm / 25.4, lambda x_in: x_in * 25.4)


class Scaler(InvertibleFn):
    """Encapsulates a generating function and its inverse.
    The generating function takes X and returns the fractional position in the unit output space it should locate.
    The inverse function takes a fraction of a unit output space, returning the value to indicate at that position.

    These should be monotonic over their intended range.
    """

    def __init__(self, fn, inv_fn, increasing=True):
        super().__init__(fn, inv_fn)
        self.is_increasing = increasing

    def position_of(self, value):
        return self.fn(value)

    def value_at(self, position):
        return self.inverse(position)

    def value_at_start(self):
        return self.value_at(0)

    def value_at_end(self):
        return self.value_at(1)


LOG_TEN = math.log(10)


class Scalers:
    Base = Scaler(gen_base, pos_base)
    Square = Scaler(scale_square, lambda p: pos_base(p * 2))
    Cube = Scaler(scale_cube, lambda p: pos_base(p * 3))
    Inverse = Scaler(scale_inverse, lambda p: pos_base(1 - p), increasing=False)
    SquareRoot = Scaler(scale_sqrt, lambda p: pos_base(p / 2))
    Log10 = Scaler(scale_log, lambda p: p * TEN)
    Ln = Scaler(lambda x: x / LOG_TEN, lambda p: p * LOG_TEN)
    Sin = Scaler(scale_sin, lambda p: math.asin(p))
    CoSin = Scaler(scale_cos, lambda p: math.asin(p), increasing=False)
    Tan = Scaler(scale_tan, lambda p: math.atan(pos_base(p)))
    SinTan = Scaler(scale_sin_tan, lambda p: math.atan(pos_base(p)))
    CoTan = Scaler(scale_cot, lambda p: math.atan(DEG_RIGHT_ANGLE - p), increasing=False)
    SinH = Scaler(scale_sinh, lambda p: math.asinh(pos_base(p)))
    CosH = Scaler(scale_cosh, lambda p: math.acosh(pos_base(p)))
    TanH = Scaler(scale_tanh, lambda p: math.atanh(pos_base(p)))
    Pythagorean = Scaler(scale_pythagorean, lambda p: math.sqrt(1 - (pos_base(p) / 10) ** 2))
    Chi = Scaler(lambda x: x / PI_HALF, lambda p: p * PI_HALF)
    Theta = Scaler(lambda x: x / DEG_RIGHT_ANGLE, lambda p: p * DEG_RIGHT_ANGLE)
    LogLog = Scaler(scale_log_log, lambda p: math.exp(pos_base(p)))
    LogLogNeg = Scaler(scale_neg_log_log, lambda p: math.exp(pos_base(-p)))
    Hyperbolic = Scaler(scale_hyperbolic, lambda p: math.sqrt(1 + math.pow(pos_base(p), 2)))


class Scale:

    def __init__(self, left_sym: str, right_sym: str, scaler: callable, shift: float = 0,
                 increasing=True, key=None, rule_part=RulePart.STOCK, opp_scale=None):
        self.left_sym = left_sym
        """left scale symbol"""
        self.right_sym = right_sym
        """right scale symbol"""
        self.scaler = scaler
        self.gen_fn = scaler.fn if isinstance(scaler, Scaler) else scaler
        """generating function (producing a fraction of output width)"""
        self.pos_fn = scaler.inverse if isinstance(scaler, Scaler) else None
        """positioning function (takes a proportion of output width, returning what value is there)"""
        self.shift = shift
        """scale shift from left index (as a fraction of output width)"""
        self.increasing = scaler.is_increasing if isinstance(scaler, Scaler) else increasing
        """whether the scale values increase from left to right"""
        self.key = key or left_sym
        """non-unicode name; unused"""  # TODO extend for all alternate namings?
        self.rule_part = rule_part
        """which part of the rule it's on, slide vs stock"""
        self.opp_scale = opp_scale
        """which scale, if on an edge, it's aligned with"""
        if opp_scale:
            opp_scale.opp_scale = self
        self.al = Align.UPPER if opp_scale else Align.LOWER

    def __repr__(self):
        return f'Scale({self.key}, {self.right_sym}, {self.scaler})'

    @property
    def col(self):
        return black_on_white.color_of_scale(self)

    def frac_pos_of(self, x, shift_adj=0):
        """
        Generating Function for the Scales
        :param Number x: the dependent variable
        :param Number shift_adj: how much the scale is shifted, as a fraction of the scale
        :return: float scaled so 0 and 1 are the left and right of the scale
        """
        return self.shift + shift_adj + self.gen_fn(x)

    def value_at_frac_pos(self, frac_pos, shift_adj=0) -> float:
        return self.pos_fn(frac_pos - self.shift - shift_adj)

    def value_at_start(self):
        return self.value_at_frac_pos(0)

    def value_at_end(self):
        return self.value_at_frac_pos(1)

    def value_range(self):
        return self.value_at_start(), self.value_at_end()

    def powers_of_ten_in_range(self):
        start_value, end_value = self.value_range()
        start_log = math.log10(start_value)
        end_log = math.log10(end_value)
        low_log = min(start_log, end_log)
        high_log = max(start_log, end_log)
        return range(math.ceil(low_log), math.ceil(high_log))

    def pos_of(self, x, geom) -> int:
        return round(geom.SL * self.frac_pos_of(x))

    def offset_between(self, x_start, x_end, scale_width):
        return abs(self.frac_pos_of(x_end) - self.frac_pos_of(x_start)) * scale_width

    def smallest_diff_size_for_delta(self, x_start, x_end, x_delta, scale_width=1):
        return min(
            self.offset_between(x_start, x_start + x_delta, scale_width=scale_width),
            self.offset_between(x_end - x_delta, x_end, scale_width=scale_width)
        )

    def scale_to(self, x, scale_width, shift_adj=0):
        """
        Generating Function for the Scales
        :param Number x: the dependent variable
        :param Number shift_adj: how much the scale is shifted, as a fraction of the scale
        :param int scale_width: number of pixels of scale width
        :return: int number of pixels across to the result position
        """
        return round(scale_width * self.frac_pos_of(x, shift_adj=shift_adj))

    def grad_pat_divided(self, r, geom, y_off, al, dividers, start_value=None, end_value=None):
        if dividers is None:
            dividers = [10**n for n in self.powers_of_ten_in_range()]
        if dividers:
            grad_pat(r, geom, y_off, self, al, start_value=start_value, end_value=dividers[0])
            last_i = len(dividers) - 1
            for i, di in enumerate(dividers):
                dj = dividers[i + 1] if i < last_i else end_value
                grad_pat(r, geom, y_off, self, al, start_value=di, end_value=dj)
        else:
            grad_pat(r, geom, y_off, self, al, start_value=start_value, end_value=end_value)

class Scales:
    A = Scale('A', 'x²', Scalers.Square)
    B = Scale('B', 'x²_y', Scalers.Square, rule_part=RulePart.SLIDE, opp_scale=A)
    C = Scale('C', 'x_y', Scalers.Base, rule_part=RulePart.SLIDE)
    CF = Scale('CF', 'πx_y', Scalers.Base, shift=pi_fold_shift, rule_part=RulePart.SLIDE)
    CI = Scale('CI', '1/x_y', Scalers.Inverse, increasing=False, rule_part=RulePart.SLIDE)
    CIF = Scale('CIF', '1/πx_y', Scalers.Inverse, shift=pi_fold_shift - 1, increasing=False, rule_part=RulePart.SLIDE)
    D = Scale('D', 'x', Scalers.Base, opp_scale=C)
    DF = Scale('DF', 'πx', Scalers.Base, shift=pi_fold_shift, opp_scale=CF)
    DI = Scale('DI', '1/x', Scalers.Inverse, increasing=False)
    K = Scale('K', 'x³', Scalers.Cube)
    L = Scale('L', 'log x', Scalers.Log10)
    Ln = Scale('Ln', 'ln x', Scalers.Ln)
    LL0 = Scale('LL₀', 'e^0.001x', Scalers.LogLog, shift=3)
    LL1 = Scale('LL₁', 'e^0.01x', Scalers.LogLog, shift=2)
    LL2 = Scale('LL₂', 'e^0.1x', Scalers.LogLog, shift=1)
    LL3 = Scale('LL₃', 'e^x', Scalers.LogLog)
    LL00 = Scale('LL₀₀', 'e^-0.001x', Scalers.LogLogNeg, increasing=False, shift=3)
    LL01 = Scale('LL₀₁', 'e^-0.01x', Scalers.LogLogNeg, increasing=False, shift=2)
    LL02 = Scale('LL₀₂', 'e^-0.1x', Scalers.LogLogNeg, increasing=False, shift=1)
    LL03 = Scale('LL₀₃', 'e^-x', Scalers.LogLogNeg, increasing=False)
    P = Scale('P', '√1-x²', Scalers.Pythagorean, key='P', increasing=False)
    R1 = Scale('R₁', '√x', Scalers.SquareRoot, key='R1')
    R2 = Scale('R₂', '√10x', Scalers.SquareRoot, key='R2', shift=-1)
    S = Scale('S', 'sin x°', Scalers.Sin)
    CoS = Scale('C', 'cos x°', Scalers.CoSin, increasing=False)
    SRT = Scale('SRT', 'tan 0.01x', scale_sin_tan_radians)
    ST = Scale('ST', 'tan 0.01x°', Scalers.SinTan)
    T = Scale('T', 'tan x°', Scalers.Tan)
    CoT = Scale('T', 'cot x°', Scalers.CoTan, increasing=False)
    T1 = Scale('T₁', 'tan x°', Scalers.Tan, key='T1')
    T2 = Scale('T₂', 'tan 0.1x°', Scalers.Tan, key='T2', shift=-1)
    W1 = Scale('W₁', '√x', Scalers.SquareRoot, key='W1')
    W2 = Scale('W₂', '√10x', Scalers.SquareRoot, key='W2', shift=-1)
    W1Prime = Scale("W₁", '√x', Scalers.SquareRoot, key='W1Prime', opp_scale=W1)
    W2Prime = Scale("W₂", '√10x', Scalers.SquareRoot, key='W2Prime', shift=-1, opp_scale=W2)

    H1 = Scale('H₁', '√1+0.1x²', Scalers.Hyperbolic, key='H1', shift=1)
    H2 = Scale('H₂', '√1+x²', Scalers.Hyperbolic, key='H2')
    Sh1 = Scale('Sh₁', 'sinh x', Scalers.SinH, key='Sh1', shift=1)
    Sh2 = Scale('Sh₂', 'sinh x', Scalers.SinH, key='Sh2')
    Ch1 = Scale('Ch', 'cosh x', Scalers.CosH)
    Th = Scale('Th', 'tanh x', Scalers.TanH, shift=1)

    # EE-specific
    # Hemmi 153:
    Chi = Scale('χ', '', Scalers.Chi)
    Theta = Scale('θ', '°', Scalers.Theta, key='Theta')
    # Pickett N-515-T:
    f_x = Scale('f_x', 'x/2π', Scalers.Base, shift=gen_base(TAU))
    L_r = Scale('(L_r) H', '1/(2πx)²', Scaler(
        lambda x: gen_base(TAU * x) / -2,
        lambda p: 1/(TAU * pos_base(p))**2
    ), increasing=False)


# TODO scales from Aristo 965 Commerz II: KZ, %, Z/ZZ1/ZZ2/ZZ3 compound interest
# TODO meta-scale showing % with 100% over 1/unity
#  special marks being 0,5,10,15,20,25,30,33⅓,40,50,75,100 in both directions


SCALE_NAMES = ['A', 'B', 'C', 'D',
               'K', 'R1', 'R2', 'CI',
               'DI', 'CF', 'DF', 'CIF', 'L',
               'S', 'T', 'ST',
               'Ln', 'T1', 'T2', 'P',
               'LL0', 'LL1', 'LL2', 'LL3',
               'LL00', 'LL01', 'LL02', 'LL03',
               'W1', 'W1Prime', 'W2', 'W2Prime',
               'H1', 'H2',
               'Sh1', 'Sh2', 'Th',
               'Chi', 'Theta', 'f_x', 'L_r']


class Layout:
    def __init__(self, front_layout: str, rear_layout: str = None):
        if not rear_layout and '\n' in front_layout:
            (front_layout, rear_layout) = front_layout.splitlines()
        self.front_layout = self.parse_side_layout(front_layout)
        self.rear_layout = self.parse_side_layout(rear_layout)
        self.check_scales()

    def __repr__(self):
        return f'Layout({self.front_layout}, {self.rear_layout})'

    @classmethod
    def parse_segment_layout(cls, segment_layout: str) -> [str]:
        if segment_layout:
            return re.split(r'[, ]+', segment_layout.strip(' '))
        else:
            return None

    @classmethod
    def parts_of_side_layout(cls, side_layout: str) -> [str]:
        if '/' in side_layout:
            return side_layout.split('/')
        parts = re.fullmatch(r'\|?\s*(.+)\[(.+)](.*)\s*\|?', side_layout)
        if parts:
            return [parts.group(1), parts.group(2), parts.group(3)]
        else:
            return [side_layout, '', '']

    @classmethod
    def parse_side_layout(cls, layout):
        """
        :param str layout:
        :return: [[str], [str], [str]]
        """
        upper_frame_scales = None
        lower_frame_scales = None
        slide_scales = None
        if layout:
            major_parts = [cls.parse_segment_layout(x) for x in cls.parts_of_side_layout(layout.strip(' |'))]
            num_parts = len(major_parts)
            if num_parts == 1:
                (slide_scales) = major_parts
            elif num_parts == 3:
                (upper_frame_scales, slide_scales, lower_frame_scales) = major_parts
        return [upper_frame_scales, slide_scales, lower_frame_scales]

    def check_scales(self):
        for front_part in self.front_layout:
            if not front_part:
                continue
            for scale_name in front_part:
                if scale_name not in SCALE_NAMES:
                    raise Exception(f'Unrecognized front scale name: {scale_name}')
        for rear_part in self.front_layout:
            if not rear_part:
                continue
            for scale_name in rear_part:
                if scale_name not in SCALE_NAMES:
                    raise Exception(f'Unrecognized rear scale name: {scale_name}')


class Layouts:
    MannheimOriginal = Layout('A/B C/D')
    RegleDesEcoles = Layout('DF/CF C/D')
    Mannheim = Layout('A/B CI C/D K', 'S L T')
    Rietz = Layout('K A/B CI C/D L', 'S ST T')
    Darmstadt = Layout('S T A/B K CI C/D P', 'L LL1 LL2 LL3')

    # BOGELEX 1000 -- LEFT HANDED LIMACON 2020 -- KWENA & TOOR CO.S
    Demo = Layout('|  K,  A  [ B, T, ST, S ] D,  DI    |',
                  '|  L,  DF [ CF,CIF,CI,C ] D, R1, R2 |')


class Model:
    def __init__(self, brand: str, model_name: str, layout: Layout):
        self.brand = brand
        self.name = model_name
        self.layout = layout

    def __repr__(self):
        return f'Model({self.brand}, {self.name}, {self.layout})'


class Models:
    Demo = Model('BOGELEX 1000', 'LEFT HANDED LIMACON 2020', Layouts.Demo)
    Aristo868 = Model('Aristo', '868', Layout(
        'ST T1 T2 DF/CF CIF CI C/D P S',
        'LL01 LL02 LL03 A/B L K C/D LL3 LL2 LL1'
    ))
    PickettN515T = Model('Pickett', 'N-515-T', Layout(
        'L_r f_x A/B S T CI C/D L Ln'
    ))
    FaberCastell283 = Model('Faber-Castell', '2/83', Layout(
        'K T1 T2 DF/CF CIF CI C/D S ST P',
        'LL03 LL02 LL01 W2/W2Prime L C W1Prime/W1 LL1 LL2 LL3'
    ))
    FaberCastell283N = Model('Faber-Castell', '2/83', Layout(
        'K T1 T2 DF/CF CIF CI C/D S ST P',
        'LL03 LL02 LL01 LL00 W2/W2Prime L C W1Prime/W1 LL0 LL1 LL2 LL3'
    ))


class GaugeMark:
    def __init__(self, sym, value, comment=None):
        self.sym = sym
        self.value = value
        self.comment = comment

    def draw(self, r, geom, y_off, sc, font_size, al, col=FG, shift_adj=0):
        """
        :param ImageDraw.Draw r:
        :param Geometry geom:
        :param int y_off: y pos
        :param Scale sc:
        :param FontSize font_size: font size
        :param Align al: alignment
        :param str col: color
        :param Number shift_adj:
        """
        x = sc.scale_to(self.value, geom.SL, shift_adj=shift_adj)
        draw_tick(r, geom, col, y_off, x, geom.tick_h(HMod.MED), al)
        sym_h = geom.tick_h(HMod.XL if al == Align.LOWER else HMod.MED)
        draw_symbol(r, geom, col, y_off, self.sym, x, sym_h, font_size, FontStyle.REG, al)


class Marks:
    e = GaugeMark('e', math.e, comment='base of natural logarithms')
    inv_e = GaugeMark('1/e', 1 / math.e, comment='base of natural logarithms')
    tau = GaugeMark('τ', TAU, comment='ratio of circle circumference to radius')
    pi = GaugeMark('π', PI, comment='ratio of circle circumference to diameter')
    pi_half = GaugeMark('π/2', PI_HALF, comment='ratio of quarter arc length to radius')
    inv_pi = GaugeMark('M', 1 / PI, comment='reciprocal of π')

    deg_per_rad = GaugeMark('r', DEG_FULL / TAU / TEN, comment='degrees per radian')
    rad_per_deg = GaugeMark('ρ', TAU / DEG_FULL, comment='radians per degree')
    rad_per_min = GaugeMark('ρ′', TAU / DEG_FULL * 60, comment='radians per minute')
    rad_per_sec = GaugeMark('ρ″', TAU / DEG_FULL * 60 * 60, comment='radians per second')

    ln_over_log10 = GaugeMark('L', 1 / math.log10(math.e), comment='ratio of natural log to log base 10')

    sqrt_ten = GaugeMark('√10', math.sqrt(TEN), comment='square root of 10')
    cube_root_ten = GaugeMark('c', math.pow(TEN, 1 / 3), comment='cube root of 10')

    hp_per_kw = GaugeMark('N', 1.341022, comment='mechanical horsepower per kW')


def gen_scale(r, geom, y_off, sc, al, overhang=0.02):
    """
    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param int y_off: y pos
    :param Scale sc:
    :param Align al: alignment
    :param float overhang: fraction of total width to overhang each side to label
    """

    # Place Index Symbols (Left and Right)
    fs_lbl = FontSize.ScaleLBL
    fs_lgn = FontSize.NumLG
    fs_mdn = FontSize.NumMD
    fs_smn = FontSize.NumSM
    reg = FontStyle.REG
    italic = FontStyle.ITALIC
    font = font_for_family(reg, fs_lbl)

    if DEBUG:
        r.rectangle((geom.li, y_off, geom.li + geom.SL, y_off + geom.SH), outline='grey')

    # Right
    (right_sym, _, _) = symbol_parts(sc.right_sym)
    w2, h2 = get_font_size(right_sym, font)
    y2 = (geom.SH - h2) / 2
    x_right = (1 + overhang) * geom.SL + w2 / 2
    sym_col = black_on_white.color_of_scale(sc)
    draw_symbol(r, geom, sym_col, y_off, sc.right_sym, x_right, y2, fs_lbl, reg, al)

    # Left
    (left_sym, _, _) = symbol_parts(sc.left_sym)
    w1, h1 = get_font_size(left_sym, font)
    y1 = (geom.SH - h1) / 2
    x_left = (0 - overhang) * geom.SL - w1 / 2
    draw_symbol(r, geom, sym_col, y_off, sc.left_sym, x_left, y1, fs_lbl, reg, al)

    # Special Symbols for S, and T
    sc_alt = None
    if sc == Scales.S:
        sc_alt = Scales.CoS
    elif sc == Scales.T:
        sc_alt = Scales.CoT

    if sc_alt:
        draw_symbol(r, geom, sc_alt.col, y_off, sc_alt.left_sym, x_left - get_width('__', fs_lbl, reg),
                    y2, fs_lbl, reg, al)
        draw_symbol(r, geom, sc_alt.col, y_off, sc_alt.right_sym, x_right, y2 - h2 * 0.8, fs_lbl, reg, al)
    elif sc == Scales.ST:
        draw_symbol(r, geom, sym_col, y_off, 'sin 0.01x°', x_right, y2 - h2 * 0.8, fs_lbl, reg, al)

    full_range = i_range_tenths(1, 10)

    med_h = geom.tick_h(HMod.MED)

    is_cd = sc.scaler == Scalers.Base and sc.shift == 0  # C/D

    # Tick Placement (the bulk!)
    if is_cd or (sc.scaler == Scalers.Inverse and sc.shift == 0):

        # Ticks
        pat(r, geom, y_off, sc, HMod.MED, full_range, (0, 100), None, al)
        pat(r, geom, y_off, sc, HMod.XL, full_range, (50, 100), (150, 1000), al)
        pat(r, geom, y_off, sc, HMod.SM, full_range, (0, 10), (150, 100), al)
        range_1to2 = i_range_tenths(1, 2, False)
        pat(r, geom, y_off, sc, HMod.SM, range_1to2, (5, 10), None, al)
        pat(r, geom, y_off, sc, HMod.XS, range_1to2, (0, 1), (0, 5), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range_tenths(2, 4, False), (0, 2), (0, 10), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range_tenths(4, 10), (0, 5), (0, 10), al)

        # 1-10 Labels
        for x in range(1, 11):
            draw_symbol(r, geom, sym_col, y_off, leading_digit_of(x), sc.pos_of(x, geom), med_h, fs_lbl, reg, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            draw_numeral(r, geom, sym_col, y_off, last_digit_of(x), sc.pos_of(x / 10, geom), geom.tick_h(HMod.SM), fs_lgn, reg, al)

        # Gauge Points
        Marks.pi.draw(r, geom, y_off, sc, fs_lbl, al, col=sym_col)

        if y_off < geom.side_height + geom.oY:
            Marks.deg_per_rad.draw(r, geom, y_off, sc, fs_lbl, al, col=sym_col)
            Marks.tau.draw(r, geom, y_off, sc, fs_lbl, al, col=sym_col)

    elif sc.scaler == Scalers.Square:

        # Ticks
        pat(r, geom, y_off, sc, HMod.MED, full_range, (0, 100), None, al)
        pat(r, geom, y_off, sc, HMod.MED, i_range(1000, 10001, True), (0, 1000), None, al)
        pat(r, geom, y_off, sc, HMod.SM, i_range_tenths(1, 5), (0, 10), (50, 100), al)
        pat(r, geom, y_off, sc, HMod.SM, i_range(1000, 5001, True), (0, 100), (500, 1000), al)
        pat(r, geom, y_off, sc, HMod.XL, full_range, (50, 100), None, al)
        pat(r, geom, y_off, sc, HMod.XL, i_range(1000, 10001, True), (500, 1000), None, al)
        pat(r, geom, y_off, sc, HMod.XS, i_range_tenths(1, 2), (0, 2), None, al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(1000, 2000, True), (0, 20), None, al)
        pat(r, geom, y_off, sc, HMod.XS, i_range_tenths(2, 5, False), (5, 10), None, al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(2000, 5000, True), (50, 100), None, al)
        pat(r, geom, y_off, sc, HMod.XS, i_range_tenths(5, 10), (0, 10), (0, 50), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(5000, 10001, True), (0, 100), (0, 500), al)

        # 1-10 Labels
        for x in range(1, 11):
            sym = leading_digit_of(x)
            draw_symbol(r, geom, sym_col, y_off, sym, sc.pos_of(x, geom), med_h, fs_lbl, reg, al)
            draw_symbol(r, geom, sym_col, y_off, sym, sc.pos_of(x * 10, geom), med_h, fs_lbl, reg, al)

        # Gauge Points
        Marks.pi.draw(r, geom, y_off, sc, fs_lbl, al)
        Marks.pi.draw(r, geom, y_off, sc, fs_lbl, al, shift_adj=0.5)

    elif sc == Scales.K:
        # Ticks per power of 10
        for b in [10 ** foo for foo in range(0, 3)]:
            pat(r, geom, y_off, sc, HMod.MED, i_range_tenths(1 * b, 10 * b), (0, 100 * b), None, al)
            pat(r, geom, y_off, sc, HMod.XL, i_range_tenths(1 * b, 6 * b), (50 * b, 100 * b), None, al)
            pat(r, geom, y_off, sc, HMod.SM, i_range_tenths(1 * b, 3 * b), (0, 10 * b), None, al)
            pat(r, geom, y_off, sc, HMod.XS, i_range_tenths(1 * b, 3 * b), (5 * b, 10 * b), None, al)
            pat(r, geom, y_off, sc, HMod.XS, i_range_tenths(3 * b, 6 * b), (0, 10 * b), None, al)
            pat(r, geom, y_off, sc, HMod.XS, i_range_tenths(6 * b, 10 * b), (0, 20 * b), None, al)

        # 1-10 Labels
        f = FontSize.NumXL
        for x in range(1, 11):
            sym = leading_digit_of(x)
            draw_symbol(r, geom, sym_col, y_off, sym, sc.pos_of(x, geom), med_h, f, reg, al)
            draw_symbol(r, geom, sym_col, y_off, sym, sc.pos_of(x * 10, geom), med_h, f, reg, al)
            draw_symbol(r, geom, sym_col, y_off, sym, sc.pos_of(x * 100, geom), med_h, f, reg, al)

    elif sc == Scales.R1:

        # Ticks
        sf = 1000
        pat(r, geom, y_off, sc, HMod.MED, i_range(1000, 3200, True), (0, 100), None, al, sf=sf)
        pat(r, geom, y_off, sc, HMod.XL, i_range(1000, 2000, True), (0, 50), (0, 100), al, sf=sf)
        pat(r, geom, y_off, sc, HMod.SM, i_range(2000, 3200, True), (0, 50), None, al, sf=sf)
        pat(r, geom, y_off, sc, HMod.SM, i_range(1000, 2000, True), (0, 10), (0, 50), al, sf=sf)
        pat(r, geom, y_off, sc, HMod.XS, i_range(1000, 2000, True), (5, 10), None, al, sf=sf)
        pat(r, geom, y_off, sc, HMod.XS, i_range(2000, 3180, True), (0, 10), (0, 50), al, sf=sf)

        # 1-10 Labels
        for x in range(1, 4):
            draw_numeral(r, geom, sym_col, y_off, x, sc.pos_of(x, geom), med_h, fs_lbl, reg, al)

        # 0.1-3.1 Labels
        for x in range(11, 20):
            draw_numeral(r, geom, sym_col, y_off, last_digit_of(x), sc.pos_of(x / 10, geom), med_h, fs_lgn, reg, al)
        for x in range(21, 30):
            draw_numeral(r, geom, sym_col, y_off, last_digit_of(x), sc.pos_of(x / 10, geom), med_h, fs_lgn, reg, al)
        draw_numeral(r, geom, sym_col, y_off, last_digit_of(31), sc.pos_of(31 / 10, geom), med_h, fs_lgn, reg, al)
        # Marks.sqrt_ten.draw(r, geom, y_off, sc, fs_lgn, al, sym_col)

    elif sc in {Scales.W1, Scales.W1Prime}:
        sc.grad_pat_divided(r, geom, y_off, al, [2])
        Marks.sqrt_ten.draw(r, geom, y_off, sc, fs_lgn, al, sym_col)

    elif sc in {Scales.W2, Scales.W2Prime}:
        sc.grad_pat_divided(r, geom, y_off, al, [5])
        Marks.sqrt_ten.draw(r, geom, y_off, sc, fs_lgn, al, sym_col)

    elif sc == Scales.R2:

        # Ticks
        sf = 1000
        pat(r, geom, y_off, sc, HMod.MED, i_range(4000, 10000, True), (0, 1000), None, al, sf=sf)
        pat(r, geom, y_off, sc, HMod.XL, i_range(5000, 10000, False), (500, 1000), None, al, sf=sf)
        pat(r, geom, y_off, sc, HMod.SM, i_range(3200, 10000, False), (0, 100), (0, 1000), al, sf=sf)
        pat(r, geom, y_off, sc, HMod.SM, i_range(3200, 5000, False), (0, 50), None, al, sf=sf)
        pat(r, geom, y_off, sc, HMod.XS, i_range(3160, 5000, False), (0, 10), (0, 50), al, sf=sf)
        pat(r, geom, y_off, sc, HMod.XS, i_range(5000, 10000, False), (0, 20), (0, 100), al, sf=sf)

        # Marks.sqrt_ten.draw(r, geom, y_off, sc, fs_lgn, al, sym_col)
        # 4-10 Labels
        for x in range(4, 10):
            draw_numeral(r, geom, sym_col, y_off, x, sc.pos_of(x, geom), med_h, fs_lbl, reg, al)
        draw_symbol(r, geom, sym_col, y_off, '1', geom.SL, med_h, fs_lbl, reg, al)

        # 3.1-4.9 Labels
        for x in range(32, 40):
            draw_numeral(r, geom, sym_col, y_off, last_digit_of(x), sc.pos_of(x / 10, geom), med_h, fs_lgn, reg, al)
        for x in range(41, 50):
            draw_numeral(r, geom, sym_col, y_off, last_digit_of(x), sc.pos_of(x / 10, geom), med_h, fs_lgn, reg, al)

    elif sc == Scales.H1:
        draw_numeral(r, geom, sym_col, y_off, 1.005, sc.pos_of(1.005, geom), geom.tick_h(HMod.XL), fs_lgn, reg, al)
        sc.grad_pat_divided(r, geom, y_off, al, [1.03, 1.1])

    elif sc == Scales.H2:
        draw_numeral(r, geom, sym_col, y_off, 1.5, sc.pos_of(1.5, geom), geom.tick_h(HMod.XL), fs_lgn, reg, al)
        sc.grad_pat_divided(r, geom, y_off, al, [4])

    elif sc.scaler == Scalers.Base and sc.shift == pi_fold_shift:  # CF/DF

        # Ticks
        pat(r, geom, y_off, sc, HMod.MED, i_range_tenths(1, 3), (0, 100), None, al)
        pat(r, geom, y_off, sc, HMod.MED, i_range_tenths(4, 10), (0, 100), None, al, shift_adj=-1)
        pat(r, geom, y_off, sc, HMod.XL, i_range_tenths(2, 3), (50, 100), None, al)
        pat(r, geom, y_off, sc, HMod.SM, i_range_tenths(1, 2), (0, 5), None, al)
        pat(r, geom, y_off, sc, HMod.SM, i_range(200, 310, True), (0, 10), None, al)
        pat(r, geom, y_off, sc, HMod.XL, i_range(320, 1000, True), (50, 100), None, al, shift_adj=-1)
        pat(r, geom, y_off, sc, HMod.SM, i_range(320, 1000, True), (0, 10), (150, 100), al, shift_adj=-1)
        pat(r, geom, y_off, sc, HMod.XS, i_range(100, 200, True), (0, 1), (0, 5), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(200, 314, False), (0, 2), (0, 10), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(316, 400, True), (0, 2), (0, 10), al, shift_adj=-1)
        pat(r, geom, y_off, sc, HMod.XS, i_range(400, 1000, True), (0, 5), (0, 10), al, shift_adj=-1)

        # 1-10 Labels
        for x in range(1, 4):
            draw_numeral(r, geom, sym_col, y_off, x, sc.pos_of(x, geom), med_h, fs_lbl, reg, al)
        for x in range(4, 10):
            draw_numeral(r, geom, sym_col, y_off, x, sc.pos_of(x, geom) - geom.SL, med_h, fs_lbl, reg, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            draw_numeral(r, geom, sym_col, y_off, last_digit_of(x), sc.pos_of(x / 10, geom), geom.tick_h(HMod.SM),
                         fs_lgn, reg, al)

        # Gauge Points
        Marks.pi.draw(r, geom, y_off, sc, fs_lbl, al)
        Marks.pi.draw(r, geom, y_off, sc, fs_lbl, al, shift_adj=-1)

    elif sc == Scales.CIF:

        # Ticks
        pat(r, geom, y_off, sc, HMod.MED, i_range(100, 300, True), (0, 100), None, al)
        pat(r, geom, y_off, sc, HMod.MED, i_range(400, 1000, True), (0, 100), None, al, shift_adj=1)

        pat(r, geom, y_off, sc, HMod.XL, i_range(200, 300, True), (50, 100), None, al)
        pat(r, geom, y_off, sc, HMod.SM, i_range(100, 200, True), (0, 5), None, al)
        pat(r, geom, y_off, sc, HMod.SM, i_range(200, 320, True), (0, 10), None, al)
        pat(r, geom, y_off, sc, HMod.XL, i_range(320, 1000, True), (50, 100), None, al, shift_adj=1)
        pat(r, geom, y_off, sc, HMod.SM, i_range(310, 1000, True), (0, 10), (150, 100), al, shift_adj=1)
        pat(r, geom, y_off, sc, HMod.XS, i_range(100, 200, True), (0, 1), (0, 5), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(200, 320, True), (0, 2), (0, 10), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(310, 400, True), (0, 2), (0, 10), al, shift_adj=1)
        pat(r, geom, y_off, sc, HMod.XS, i_range(400, 1000, True), (0, 5), (0, 10), al, shift_adj=1)

        # 1-10 Labels
        for x in range(4, 10):
            draw_numeral(r, geom, RED, y_off, x, sc.pos_of(x, geom) + geom.SL, med_h, fs_lbl, reg, al)
        for x in range(1, 4):
            draw_numeral(r, geom, RED, y_off, x, sc.pos_of(x, geom), med_h, fs_lbl, reg, al)

        # 0.1-0.9 Labels
        small_h = geom.tick_h(HMod.SM)
        for x in range(11, 20):
            draw_numeral(r, geom, RED, y_off, last_digit_of(x), sc.pos_of(x / 10, geom), small_h, fs_lgn, reg, al)

    elif sc == Scales.L:

        # Ticks
        range1 = i_range(0, 1000, True)
        range2 = i_range(1, 1000, True)
        pat(r, geom, y_off, sc, HMod.MED, range1, (0, 10), (50, 50), al)
        pat(r, geom, y_off, sc, HMod.XL, range2, (50, 100), None, al)
        pat(r, geom, y_off, sc, HMod.LG, range1, (0, 100), None, al)
        pat(r, geom, y_off, sc, HMod.XS, range2, (0, 2), (0, 50), al)

        # Labels
        for x in range(0, 11):
            if x == 0:
                draw_numeral(r, geom, sym_col, y_off, x, sc.pos_of(x, geom), med_h, fs_lbl, reg, al)
            if x == 10:
                draw_numeral(r, geom, sym_col, y_off, 1, sc.pos_of(x, geom), med_h, fs_lbl, reg, al)
            elif x in range(1, 10):
                draw_numeral(r, geom, sym_col, y_off, x / 10, sc.pos_of(x, geom), med_h, fs_lbl, reg, al)

    elif sc == Scales.Ln:
        grad_pat(r, geom, y_off, sc, al)

    elif sc.scaler == Scalers.Sin:

        # Ticks
        pat(r, geom, y_off, sc, HMod.XL, i_range(1000, 7000, True), (0, 1000), None, al)
        pat(r, geom, y_off, sc, HMod.MED, i_range(7000, 10000, True), (0, 1000), None, al)
        pat(r, geom, y_off, sc, HMod.XL, i_range(600, 2000, True), (0, 100), None, al)
        pat(r, geom, y_off, sc, HMod.SM, i_range(600, 2000, False), (50, 100), (0, 100), al)
        pat(r, geom, y_off, sc, HMod.XL, i_range(2000, 6000, False), (500, 1000), (0, 1000), al)
        pat(r, geom, y_off, sc, HMod.SM, i_range(2000, 6000, False), (0, 100), (0, 500), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(570, 2000, False), (0, 10), (0, 50), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(2000, 3000, False), (0, 20), (0, 100), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(3000, 6000, False), (0, 50), (0, 100), al)
        pat(r, geom, y_off, sc, HMod.SM, i_range(6000, 8500, True), (500, 1000), None, al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(6000, 8000, False), (0, 100), None, al)

        # Degree Labels

        for x in range(6, 16):
            x_coord = sc.pos_of(x, geom) + 1.2 / 2 * get_width(x, fs_smn, italic)
            draw_numeral(r, geom, sym_col, y_off, x, x_coord, med_h, fs_smn, reg, al)
            xi = angle_opp(x)
            x_coord_opp = sc.pos_of(x, geom) - 1.4 / 2 * get_width(xi, fs_smn, italic)
            draw_numeral(r, geom, RED, y_off, xi, x_coord_opp,
                         med_h, fs_lgn, italic, al)

        for x in range(16, 20):
            x_coord = sc.pos_of(x, geom) + 1.2 / 2 * get_width(x, fs_mdn, italic)
            draw_numeral(r, geom, sym_col, y_off, x, x_coord, med_h, fs_mdn, reg, al)

        for x in range(20, 71, 5):
            if (x % 5 == 0 and x < 40) or last_digit_of(x) == 0:
                x_coord = sc.pos_of(x, geom) + 1.2 / 2 * get_width(x, fs_mdn, italic)
                draw_numeral(r, geom, sym_col, y_off, x, x_coord, med_h, fs_mdn, reg, al)
                if x != 20:
                    xi = angle_opp(x)
                    x_coord = sc.pos_of(x, geom) - 1.4 / 2 * get_width(xi, fs_mdn, italic)
                    if xi != 40:
                        draw_numeral(r, geom, RED, y_off, xi, x_coord, med_h, fs_mdn, italic, al)
                    elif xi == 40:
                        draw_numeral(r, geom, RED, y_off + 11, 40, x_coord, med_h, fs_mdn, italic, al)

        draw_numeral(r, geom, sym_col, y_off, DEG_RIGHT_ANGLE, geom.SL, med_h, fs_lgn, reg, al)

    elif sc == Scales.T or sc == Scales.T1:

        # Ticks
        pat(r, geom, y_off, sc, HMod.XL, i_range(600, 2500, True), (0, 100), None, al)
        pat(r, geom, y_off, sc, HMod.XL, i_range(600, 1000, True), (50, 100), None, al)
        pat(r, geom, y_off, sc, HMod.XL, i_range(2500, 4500, True), (0, 500), None, al)
        pat(r, geom, y_off, sc, HMod.MED, i_range(2500, 4500, True), (0, 100), None, al)
        draw_tick(r, geom, sym_col, y_off, geom.SL, round(med_h), al)
        pat(r, geom, y_off, sc, HMod.MED, i_range(600, 950, True), (50, 100), None, al)
        pat(r, geom, y_off, sc, HMod.SM, i_range(570, 1000, True), (0, 10), (0, 50), al)
        pat(r, geom, y_off, sc, HMod.SM, i_range(1000, 2500, False), (50, 100), None, al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(570, 1000, True), (5, 10), (0, 10), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(1000, 2500, False), (0, 10), (0, 50), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(2500, 4500, True), (0, 20), (0, 100), al)

        # Degree Labels
        f = geom.tick_h(HMod.LG)
        opp_col = RED
        for x in range(6, 16):
            x_coord = sc.pos_of(x, geom) + 1.2 / 2 * get_width(x, fs_smn, italic)
            draw_numeral(r, geom, sym_col, y_off, x, x_coord, f, fs_smn, reg, al)
            xi = angle_opp(x)
            x_coord_opp = sc.pos_of(x, geom) - 1.4 / 2 * get_width(xi, fs_smn, italic)
            draw_numeral(r, geom, opp_col, y_off, xi, x_coord_opp, f, fs_smn, italic, al)

        for x in range(16, 21):
            x_coord = sc.pos_of(x, geom) + 1.2 / 2 * get_width(x, fs_mdn, italic)
            draw_numeral(r, geom, sym_col, y_off, x, x_coord, f, fs_mdn, reg, al)

        for x in range(25, 41, 5):
            if x % 5 == 0:
                x_coord = sc.pos_of(x, geom) + 1.2 / 2 * get_width(x, fs_mdn, italic)
                draw_numeral(r, geom, sym_col, y_off, x, x_coord, f, fs_mdn,
                             reg, al)
                xi = angle_opp(x)
                x_coord_opp = sc.pos_of(x, geom) - 1.4 / 2 * get_width(xi, fs_mdn, italic)
                draw_numeral(r, geom, opp_col, y_off, xi, x_coord_opp,
                             f, fs_mdn, italic, al)

        draw_numeral(r, geom, sym_col, y_off, 45, geom.SL, f, fs_lgn, reg, al)

    elif sc == Scales.T2:
        f = geom.tick_h(HMod.LG)
        # Ticks
        fp1 = 4500
        fp2 = 7500
        fpe = 8450
        pat(r, geom, y_off, sc, HMod.MED, range(fp1, fpe, True), (0, 100), None, al)
        pat(r, geom, y_off, sc, HMod.XL, range(fp1, fpe, True), (50, 100), None, al)
        pat(r, geom, y_off, sc, HMod.DOT, range(fp1, fp2, True), (0, 10), (0, 50), al)
        pat(r, geom, y_off, sc, HMod.XS, range(fp2, fpe, True), (0, 10), (0, 50), al)
        pat(r, geom, y_off, sc, HMod.DOT, range(fp2, fpe, True), (0, 5), (0, 10), al)
        # Degree Labels
        for x in range(45, 85, 5):
            draw_numeral(r, geom, sym_col, y_off, x, sc.pos_of(x, geom), f, fs_lgn, reg, al)

    elif sc == Scales.ST:

        # Ticks
        pat(r, geom, y_off, sc, HMod.MED, i_range(100, 550, True), (0, 50), None, al)
        pat(r, geom, y_off, sc, HMod.LG2, i_range(60, 100, False), (0, 10), None, al)
        pat(r, geom, y_off, sc, HMod.XL, i_range(60, 100, False), (5, 10), None, al)
        pat(r, geom, y_off, sc, HMod.MED, i_range(100, 200, False), (0, 10), (0, 50), al)
        pat(r, geom, y_off, sc, HMod.SM, i_range(200, 590, False), (0, 10), None, al)
        pat(r, geom, y_off, sc, HMod.SM, i_range(57, 100, False), (0, 1), None, al)
        pat(r, geom, y_off, sc, HMod.SM, i_range(100, 200, False), (0, 5), None, al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(100, 200, False), (0, 1), (0, 5), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(200, 400, False), (0, 2), (0, 10), al)
        pat(r, geom, y_off, sc, HMod.XS, i_range(400, 585, False), (5, 10), None, al)

        for x in range(570, 1000):
            if x % 5 == 0 and last_digit_of(x) != 0:
                draw_tick(r, geom, sym_col, y_off, sc.pos_of(x / 1000, geom), geom.tick_h(HMod.XS), al)

        # Degree Labels
        draw_symbol(r, geom, sym_col, y_off, '1°', sc.pos_of(1, geom), med_h, fs_lbl, reg, al)
        for x in range(6, 10):
            x_value = x / 10
            draw_numeral(r, geom, sym_col, y_off, x_value, sc.pos_of(x_value, geom), med_h, fs_lbl, reg, al)
        for x in range(1, 4):
            x_value = x + 0.5
            draw_numeral(r, geom, sym_col, y_off, x_value, sc.pos_of(x_value, geom), med_h, fs_lbl, reg, al)
        for x in range(2, 6):
            draw_numeral(r, geom, sym_col, y_off, x, sc.pos_of(x, geom), med_h, fs_lbl, reg, al)

    elif sc == Scales.P:
        # Labels
        label_h = geom.tick_h(HMod.MED)
        font_s = fs_smn
        for x in [0.995]:
            draw_numeral(r, geom, sym_col, y_off, x, sc.pos_of(x, geom), label_h, font_s, reg, al)
        sc.grad_pat_divided(r, geom, y_off, al, [0.3, 0.7, 0.9, 0.98],
                            start_value=0.1, end_value=.995)

    elif sc == Scales.Sh1:
        sc.grad_pat_divided(r, geom, y_off, al, [0.2, 0.4])

    elif sc == Scales.Sh2:
        grad_pat(r, geom, y_off, sc, al)

    elif sc == Scales.Th:
        sf = 1000
        d2 = 1
        d3 = 2
        de = 3
        sc.grad_pat_divided(r, geom, y_off, al, dividers=[0.2, 0.4], end_value=d2)
        pat(r, geom, y_off, sc, HMod.MED, i_range(d2*sf, de*sf, True), (0, 500), None, al, sf=sf)
        pat(r, geom, y_off, sc, HMod.XS, i_range(d2*sf, d3*sf, True), (0, 100), (0, 500), al, sf=sf)
        pat(r, geom, y_off, sc, HMod.DOT, i_range(d2*sf, d3*sf, True), (0, 50), (0, 100), al, sf=sf)
        pat(r, geom, y_off, sc, HMod.DOT, i_range(d3*sf, de*sf, True), (0, 100), (0, 500), al, sf=sf)
        # Labels
        label_h = geom.tick_h(HMod.MED)
        for x in [1, 1.5, 2, 3]:
            draw_numeral(r, geom, sym_col, y_off, x, sc.pos_of(x, geom), label_h, fs_smn, reg, al)

    elif sc == Scales.Chi:
        grad_pat(r, geom, y_off, sc, al)
        Marks.pi_half.draw(r, geom, y_off, sc, fs_lbl, al, sym_col)

    elif sc == Scales.Theta:
        grad_pat(r, geom, y_off, sc, al)

    elif sc == Scales.f_x:
        sc.grad_pat_divided(r, geom, y_off, al, [0.2, 0.5, 1])

    elif sc == Scales.L_r:
        pass  # grad_pat(r, geom, y_off, sc, al)

    elif sc == Scales.LL0:
        sc.grad_pat_divided(r, geom, y_off, al, [1.002, 1.005],
                            start_value=1.00095, end_value=1.0105)

    elif sc == Scales.LL1:
        sc.grad_pat_divided(r, geom, y_off, al, [1.02, 1.05],
                            start_value=1.0095, end_value=1.11)

    elif sc == Scales.LL2:
        sc.grad_pat_divided(r, geom, y_off, al, [1.2, 2],
                            start_value=1.1, end_value=3)
        Marks.e.draw(r, geom, y_off, sc, fs_lgn, al, sym_col)

    elif sc == Scales.LL3:
        sc.grad_pat_divided(r, geom, y_off, al, [10, 50, 100, 1000, 10000],
                            start_value=2.5, end_value=60000)
        Marks.e.draw(r, geom, y_off, sc, fs_lgn, al, sym_col)

    elif sc == Scales.LL03:
        sc.grad_pat_divided(r, geom, y_off, al, [0.001, 0.01, 0.1],
                            start_value=0.0001, end_value=0.39)
        Marks.inv_e.draw(r, geom, y_off, sc, fs_smn, al, sym_col)

    elif sc == Scales.LL02:
        sc.grad_pat_divided(r, geom, y_off, al, [0.75],
                            start_value=0.35, end_value=0.91)
        Marks.inv_e.draw(r, geom, y_off, sc, fs_smn, al, sym_col)

    elif sc == Scales.LL01:
        sc.grad_pat_divided(r, geom, y_off, al, [0.95, 0.98],
                            start_value=0.9, end_value=0.9906)

    elif sc == Scales.LL00:
        sc.grad_pat_divided(r, geom, y_off, al, [0.998],
                            start_value=0.989, end_value=0.9991)

    else:
        sc.grad_pat_divided(r, geom, y_off, al, None)


def leading_digit_of(x: int) -> str:
    return str(x)[0]


def last_digit_of(x: int) -> int:
    return x % TEN


# ----------------------4. Line Drawing Functions----------------------------

# These functions are unfortunately difficult to modify,
# since I built them with specific numbers rather than variables


def draw_borders(r, geom, y0, side, color=RGB_BLACK):
    """
    Place initial borders around scales
    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param y0: vertical offset
    :param Side side:
    :param string|tuple color:
    """

    # Main Frame
    y_offsets = [0, 479, 1119, 1598]

    total_width = geom.total_width
    side_height = geom.side_height
    for horizontal_y in [y0 + y_off for y_off in y_offsets]:
        r.rectangle((geom.oX, horizontal_y, total_width - geom.oX, horizontal_y + 2), outline=color)
    for vertical_x in [geom.oX, total_width - geom.oX]:
        r.rectangle((vertical_x, y0, vertical_x + 2, y0 + side_height), fill=color)

    # Top Stator Cut-outs
    # if side == SlideSide.FRONT:
    y_start = y0
    if side == Side.REAR:
        y_start += 1120
    y_end = 480 + y_start
    for horizontal_x in [240 + geom.oX, (total_width - 240) - geom.oX]:
        r.rectangle((horizontal_x, y_start, horizontal_x + 2, y_end), fill=color)


def draw_metal_cutoffs(r, geom, y0, side):
    """
    Use to temporarily view the metal bracket locations
    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param int y0: vertical offset
    :param Side side:
    """
    b = 30  # offset of metal from boundary

    # Initial Boundary verticals
    verticals = [480 + geom.oX, geom.total_width - 480 - geom.oX]
    for i, start in enumerate(verticals):
        r.rectangle((start - 1, y0, start + 1, y0 + i), RGB_CUTOFF)

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

    side_height = geom.side_height
    total_width = geom.total_width
    # Create the left piece using format: (x1,x2,y1,y2)
    coords = [[240 + b + geom.oX, 480 - b + geom.oX, b + y0, b + y0],  # 1
              [b + geom.oX, 240 + b + geom.oX, 1120 + b + y0, 1120 + b + y0],  # 2
              [b + geom.oX, 480 - b + geom.oX, side_height - b + y0, side_height - b + y0],  # 3
              [240 + b + geom.oX, 240 + b + geom.oX, b + y0, 1120 + b + y0],  # 4
              [b + geom.oX, b + geom.oX, 1120 + b + y0, side_height - b + y0],  # 5
              [480 - b + geom.oX, 480 - b + geom.oX, b + y0, side_height - b + y0]]  # 6

    # Symmetrically create the right piece
    for i in range(0, len(coords)):
        (x1, x2, y1, y2) = coords[i]
        coords.append([total_width - x2, total_width - x1, y1, y2])

    # Transfer coords to points for printing (yeah I know it's dumb)
    points = coords
    # If backside, first apply a vertical reflection
    if side == Side.REAR:
        for i in range(0, len(coords)):
            (x1, x2, y1, y2) = coords[i]
            mid_y = 2 * y0 + side_height
            points.append([x1, x2, mid_y - y2, mid_y - y1])
    for i in range(0, 12):
        (x1, x2, y1, y2) = points[i]
        r.rectangle((x1 - 1, y1 - 1, x2 + 1, y2 + 1), fill=RGB_CUTOFF2)


# User Prompt Section

class Mode(Enum):
    RENDER = 'render'
    DIAGNOSTIC = 'diagnostic'
    STICKERPRINT = 'stickerprint'


VALID_MODES = [Mode.RENDER, Mode.DIAGNOSTIC, Mode.STICKERPRINT]


def prompt_for_mode():
    print('Type render, diagnostic, or stickerprint to set the desired mode')
    print('Each one does something different, so play around with it!')
    mode_accepted = False
    mode = None
    while not mode_accepted:
        mode = input('Mode selection: ')
        if mode in VALID_MODES:
            mode_accepted = True
            continue
        else:
            print('Check your spelling, and try again')
    return mode


# ---------------------- 6. Stickers -----------------------------


def draw_box(img_renderer, x0, y0, dx, dy):
    """
    :param ImageDraw.ImageDraw img_renderer:
    :param int x0: First corner of box
    :param int y0: First corner of box
    :param int dx: width extension of box in positive direction
    :param int dy: height extension of box in positive direction
    :return:
    """
    img_renderer.rectangle((x0, y0, x0 + dx, y0 + dy), outline=CUT_COLOR)


def draw_corners(r, x1, y1, x2, y2, arm_width=20):
    """
    :param ImageDraw.ImageDraw r:
    :param int x1: First corner of box
    :param int y1: First corner of box
    :param int x2: Second corner of box
    :param int y2: Second corner of box
    :param int arm_width: width of extension cross arms
    """

    # horizontal cross arms at 4 corners:
    r.line((x1 - arm_width, y1, x1 + arm_width, y1), CUT_COLOR)
    r.line((x1 - arm_width, y2, x1 + arm_width, y2), CUT_COLOR)
    r.line((x2 - arm_width, y1, x2 + arm_width, y1), CUT_COLOR)
    r.line((x2 - arm_width, y2, x2 + arm_width, y2), CUT_COLOR)
    # vertical cross arms at 4 corners:
    r.line((x1, y1 - arm_width, x1, y1 + arm_width), CUT_COLOR)
    r.line((x1, y2 - arm_width, x1, y2 + arm_width), CUT_COLOR)
    r.line((x2, y1 - arm_width, x2, y1 + arm_width), CUT_COLOR)
    r.line((x2, y2 - arm_width, x2, y2 + arm_width), CUT_COLOR)


def transcribe(src_img, dst_img, src_x, src_y, size_x, size_y, target_x, target_y):
    """
    (x0,y0) First corner of SOURCE (rendering)
    (dx,dy) Width and Length of SOURCE chunk to transcribe
    (xT,yT) Target corner of DESTINATION; where to in-plop (into stickerprint)

    Note to self: this is such a bad way to do this, instead of
    transcribing over literally thousands of pixels I should have
    just generated the scales in the place where they are needed

    :param src_img: SOURCE of pixels
    :param dst_img: DESTINATION of pixels
    :param src_x: First corner of SOURCE (rendering)
    :param src_y: First corner of SOURCE (rendering)
    :param size_x: Width of SOURCE chunk to transcribe
    :param size_y: Length of SOURCE chunk to transcribe
    :param target_x: Target corner of DESTINATION; where to in-plop (into stickerprint)
    :param target_y: Target corner of DESTINATION; where to in-plop (into stickerprint)
    """

    src_box = src_img.crop((src_x, src_y, src_x + size_x, src_y + size_y))
    dst_img.paste(src_box, (target_x, target_y))


def save_png(img_to_save, basename, output_suffix=None):
    output_filename = f"{basename}{'.' + output_suffix if output_suffix else ''}.png"
    img_to_save.save(output_filename, 'PNG')
    print(f'The result has been saved to {output_filename}')


def main():
    import argparse
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
    args_parser.add_argument('--debug',
                             action='store_true',
                             help='Render debug indications (corners and bounding boxes)')
    cli_args = args_parser.parse_args()
    render_mode = next(mode for mode in Mode if mode.value == (cli_args.mode or prompt_for_mode()))
    output_suffix = cli_args.suffix or ('test' if cli_args.test else None)
    render_cutoffs = cli_args.cutoffs
    should_delineate = True
    global DEBUG
    # DEBUG = cli_args.debug

    start_time = time.time()

    scale_width = 6500

    reg = FontStyle.REG
    upper = Align.UPPER
    lower = Align.LOWER
    geom = Geometry(
        (8000, 1600),
        (0, 0) if render_mode == Mode.STICKERPRINT else (100, 100),
        (5600, 160),
        (4, 70)
    )
    cols = black_on_white
    sliderule_img = Image.new('RGB', (geom.total_width, geom.print_height), cols.bg)
    r = ImageDraw.Draw(sliderule_img)
    if render_mode == Mode.RENDER or render_mode == Mode.STICKERPRINT:
        y_front_end = geom.side_height + 2 * geom.oY
        if render_mode == Mode.RENDER:
            draw_borders(r, geom, geom.oY, Side.FRONT)
            if render_cutoffs:
                draw_metal_cutoffs(r, geom, geom.oY, Side.FRONT)
            draw_borders(r, geom, y_front_end, Side.REAR)
            if render_cutoffs:
                draw_metal_cutoffs(r, geom, y_front_end, Side.REAR)

        # Front Scale
        gen_scale(r, geom, 110 + geom.oY, Scales.L, lower)
        gen_scale(r, geom, 320 + geom.oY, Scales.DF, lower)

        gen_scale(r, geom, 480 + geom.oY, Scales.CF, upper)
        gen_scale(r, geom, 640 + geom.oY, Scales.CIF, upper)
        gen_scale(r, geom, 800 + geom.oY, Scales.CI, lower)
        gen_scale(r, geom, 960 + geom.oY, Scales.C, lower)

        gen_scale(r, geom, 1120 + geom.oY, Scales.D, upper)
        gen_scale(r, geom, 1280 + geom.oY, Scales.R1, upper)
        gen_scale(r, geom, 1435 + geom.oY, Scales.R2, upper)

        # These are my weirdo alternative universe "brand names", "model name", etc.
        # Feel free to comment them out
        fs_lbl = FontSize.ScaleLBL
        side_width = geom.side_width
        li = geom.li
        y_off = 25 + geom.oY
        draw_symbol(r, geom, RED, y_off, 'BOGELEX 1000', side_width * 1 / 4 - li, 0, fs_lbl, reg, upper)
        draw_symbol(r, geom, RED, y_off, 'LEFT HANDED LIMAÇON 2020', side_width * 2 / 4 - li + geom.oX, 0, fs_lbl, reg,
                    upper)
        draw_symbol(r, geom, RED, y_off, 'KWENA & TOOR CO.', side_width * 3 / 4 - li, 0, fs_lbl, reg, upper)

        # Back Scale
        gen_scale(r, geom, 110 + y_front_end, Scales.K, lower)
        gen_scale(r, geom, 320 + y_front_end, Scales.A, lower)

        gen_scale(r, geom, 480 + y_front_end, Scales.B, upper)
        gen_scale(r, geom, 640 + y_front_end, Scales.T, lower)
        gen_scale(r, geom, 800 + y_front_end, Scales.ST, lower)
        gen_scale(r, geom, 960 + y_front_end, Scales.S, lower)

        gen_scale(r, geom, 1120 + y_front_end, Scales.D, upper)
        gen_scale(r, geom, 1360 + y_front_end, Scales.DI, upper)

    if render_mode == Mode.RENDER:
        save_png(sliderule_img, 'SlideRuleScales', output_suffix)

    if render_mode == Mode.DIAGNOSTIC:
        # If you're reading this, you're a real one
        # +5 brownie points to you
        scale_height = 160
        k = 120 + scale_height
        sh_with_margins = scale_height + 40
        total_height = k + (len(SCALE_NAMES) + 1) * sh_with_margins
        geom = Geometry(
            (6500, total_height),
            (250, 250),  # remove y-margin to stack scales
            (5600, scale_height),
            (4, 70)
        )
        diagnostic_img = Image.new('RGB', (geom.total_width, total_height), BG)
        r = ImageDraw.Draw(diagnostic_img)

        title_x = geom.midpoint_x - geom.li
        draw_symbol(r, geom, FG, 50, 'Diagnostic Test Print of Available Scales', title_x, 0,
                    FontSize.Title, reg, upper)
        draw_symbol(r, geom, FG, 200, 'A B C D K R1 R2 CI DI CF DF CIF L S T ST', title_x, 0,
                    FontSize.Subtitle, reg, upper)

        for n, sc_name in enumerate(SCALE_NAMES):
            overhang = 0.06 if n > 18 else 0.02
            sc = getattr(Scales, sc_name)
            gen_scale(r, geom, k + (n + 1) * sh_with_margins, sc, sc.al, overhang=overhang)

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
        geom = Geometry(
            (scale_width, 5075),
            (0, 0),
            (scale_width, 160),
            (4, 70)
        )
        total_width = scale_width + 2 * o_x2
        total_height = 5075

        stickerprint_img = Image.new('RGB', (total_width, total_height), BG)
        r = ImageDraw.Draw(stickerprint_img)

        # fsUM,MM,LM:
        y_bleed = 0

        y_bleed += o_y2 + o_a
        x_left = geom.oX + 750
        transcribe(sliderule_img, stickerprint_img, x_left, geom.oY, scale_width, 480, o_x2, y_bleed)
        extend(stickerprint_img, geom, y_bleed + 480 - 1, BleedDir.DOWN, ext)
        if should_delineate:
            draw_corners(r, o_x2, y_bleed - o_a, o_x2 + scale_width, y_bleed + 480)

        y_bleed += 480 + o_a
        transcribe(sliderule_img, stickerprint_img, x_left, geom.oY + 481, scale_width, 640, o_x2, y_bleed)
        extend(stickerprint_img, geom, y_bleed + 1, BleedDir.UP, ext)
        extend(stickerprint_img, geom, y_bleed + 640 - 1, BleedDir.DOWN, ext)
        if should_delineate:
            draw_corners(r, o_x2, y_bleed, o_x2 + scale_width, y_bleed + 640)

        y_bleed += 640 + o_a
        transcribe(sliderule_img, stickerprint_img, x_left, geom.oY + 1120, scale_width, 480, o_x2, y_bleed)
        extend(stickerprint_img, geom, y_bleed + 1, BleedDir.UP, ext)
        extend(stickerprint_img, geom, y_bleed + 480 - 1, BleedDir.DOWN, ext)
        if should_delineate:
            draw_corners(r, o_x2, y_bleed, o_x2 + scale_width, y_bleed + 480 + o_a)

        # bsUM,MM,LM:

        y_bleed += 480 + o_a + o_a + o_a

        y_start = geom.oY + geom.side_height + geom.oY
        transcribe(sliderule_img, stickerprint_img, x_left, y_start, scale_width, 480, o_x2, y_bleed)
        extend(stickerprint_img, geom, y_bleed + 480 - 1, BleedDir.DOWN, ext)
        if should_delineate:
            draw_corners(r, o_x2, y_bleed - o_a, o_x2 + scale_width, y_bleed + 480)

        y_bleed += 480 + o_a
        transcribe(sliderule_img, stickerprint_img, x_left, y_start + 481 - 3, scale_width, 640, o_x2, y_bleed)
        extend(stickerprint_img, geom, y_bleed + 1, BleedDir.UP, ext)
        extend(stickerprint_img, geom, y_bleed + 640 - 1, BleedDir.DOWN, ext)
        if should_delineate:
            draw_corners(r, o_x2, y_bleed, o_x2 + scale_width, y_bleed + 640)

        y_bleed += 640 + o_a
        transcribe(sliderule_img, stickerprint_img, x_left, y_start + 1120, scale_width, 480, o_x2, y_bleed)
        extend(stickerprint_img, geom, y_bleed + 1, BleedDir.UP, ext)
        extend(stickerprint_img, geom, y_bleed + 480 - 1, BleedDir.DOWN, ext)
        if should_delineate:
            draw_corners(r, o_x2, y_bleed, o_x2 + scale_width, y_bleed + 480 + o_a)

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
                draw_box(r, x0, y0, dx, dy)
                draw_box(r, x0, y0 + 640 + o_a, dx, dy)

            x0 = round(2 * (6.5 * o_a + 510 + 2 * 750) - x0 - dx)

            if should_delineate:
                draw_box(r, x0, y0, dx, dy)
                draw_box(r, x0, y0 + 640 + o_a, dx, dy)

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
            r.ellipse((p_x - hole_radius, p_y - hole_radius,
                       p_x + hole_radius, p_y + hole_radius),
                      fill=BG,
                      outline=CUT_COLOR)

            p_x = round(2 * (6.5 * o_a + 510 + 2 * 750) - p_x)

            r.ellipse((p_x - hole_radius, p_y - hole_radius,
                       p_x + hole_radius, p_y + hole_radius),
                      fill=BG,
                      outline=CUT_COLOR)

        save_png(stickerprint_img, 'StickerCut', output_suffix)

    print(f'The program took {round(time.time() - start_time, 2)} seconds to run')


if __name__ == '__main__':
    main()
