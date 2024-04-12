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
import inspect
import math
import re
import time
import unicodedata
from enum import Enum
from functools import cache

from PIL import Image, ImageFont, ImageDraw


def keys_of(obj):
    return [k for k, _ in inspect.getmembers(obj) if not k.startswith('__')]


# Angular constants:
TAU = math.tau
PI = math.pi
PI_HALF = PI / 2
DEG_FULL = 360
DEG_SEMI = 180
DEG_RIGHT_ANGLE = 90


LOG_ZERO = -math.inf


class BleedDir(Enum):
    UP = 'up'
    DOWN = 'down'


# ----------------------1. Setup----------------------------


class Colors:
    MAX = 255
    WHITE = (MAX, MAX, MAX)
    RED = (MAX, 0, 0)
    GREEN = (0, MAX, 0)
    BLUE = (0, 0, MAX)
    BLACK = (0, 0, 0)

    CUT = BLUE  # color which indicates CUT
    CUTOFF = (230, 230, 230)
    CUTOFF2 = (234, 36, 98)
    SYM_GREEN = (34, 139, 30)  # Override PIL for green for slide rule symbol conventions
    FC_LIGHT_BLUE_BG = (194, 235, 247)  # Faber-Castell scale background
    FC_LIGHT_GREEN_BG = (203, 243, 225)  # Faber-Castell scale background

    RED_WHITE_1 = (MAX, 224, 224)
    RED_WHITE_2 = (MAX, 192, 192)
    RED_WHITE_3 = (MAX, 160, 160)


class FontStyle(Enum):
    REG = 0  # regular
    ITALIC = 1  # italic
    BOLD = 2  # bold
    BOLD_ITALIC = 3  # bold italic


# per https://cm-unicode.sourceforge.io/font_table.html:
class FontFamilies:
    CMUTypewriter = (
        'cmuntt.ttf',  # CMUTypewriter-Regular
        'cmunit.ttf',  # CMUTypewriter-Italic
        'cmuntb.ttf',  # CMUTypewriter-Bold
        'cmuntx.ttf',  # CMUTypewriter-BoldItalic
    )
    CMUSansSerif = (
        'cmunss.ttf',  # CMUSansSerif
        'cmunsi.ttf',  # CMUSansSerif-Oblique
        'cmunsx.ttf',  # CMUSansSerif-Bold
        'cmunso.ttf',  # CMUSansSerif-BoldOblique
    )
    CMUConcrete = (
        'cmunorm.ttf',  # CMUConcrete-Roman
        'cmunoti.ttf',  # CMUConcrete-Italic
        'cmunobx.ttf',  # CMUConcrete-Bold
        'cmunobi.ttf',  # CMUConcrete-BoldItalic
    )
    CMUBright = (
        # 'cmunbmr.ttf',  # CMUBright-Roman
        # 'cmunbmo.ttf',  # CMUBright-Oblique
        'cmunbsr.ttf',  # CMUBright-SemiBold
        'cmunbso.ttf',  # CMUBright-SemiBoldOblique
        'cmunbsr.ttf',  # CMUBright-SemiBold
        'cmunbso.ttf',  # CMUBright-SemiBoldOblique
    )


class Style:
    fg = Colors.BLACK
    """foreground color black"""
    bg = Colors.WHITE
    """background color white"""
    overrides_by_sc_key: dict[str, dict] = {}

    def __init__(self, fg_color=fg, bg_color=bg,
                 decreasing_color=Colors.RED,
                 decimal_color=None,
                 sc_bg_colors: dict = None,
                 font_family=FontFamilies.CMUTypewriter,
                 overrides_by_sc_key=None):
        self.fg = fg_color
        self.bg = bg_color
        self.dec_color = decreasing_color
        self.decimal_color = decimal_color
        self.sc_bg_colors = sc_bg_colors or {}
        self.font_family = font_family
        self.overrides_by_sc_key = overrides_by_sc_key or {}

    def __repr__(self):
        return (f'Style(fg_color={self.fg}, bg_color={self.bg},'
                f' decreasing_color={self.dec_color}, decimal_color={self.decimal_color},'
                f' sc_bg_colors={self.sc_bg_colors}'
                f' font_family={self.font_family})')

    def scale_fg_col(self, sc):
        """:type sc: Scale"""
        return self.fg if sc.is_increasing else self.dec_color

    def scale_bg_col(self, sc):
        """:type sc: Scale"""
        return self.sc_bg_colors.get(sc.key)

    def overrides_for(self, sc) -> dict:
        """
        :type sc: Scale
        """
        return self.overrides_by_sc_key.get(sc.key)

    def override_for(self, sc, key, default):
        sc_overrides = self.overrides_for(sc)
        return sc_overrides.get(key, default) if sc_overrides else default

    def numeral_decimal_color(self):
        return self.decimal_color

    @cache
    def font_for(self, font_size, font_style=FontStyle.REG, h_ratio=None):
        """
        :param FontStyle font_style: font style
        :param FontSize|int font_size: font size
        :param h_ratio: proportion of requested font size to downsize by
        :return: FreeTypeFont
        """
        font_name = self.font_family[font_style.value]
        fs = font_size if isinstance(font_size, int) else font_size.value
        if h_ratio and h_ratio != 1:
            fs = round(fs * h_ratio)
        return ImageFont.truetype(font_name, fs)

    @staticmethod
    def sym_dims(symbol, font):
        """
        Gets the size dimensions (width, height) of the input text
        :param str symbol: the text
        :param FreeTypeFont font: font
        :return: Tuple[int, int]
        """
        (x1, y1, x2, y2) = font.getbbox(symbol)
        return x2 - x1, y2 - y1 + 20

    @classmethod
    def sym_width(cls, symbol, font):
        """
        :param str symbol:
        :param FreeTypeFont font:
        :return: int
        """
        (x1, _, x2, _) = font.getbbox(symbol)
        return x2 - x1


class Styles:
    Default = Style()

    PickettEyeSaver = Style(
        font_family=FontFamilies.CMUBright,
        bg_color=(253, 253, 150)  # pastel yellow
    )

    Graphoplex = Style(
        font_family=FontFamilies.CMUBright,
        decimal_color='lightblue'
    )


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
    side_w: int = 8000  # 30cm = 11.8in
    side_h: int = 1600  # 6cm = 2.36in
    slide_h: int = 640  # 2.4cm = 0.945in

    SH: int = 160  # 6mm
    """scale height"""
    SL: int = 5600  # 21cm = 8.27in
    """scale length"""
    DEFAULT_SCALE_WH = (SL, SH)

    # Ticks, Labels, are referenced from li as to be consistent
    STH: int = 70  # 2.62mm
    """standard tick height"""
    STT: int = 4  # 0.15mm
    """standard tick thickness"""
    PixelsPerCM = 1600 / 6
    PixelsPerIN = PixelsPerCM * 2.54

    NO_MARGINS = (0, 0)
    DEFAULT_TICK_WH = (STT, STH)

    top_margin = 110

    def __init__(self, side_wh: (int, int), margins_xy: (int, int), scale_wh: (int, int), tick_wh: (int, int),
                 slide_h: int,
                 top_margin: int = None,
                 scale_h_front: dict[str, int] = None,
                 scale_h_rear: dict[str, int] = None):
        (self.side_w, self.side_h) = side_wh
        (self.oX, self.oY) = margins_xy
        (self.SL, self.SH) = scale_wh
        (self.STT, self.STH) = tick_wh
        self.slide_h = slide_h
        if top_margin:
            self.top_margin = top_margin
        self.scale_h_front_by_key = scale_h_front or {}
        self.scale_h_rear_by_key = scale_h_rear or {}

    @property
    def total_w(self):
        return self.side_w + 2 * self.oX

    @property
    def midpoint_x(self):
        return self.total_w / 2

    @property
    def print_height(self):
        return self.side_h * 2 + 3 * self.oY

    @property
    def stator_h(self):
        return (self.side_h - self.slide_h) // 2

    @property
    def cutoff_w(self):
        """Default cutoff width ensures a square anchor piece."""
        return self.stator_h

    @property
    def li(self):
        """left index offset from left edge"""
        return (self.total_w - self.SL) // 2

    @property
    def min_tick_offset(self):
        """minimum tick horizontal offset"""
        return self.STT * 2  # separate each tick by at least the space of its width

    def tick_h(self, h_mod: HMod = None, h_ratio=None) -> int:
        result = self.STH
        if h_mod:
            result *= h_mod.value
        if h_ratio and h_ratio != 1:
            result *= h_ratio
        return round(result)

    def scale_h(self, sc, side=None, default=None) -> int:
        """
        :param Scale sc:
        :param Side side:
        :param int default:
        """
        if side == Side.FRONT:
            return self.scale_h_front_by_key.get(sc.key, default or self.SH)
        if side == Side.REAR:
            return self.scale_h_rear_by_key.get(sc.key, default or self.SH)
        return self.scale_h_front_by_key.get(
            sc.key,
            self.scale_h_rear_by_key.get(sc.key, default or self.SH))

    def scale_h_ratio(self, sc, side=None):
        scale_h = self.scale_h(sc, side=side)
        default_scale_h = self.DEFAULT_SCALE_WH[1]
        return scale_h / default_scale_h if scale_h != default_scale_h else None


class FontSize(Enum):
    TITLE = 140
    SUBTITLE = 120
    SC_LBL = 90
    N_XL = 75
    N_LG = 60
    N_MD = 55
    N_SM = 45
    N_XS = 35


class Align(Enum):
    """Scale Alignment (ticks and labels against upper or lower bounds)"""
    UPPER = 'upper'  # Upper alignment
    LOWER = 'lower'  # Lower Alignment


class Side(Enum):
    """Side of the slide (front or rear)"""
    FRONT = 'front'
    REAR = 'rear'


# ----------------------2. Fundamental Functions----------------------------


def draw_tick(r, geom, col, y_off, scale_h, x, height, al):
    """
    Places an individual tick
    :param ImageDraw.ImageDraw r:
    :param Geometry geom:
    :param str col: color
    :param y_off: y pos
    :param scale_h: scale height
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
        y0 = y_off + scale_h - height
        y1 = y_off + scale_h
    r.rectangle((x0, y0, x0 + geom.STT, y1), fill=col)


def i_range(first: int, last: int, include_last: bool):
    return range(first, last + (1 if include_last else 0))


def range_div(first: int, last: int, scale_factor: int, include_last: bool):
    return (x / scale_factor for x in i_range(first, last, include_last))


def range_mul(first: int, last: int, scale_factor: int, include_last: bool):
    return (x * scale_factor for x in i_range(first, last, include_last))


def i_range_tenths(first: int, last: int, include_last=True, sf=100) -> range:
    return i_range(first * sf, last * sf, include_last)


def grad_pat_auto(r, geom, style, y_off, sc, al, start_value=None, end_value=None, include_last=False):
    """
    Draw a graduated pattern of tick marks across the scale range.
    Determine the lowest digit tick mark spacing and work upwards from there.

    Tick Patterns:
    * 1-.5-.1-.05
    * 1-.5-.1-.02
    * 1-.5-.1-.01
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
    scale_w = geom.SL
    sf = 10 ** num_digits  # ensure enough precision for int ranges
    # Ensure between 6 and 15 numerals will display? Target log10 in 0.8..1.17
    frac_width = sc.offset_between(start_value, end_value, 1)
    step_numeral = 10 ** (math.floor(math.log10(abs(end_value - start_value)) - 0.5 * frac_width) + num_digits)
    step_half = step_numeral // 2
    step_tenth = step_numeral // 10  # second level
    tenth_tick_offset = sc.smallest_diff_size_for_delta(start_value, end_value, step_tenth / sf, scale_w)
    if tenth_tick_offset < min_tick_offset:
        step_tenth = step_numeral
    step_last = step_tenth  # last level
    for tick_div in [10, 5, 2]:
        v = step_tenth / tick_div / sf
        smallest_tick_offset = sc.smallest_diff_size_for_delta(start_value, end_value, v, scale_w)
        if smallest_tick_offset >= min_tick_offset:
            step_last = max(round(step_tenth / tick_div), 1)
            break
    scale_hf = geom.scale_h_ratio(sc)
    num_th = geom.tick_h(HMod.MED, scale_hf)
    half_th = geom.tick_h(HMod.XL if step_tenth < step_numeral else HMod.XS, scale_hf)
    dot_th = geom.tick_h(HMod.DOT, scale_hf)
    tenth_th = geom.tick_h(HMod.XS, scale_hf) if step_last < step_tenth else dot_th
    # Ticks and Labels
    i_start = int(start_value * sf)
    i_offset = i_start % step_last
    if i_offset > 0:  # Align to first tick on or after start
        i_start = i_start - i_offset + step_last
    num_font = style.font_for(FontSize.N_LG, h_ratio=scale_hf)
    numeral_tick_offset = sc.smallest_diff_size_for_delta(start_value, end_value, step_numeral / sf, scale_w)
    max_num_chars = numeral_tick_offset / style.sym_width('_', num_font)
    if max_num_chars < 4:
        num_font = style.font_for(FontSize.N_SM, h_ratio=scale_hf)
    single_digit = max_num_chars < 2
    tenth_font = style.font_for(FontSize.N_XS, h_ratio=scale_hf)
    # If there are sub-digit ticks to draw, and enough space for single-digit numerals:
    draw_tenth = (step_last < step_tenth < step_numeral) and max_num_chars > 8
    i_end = int(end_value * sf + (1 if include_last else 0))
    grad_pat(r, geom, style, y_off, sc, al,
             i_start, i_end, sf,
             (step_numeral, step_half, step_tenth, step_last),
             (num_th, half_th, tenth_th, dot_th),
             (num_font, None, tenth_font if draw_tenth else None),
             single_digit)


def grad_pat(r, geom, style, y_off: int, sc, al: Align,
             i_start, i_end, i_sf, steps_i, steps_th, steps_font, single_digit):
    """
    Place ticks in a graduated pattern. All options are given, not inferred.

    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param Style style:
    :param y_off: y pos
    :param Scale sc:
    :param Align al: alignment
    :param int i_start: index across the scale to start at
    :param int i_end: index across the scale to end at
    :param int i_sf: scale factor - how much to divide the inputs by before scaling (to generate fine decimals)
    :param tuple[int, int, int, int] steps_i: patterning steps, large to small
    :param tuple[int, int, int, int] steps_th: tick sizes, large to small, per step
    :param tuple[FreeTypeFont, FreeTypeFont, FreeTypeFont] steps_font: optional font sizes, for numerals above ticks
    :param bool single_digit: whether to show the main numerals as the most relevant digit only
    """
    step1, step2, step3, step4 = steps_i
    th1, th2, th3, th4 = steps_th
    font1, font2, font3 = steps_font
    scale_w = geom.SL
    scale_h = geom.scale_h(sc)
    sym_col = sc.col
    tenth_col = style.decimal_color
    for i in range(i_start, i_end, step4):
        num = i / i_sf
        x = sc.scale_to(num, scale_w)
        tick_h = th4
        if i % step1 == 0:
            tick_h = th1
            if font1:
                if single_digit:
                    num = sig_digit_of(num)
                draw_numeral(r, geom, style, sym_col, y_off, scale_h, num, x, th1, font1, al)
        elif i % step2 == 0:
            tick_h = th2
            if font2:
                draw_numeral(r, geom, style, sym_col, y_off, scale_h, last_digit_of(num), x, th2, font2, al)
        elif i % step3 == 0:
            tick_h = th3
            if font3:
                draw_numeral(r, geom, style, tenth_col, y_off, scale_h, last_digit_of(num), x, th3, font3, al)
        draw_tick(r, geom, sym_col, y_off, scale_h, x, tick_h, al)


DEBUG = False
DRAW_RADICALS = True


def draw_symbol(r, style, symbol, color, x_left, y_top, font):
    """
    :param ImageDraw.Draw r:
    :param Style style:
    :param str symbol:
    :param str|tuple[int] color:
    :param Number x_left:
    :param Number y_top:
    :param FreeTypeFont font:
    """
    if '∡' in symbol:
        symbol = symbol.replace('∡', '')
    if DEBUG:
        w, h = style.sym_dims(symbol, font)
        print(f'draw_symbol_inner: {symbol}\t{x_left} {y_top} {w} {h}')
        r.rectangle((x_left, y_top, x_left + w, y_top + h), outline='grey')
        r.rectangle((x_left, y_top, x_left + 10, y_top + 10), outline='navy', width=4)
    r.text((x_left, y_top), symbol, font=font, fill=color)
    if DRAW_RADICALS:
        radicals = re.search(r'[√∛∜]', symbol)
        if radicals:
            w, h = style.sym_dims(symbol, font)
            n_ch = radicals.start() + 1
            (w_ch, h_rad) = style.sym_dims('√', font)
            (_, h_num) = style.sym_dims('1', font)
            if DEBUG:
                print(f"DRAW_RADICALS: {h_rad}, {h}, {h_num}")
            line_w = h_rad // 14
            y_bar = y_top + max(10, round(h - h_num - line_w * 2))
            r.line((x_left + w_ch * n_ch - w_ch // 10, y_bar, x_left + w, y_bar), width=line_w, fill=color)


def draw_sym_al(r, geom, style, color, y_off, al_h, symbol, x, y, font, al):
    """
    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param Style style:
    :param str|tuple[int] color: color name that PIL recognizes
    :param int y_off: y pos
    :param int al_h: height for alignment
    :param str symbol: content (text or number)
    :param x: offset of centerline from left index (li)
    :param y: offset of base from baseline (LOWER) or top from upper line (UPPER)
    :param FreeTypeFont font:
    :param Align al: alignment
    """

    if not symbol:
        return
    (base_sym, exponent, subscript) = symbol_parts(symbol)
    w, h = Style.sym_dims(base_sym, font)

    y_top = y_off
    if al == Align.UPPER:
        y_top += y
    elif al == Align.LOWER:
        y_top += al_h - 1 - y - h * 1.2
    x_left = x + geom.li - w / 2 + geom.STT / 2
    draw_symbol(r, style, base_sym, color, round(x_left), y_top, font)

    if exponent or subscript:
        sub_font_size = FontSize.N_LG if font.size == FontSize.SC_LBL else font.size
        sub_font = style.font_for(sub_font_size, h_ratio=0.75)
        x_right = round(x_left + w)
        if exponent:
            draw_symbol_sup(r, style, exponent, color, h, x_right, y_top, sub_font)
        if subscript:
            draw_symbol_sub(r, style, subscript, color, h, x_right, y_top, sub_font)


def draw_numeral(r, geom, style, color, y_off, scale_h, num, x, y, font, al):
    """
    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param Style style:
    :param str|tuple[int] color: color name that PIL recognizes
    :param int y_off: y pos
    :param int scale_h:
    :param Number num: the number to draw
    :param x: offset of centerline from left index (li)
    :param y: offset of base from baseline (LOWER) or top from upper line (UPPER)
    :param FreeTypeFont font:
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

    draw_sym_al(r, geom, style, color, y_off, scale_h, num_sym, x, y, font, al)


RE_EXPON_CARET = re.compile(r'^(.+)\^([-0-9.A-Za-z]+)$')
RE_SUB_UNDERSCORE = re.compile(r'^(.+)_([-0-9.A-Za-z]+)$')
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


PRIMES = "'ʹʺ′″‴"


def symbol_with_expon(symbol: str):
    if len(symbol) > 1 and symbol[-1] in PRIMES:
        return symbol[:-1], symbol[-1:]
    return split_symbol_by(symbol, RE_EXPON_CARET, RE_EXPON_UNICODE)


def symbol_with_subscript(symbol: str):
    return split_symbol_by(symbol, RE_SUB_UNDERSCORE, RE_SUB_UNICODE)


def symbol_parts(symbol: str):
    (base_sym, subscript) = symbol_with_subscript(symbol)
    (base_sym, expon) = symbol_with_expon(base_sym)
    return base_sym, expon, subscript


def draw_symbol_sup(r, style, sup_sym, color, h_base, x_left, y_base, font):
    if len(sup_sym) == 1 and unicodedata.category(sup_sym) == 'No':
        sup_sym = str(unicodedata.digit(sup_sym))
    draw_symbol(r, style, sup_sym, color, x_left, y_base - (0 if sup_sym in PRIMES else h_base / 2), font)


def draw_symbol_sub(r, style, sub_sym, color, h_base, x_left, y_base, font):
    draw_symbol(r, style, sub_sym, color, x_left, y_base + h_base / 2, font)


def extend(image, geom, y, direction, amplitude):
    """
    Used to create bleed for sticker cutouts
    :param Image.Image image:
    :param Geometry geom:
    :param int y: y pixel row to duplicate
    :param BleedDir direction: direction
    :param int amplitude: number of pixels to extend
    """

    w = geom.total_w
    for x in range(0, w):
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
def scale_inverse_square(x): return 1 - gen_base(x) / 2


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
    # y = math.sqrt(1+x**2) = math.hypot(1, x)
    return gen_base(math.sqrt((x ** 2) - 1))


def scale_log_log(x): return gen_base(math.log(x))


def scale_neg_log_log(x): return gen_base(-math.log(x))


def angle_opp(x: int) -> int:
    """The opposite angle in degrees across a right triangle."""
    return DEG_RIGHT_ANGLE - x


class RulePart(Enum):
    STATOR = 'stator'
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
    neper_to_db = InvertibleFn(lambda x_db: x_db / (20 / math.log(TEN)), lambda x_n: x_n * 20 / math.log(TEN))


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
    InverseSquare = Scaler(scale_inverse_square, lambda p: pos_base(1 - p * 2), increasing=False)
    SquareRoot = Scaler(scale_sqrt, lambda p: pos_base(p / 2))
    Log10 = Scaler(scale_log, lambda p: p * TEN)
    Ln = Scaler(lambda x: x / LOG_TEN, lambda p: p * LOG_TEN)
    Sin = Scaler(scale_sin, math.asin)
    CoSin = Scaler(scale_cos, math.acos, increasing=False)
    Tan = Scaler(scale_tan, lambda p: math.atan(pos_base(p)))
    SinTan = Scaler(scale_sin_tan, lambda p: math.atan(pos_base(p)))
    SinTanRadians = Scaler(scale_sin_tan_radians, lambda p: math.atan(pos_base(math.degrees(p))))
    CoTan = Scaler(scale_cot, lambda p: math.atan(DEG_RIGHT_ANGLE - p), increasing=False)
    SinH = Scaler(scale_sinh, lambda p: math.asinh(pos_base(p)))
    CosH = Scaler(scale_cosh, lambda p: math.acosh(pos_base(p)))
    TanH = Scaler(scale_tanh, lambda p: math.atanh(pos_base(p)))
    Pythagorean = Scaler(scale_pythagorean, lambda p: math.sqrt(1 - (pos_base(p) / 10) ** 2), increasing=False)
    Chi = Scaler(lambda x: x / PI_HALF, lambda p: p * PI_HALF)
    Theta = Scaler(lambda x: x / DEG_RIGHT_ANGLE, lambda p: p * DEG_RIGHT_ANGLE)
    LogLog = Scaler(scale_log_log, lambda p: math.exp(pos_base(p)))
    LogLogNeg = Scaler(scale_neg_log_log, lambda p: math.exp(pos_base(-p)), increasing=False)
    Hyperbolic = Scaler(scale_hyperbolic, lambda p: math.hypot(1, pos_base(p)))


class Scale:
    al: Align = None
    """If a scale has an alignment its label or relationship to another scale implies."""

    def __init__(self, left_sym: str, right_sym: str, scaler: callable, shift: float = 0,
                 increasing=True, key=None, rule_part=RulePart.STATOR, opp_scale=None):
        self.left_sym = left_sym
        """left scale symbol"""
        self.right_sym = right_sym
        """right scale symbol"""
        self.scaler: Scaler = scaler
        self.gen_fn = scaler.fn if isinstance(scaler, Scaler) else scaler
        """generating function (producing a fraction of output width)"""
        self.pos_fn = scaler.inverse if isinstance(scaler, Scaler) else None
        """positioning function (takes a proportion of output width, returning what value is there)"""
        self.shift = shift
        """scale shift from left index (as a fraction of output width)"""
        self.is_increasing = scaler.is_increasing if isinstance(scaler, Scaler) else increasing
        """whether the scale values increase from left to right"""
        self.key = key or left_sym
        """non-unicode name; unused"""  # TODO extend for all alternate namings?
        self.rule_part = rule_part
        """which part of the rule it's on, slide vs stator"""
        self.opp_scale = opp_scale
        """which scale, if on an edge, it's aligned with"""
        if opp_scale:
            opp_scale.opp_scale = self
            self.al = Align.UPPER
            opp_scale.al = Align.LOWER

    def displays_cyclic(self):
        return self.scaler in {Scalers.Base, Scalers.Inverse, Scalers.Square, Scalers.Cube}

    def can_spiral(self):
        return self.scaler in {Scalers.LogLog, Scalers.LogLogNeg}

    def __repr__(self):
        return f'Scale({self.key}, {self.right_sym}, {self.scaler})'

    @property
    def col(self):
        return Styles.Default.scale_fg_col(self)

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
        start_log = math.log10(start_value) if start_value > 0 else LOG_ZERO
        end_log = math.log10(end_value) if end_value > 0 else LOG_ZERO
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

    def grad_pat_divided(self, r, geom, style, y_off, al, dividers, start_value=None, end_value=None):
        if dividers is None:
            dividers = [10 ** n for n in self.powers_of_ten_in_range()]
        if dividers:
            grad_pat_auto(r, geom, style, y_off, self, al, start_value=start_value, end_value=dividers[0])
            last_i = len(dividers) - 1
            for i, di in enumerate(dividers):
                is_last = i >= last_i
                dj = end_value if is_last else dividers[i + 1]
                grad_pat_auto(r, geom, style, y_off, self, al, start_value=di, end_value=dj, include_last=is_last)
        else:
            grad_pat_auto(r, geom, style, y_off, self, al, start_value=start_value, end_value=end_value, include_last=True)


class Scales:
    A = Scale('A', 'x²', Scalers.Square)
    B = Scale('B', 'x²_y', Scalers.Square, rule_part=RulePart.SLIDE, opp_scale=A)
    C = Scale('C', 'x_y', Scalers.Base, rule_part=RulePart.SLIDE)
    DF = Scale('DF', 'πx', Scalers.Base, shift=pi_fold_shift)
    CF = Scale('CF', 'πx_y', Scalers.Base, shift=pi_fold_shift, rule_part=RulePart.SLIDE, opp_scale=DF)
    CI = Scale('CI', '1/x_y', Scalers.Inverse, rule_part=RulePart.SLIDE)
    CIF = Scale('CIF', '1/πx_y', Scalers.Inverse, shift=pi_fold_shift - 1, rule_part=RulePart.SLIDE)
    D = Scale('D', 'x', Scalers.Base, opp_scale=C)
    DI = Scale('DI', '1/x', Scalers.Inverse)
    K = Scale('K', 'x³', Scalers.Cube)
    L = Scale('L', 'log x', Scalers.Log10)
    Ln = Scale('Ln', 'ln x', Scalers.Ln)
    LL0 = Scale('LL₀', 'e^0.001x', Scalers.LogLog, shift=3, key='LL0')
    LL1 = Scale('LL₁', 'e^0.01x', Scalers.LogLog, shift=2, key='LL1')
    LL2 = Scale('LL₂', 'e^0.1x', Scalers.LogLog, shift=1, key='LL2')
    LL3 = Scale('LL₃', 'e^x', Scalers.LogLog, key='LL3')
    LL00 = Scale('LL₀₀', 'e^-0.001x', Scalers.LogLogNeg, shift=3, key='LL00')
    LL01 = Scale('LL₀₁', 'e^-0.01x', Scalers.LogLogNeg, shift=2, key='LL01')
    LL02 = Scale('LL₀₂', 'e^-0.1x', Scalers.LogLogNeg, shift=1, key='LL02')
    LL03 = Scale('LL₀₃', 'e^-x', Scalers.LogLogNeg, key='LL03')
    P = Scale('P', '√1-(0.1x)²', Scalers.Pythagorean, key='P')
    R1 = Scale('R₁', '√x', Scalers.SquareRoot, key='R1')
    R2 = Scale('R₂', '√10x', Scalers.SquareRoot, key='R2', shift=-1)
    S = Scale('S', '∡sin x°', Scalers.Sin)
    CoS = Scale('C', '∡cos x°', Scalers.CoSin)
    SRT = Scale('SRT', '∡tan 0.01x', Scalers.SinTanRadians)
    ST = Scale('ST', '∡tan 0.01x°', Scalers.SinTan)
    T = Scale('T', '∡tan x°', Scalers.Tan)
    CoT = Scale('T', '∡cot x°', Scalers.CoTan)
    T1 = Scale('T₁', '∡tan x°', Scalers.Tan, key='T1')
    T2 = Scale('T₂', '∡tan 0.1x°', Scalers.Tan, key='T2', shift=-1)
    W1Prime = Scale("W'₁", '√x', Scalers.SquareRoot, key='W1Prime')
    W1 = Scale('W₁', '√x', Scalers.SquareRoot, key='W1', opp_scale=W1Prime)
    W2 = Scale('W₂', '√10x', Scalers.SquareRoot, key='W2', shift=-1)
    W2Prime = Scale("W'₂", '√10x', Scalers.SquareRoot, key='W2Prime', shift=-1, opp_scale=W2)

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
    L_r = Scale('L_r', '1/(2πx)²', Scalers.InverseSquare, shift=gen_base(1 / TAU))


# TODO scales from Aristo 965 Commerz II: KZ, %, Z/ZZ1/ZZ2/ZZ3 compound interest
# TODO meta-scale showing % with 100% over 1/unity
#  special marks being 0,5,10,15,20,25,30,33⅓,40,50,75,100 in both directions


SCALE_NAMES = set(keys_of(Scales))


class Layout:
    def __init__(self, front_sc_keys: str, rear_sc_keys: str,
                 front_aligns: dict[str, Align] = None, rear_aligns: dict[str, Align] = None):
        if not rear_sc_keys and '\n' in front_sc_keys:
            (front_sc_keys, rear_sc_keys) = front_sc_keys.splitlines()
        self.front_sc_keys: list[list[str]] = self.parse_side_layout(front_sc_keys)
        self.rear_sc_keys: list[list[str]] = self.parse_side_layout(rear_sc_keys)
        self.check_scales()
        self.front_aligns_by_sc_key = front_aligns or {}
        self.rear_aligns_by_sc_key = rear_aligns or {}

    def __repr__(self):
        return f'Layout({self.front_sc_keys}, {self.rear_sc_keys})'

    @classmethod
    def parse_segment_layout(cls, segment_layout: str) -> [str]:
        if segment_layout:
            return re.split(r'[, ]+', segment_layout.strip(' '))
        return None

    @classmethod
    def parts_of_side_layout(cls, side_layout: str) -> [str]:
        if '/' in side_layout:
            return side_layout.split('/')
        parts = re.fullmatch(r'\|?\s*(.+)\[(.+)](.*)\s*\|?', side_layout)
        if parts:
            return [parts.group(1), parts.group(2), parts.group(3)]
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
        for front_part in self.front_sc_keys:
            if not front_part:
                continue
            for scale_name in front_part:
                if scale_name not in SCALE_NAMES:
                    raise Exception(f'Unrecognized front scale name: {scale_name}')
        for rear_part in self.front_sc_keys:
            if not rear_part:
                continue
            for scale_name in rear_part:
                if scale_name not in SCALE_NAMES:
                    raise Exception(f'Unrecognized rear scale name: {scale_name}')

    def scale_names_in_order(self):
        for part in self.front_sc_keys:
            if part:
                yield from part
        for part in self.rear_sc_keys:
            if part:
                yield from part

    def scale_names(self):
        seen = set()
        all_names = self.scale_names_in_order()
        return [sc_name for sc_name in all_names if not (sc_name in seen or seen.add(sc_name))]

    def scales_at(self, side: Side, part: RulePart, top: bool) -> list[Scale]:
        layout = self.front_sc_keys if side == Side.FRONT else self.rear_sc_keys
        layout_i = 0
        if part == RulePart.SLIDE:
            layout_i = 1
        elif part == RulePart.STATOR:
            layout_i = 0 if top else 2
        return [getattr(Scales, sc_name) for sc_name in layout[layout_i] or []]

    def scale_al(self, sc: Scale, side: Side, top: bool):
        default_al = sc.al or (top and Align.LOWER or Align.UPPER)
        if side == Side.FRONT:
            return self.front_aligns_by_sc_key.get(sc.key, default_al)
        if side == Side.REAR:
            return self.rear_aligns_by_sc_key.get(sc.key, default_al)


class Layouts:
    MannheimOriginal = Layout('A/B C/D', '')
    RegleDesEcoles = Layout('DF/CF C/D', '')
    Mannheim = Layout('A/B CI C/D K', 'S L T')
    Rietz = Layout('K A/B CI C/D L', 'S ST T')
    Darmstadt = Layout('S T A/B K CI C/D P', 'L LL1 LL2 LL3')


class Model:
    def __init__(self, brand: str, subtitle: str, model_name: str,
                 geometry: Geometry, layout: Layout, style: Style = Styles.Default):
        self.brand = brand
        self.subtitle = subtitle
        self.name = model_name
        self.geometry = geometry
        self.layout = layout
        self.style = style

    def __repr__(self):
        return f'Model({self.brand}, {self.name}, {self.layout}, style={self.style})'

    def scale_h_per(self, side: Side, part: RulePart, top: bool):
        result = 0
        for sc in self.layout.scales_at(side, part, top):
            result += self.geometry.scale_h(sc, side)
        return result

    def auto_stock_h(self):
        result = 0
        for side in [Side.FRONT, Side.REAR]:
            for top in True, False:
                result = max(result, self.scale_h_per(side, RulePart.STATOR, top))
        return result

    def auto_slide_h(self):
        result = 0
        for side in [Side.FRONT, Side.REAR]:
            result = max(result, self.scale_h_per(side, RulePart.SLIDE, True))
        return result


class Models:
    Demo = Model('KWENA & TOOR CO.', 'LEFT HANDED LIMAÇON 2020', 'BOGELEX 1000',
                 Geometry((8000, 1600),
                          (100, 100),
                          Geometry.DEFAULT_SCALE_WH,
                          Geometry.DEFAULT_TICK_WH,
                          640,
                          top_margin=109,
                          scale_h_front={
                              'R1': 155
                          },
                          scale_h_rear={
                              'D': 240
                          }),
                 Layout('|  L,  DF [ CF,CIF,CI,C ] D, R1, R2 |',
                        '|  K,  A  [ B, T, ST, S ] D,  DI    |',
                        front_aligns={'CIF': Align.UPPER}))

    MannheimOriginal = Model('Mannheim', 'Demo', 'Original',
                             Geometry((8000, 1000),
                                      (100, 100),
                                      Geometry.DEFAULT_SCALE_WH,
                                      Geometry.DEFAULT_TICK_WH,
                                      round(Geometry.SH * 2.5),
                                      top_margin=109),
                             Layouts.MannheimOriginal)

    Aristo868 = Model('Aristo', '', '868',
                      Geometry((8000, 1860),
                               (100, 100),
                               (Geometry.SL, 120),
                               Geometry.DEFAULT_TICK_WH,
                               590,
                               top_margin=0),
                      Layout(
                          'ST T1 T2 DF/CF CIF CI C/D P S',
                          'LL01 LL02 LL03 A/B L K C/D LL3 LL2 LL1'
                      ),
                      Style(font_family=FontFamilies.CMUBright))
    PickettN515T = Model('Pickett', '', 'N-515-T',
                         Geometry((8000, 2000),
                                  (100, 100),
                                  Geometry.DEFAULT_SCALE_WH,
                                  Geometry.DEFAULT_TICK_WH,
                                  Geometry.SH * 5),
                         Layout(
                             'L_r f_x A/B S T CI C/D L Ln', ''
                         ),
                         Styles.PickettEyeSaver)
    FaberCastell283 = Model('Faber-Castell', '', '2/83',
                            Geometry((8800, 1280),  # 33cm (px) x 4.8cm (1280px)
                                     (100, 100),
                                     (6666, 101),  # 25cm (6666.7px) x 3.5mm (93px)
                                     Geometry.DEFAULT_TICK_WH,
                                     Geometry.SH * 4,
                                     top_margin=0),
                            Layout(
                                'K T1 T2 DF/CF CIF CI C/D S ST P',
                                'LL03 LL02 LL01 W2/W2Prime L C W1Prime/W1 LL1 LL2 LL3',
                                front_aligns={
                                    'T2': Align.UPPER,
                                    'CI': Align.UPPER,
                                    'DF': Align.LOWER,
                                    'S': Align.LOWER,
                                },
                                rear_aligns={
                                    'C': Align.UPPER
                                }
                            ),
                            Style(Colors.BLACK, Colors.WHITE,
                                  font_family=FontFamilies.CMUBright,
                                  sc_bg_colors={
                                      'C': Colors.FC_LIGHT_GREEN_BG,
                                      'CF': Colors.FC_LIGHT_GREEN_BG
                                  }
                                  ))
    FaberCastell283N = Model('Faber-Castell', '', '2/83N',
                             Geometry((9866, 1520),  # 37cm (9866px) x 5.7cm (1520px)
                                      (0, 0),
                                      (6666, 101),  # 25cm (6666.7px) x 3.5mm (93px)
                                      (3, 50),
                                      510,  # 1.9cm (506.6px)
                                      top_margin=0,
                                      scale_h_front={
                                      },
                                      scale_h_rear={
                                          'LL03': 74,
                                          'LL02': 74,
                                          'LL01': 74,
                                          'LL00': 74,
                                          'LL0': 74,
                                          'LL1': 74,
                                          'LL2': 74,
                                          'LL3': 74,
                                      }),
                             Layout(
                                 'T1 T2 K A DF [CF B CIF CI C] D DI S ST P',
                                 'LL03 LL02 LL01 LL00 W2 [W2Prime CI L C W1Prime] W1 D LL0 LL1 LL2 LL3',
                                 front_aligns={
                                     'T2': Align.UPPER,
                                     'CI': Align.UPPER,
                                     'S': Align.LOWER,
                                 },
                                 rear_aligns={
                                     'LL03': Align.LOWER,
                                     'LL02': Align.UPPER,
                                     'LL01': Align.LOWER,
                                     'LL00': Align.UPPER,
                                     'C': Align.UPPER,
                                     'LL0': Align.LOWER,
                                     'LL1': Align.UPPER,
                                     'LL2': Align.LOWER,
                                     'LL3': Align.UPPER,
                                 }
                             ),
                             Style(Colors.BLACK, Colors.WHITE, sc_bg_colors={
                                 'C': Colors.FC_LIGHT_GREEN_BG,
                                 'CF': Colors.FC_LIGHT_GREEN_BG,
                                 'A': Colors.FC_LIGHT_BLUE_BG,
                                 'B': Colors.FC_LIGHT_BLUE_BG,
                                 'LL0': Colors.FC_LIGHT_GREEN_BG
                             }, font_family=FontFamilies.CMUBright))

    Graphoplex621 = Model('Graphoplex', '', '621',
                          Geometry((7740, 1070),  # 29cm (7733px) x 40cm (1066px)
                                   (100, 100),
                                   (6666, 80),  # 25cm (6666.7px) x 3.5mm (93px)
                                   Geometry.DEFAULT_TICK_WH,
                                   480,  # 1.8cm
                                   scale_h_front={
                                       'P': 70,
                                       'ST': 70,
                                       'K': 70,
                                       'L': 70,
                                   }),
                          Layout(
                              'P ST A [ B T1 S CI C ] D K L',  # 'P SRT A [ B T1 S CI C ] D K L'
                              ''
                          ), Styles.Graphoplex)


class GaugeMark:
    def __init__(self, sym, value, comment=None):
        self.sym = sym
        self.value = value
        self.comment = comment

    def draw(self, r, geom, style, y_off, sc, font, al, col=None, shift_adj=0, side=None):
        """
        :param ImageDraw.Draw r:
        :param Geometry geom:
        :param Style style:
        :param int y_off: y pos
        :param Scale sc:
        :param FreeTypeFont font:
        :param Align al: alignment
        :param str col: color
        :param Number shift_adj:
        :param Side side:
        """
        if not col:
            col = style.scale_fg_col(sc)
        x = sc.scale_to(self.value, geom.SL, shift_adj=shift_adj)
        scale_h = geom.scale_h(sc, side=side)
        scale_h_ratio = geom.scale_h_ratio(sc, side=side)
        tick_h = geom.tick_h(HMod.MED, h_ratio=scale_h_ratio)
        draw_tick(r, geom, col, y_off, scale_h, x, tick_h, al)
        sym_h = geom.tick_h(HMod.XL if al == Align.LOWER else HMod.MED, h_ratio=scale_h_ratio)
        draw_sym_al(r, geom, style, col, y_off, scale_h, self.sym, x, sym_h, font, al)


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


def gen_scale_band_bg(r, geom, y_off, sc, color, start_value=None, end_value=None):
    li = geom.li
    if start_value is None:
        start_value = sc.value_at_start()
    if end_value is None:
        end_value = sc.value_at_end()
    r.rectangle((li + sc.pos_of(start_value, geom), y_off,
                 li + sc.pos_of(end_value, geom), y_off + geom.scale_h(sc)),
                fill=color)


def gen_scale(r, geom, style, y_off, sc, al=None, overhang=None, side=None):
    """
    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param Style style:
    :param int y_off: y pos
    :param Scale sc:
    :param Align al: alignment
    :param float overhang: fraction of total width to overhang each side to label
    :param Side side:
    """

    if style.override_for(sc, 'hide', False):
        return

    if not overhang:
        overhang = 0.08 if sc.can_spiral() else 0.02

    scale_h = geom.scale_h(sc, side=side)
    scale_h_ratio = geom.scale_h_ratio(sc, side=side)

    # Place Index Symbols (Left and Right)
    italic = FontStyle.ITALIC
    f_lbl = style.font_for(FontSize.SC_LBL if scale_h > FontSize.SC_LBL.value * 1.5 else scale_h // 2)
    f_lbl_s = style.font_for(FontSize.N_XL if scale_h > FontSize.N_XL.value * 2 else scale_h // 2)
    f_xln = style.font_for(FontSize.N_XL, h_ratio=scale_h_ratio)
    f_lgn = style.font_for(FontSize.N_LG, h_ratio=scale_h_ratio)
    f_mdn = style.font_for(FontSize.N_MD, h_ratio=scale_h_ratio)
    f_smn = style.font_for(FontSize.N_SM, h_ratio=scale_h_ratio)
    f_mdn_i = style.font_for(FontSize.N_MD, font_style=italic, h_ratio=scale_h_ratio)

    if not al:
        al = sc.al
    li = geom.li
    scale_w = geom.SL
    if DEBUG:
        r.rectangle((li, y_off, li + scale_w, y_off + scale_h), outline='grey')

    sym_col = style.scale_fg_col(sc)
    bg_col = style.scale_bg_col(sc)
    dec_col = style.dec_color
    if bg_col:
        gen_scale_band_bg(r, geom, y_off, sc, bg_col)

    # Right
    f_lbl_r = f_lbl_s if len(sc.right_sym) > 6 or '^' in sc.right_sym else f_lbl
    (right_sym, _, _) = symbol_parts(sc.right_sym)
    w2, h2 = style.sym_dims(right_sym, f_lbl_r)
    y2 = (geom.SH - h2) / 2  # Ignore custom height/spacing for legend symbols
    x_right = (1 + overhang) * scale_w + w2 / 2
    draw_sym_al(r, geom, style, sym_col, y_off, scale_h, sc.right_sym, x_right, y2, f_lbl_r, al)

    # Left
    (left_sym, _, _) = symbol_parts(sc.left_sym)
    w1, h1 = style.sym_dims(left_sym, f_lbl)
    y1 = (geom.SH - h1) / 2  # Ignore custom height/spacing for legend symbols
    x_left = (0 - overhang) * scale_w - w1 / 2
    draw_sym_al(r, geom, style, sym_col, y_off, scale_h, sc.left_sym, x_left, y1, f_lbl, al)

    # Special Symbols for S, and T
    sc_alt = None
    if sc == Scales.S:
        sc_alt = Scales.CoS
    elif sc == Scales.CoS:
        sc_alt = Scales.S
    elif sc == Scales.T:
        sc_alt = Scales.CoT

    if sc_alt:
        draw_sym_al(r, geom, style, sc_alt.col, y_off, scale_h, sc_alt.left_sym, x_left - style.sym_width('__', f_lbl),
                    y2, f_lbl, al)
        draw_sym_al(r, geom, style, sc_alt.col, y_off, scale_h, sc_alt.right_sym, x_right, y2 - h2 * 0.8, f_lbl_r, al)
    elif sc == Scales.ST:
        draw_sym_al(r, geom, style, sym_col, y_off, scale_h, '∡sin 0.01x°', x_right, y2 - h2 * 0.8, f_lbl_r, al)

    th_med = geom.tick_h(HMod.MED)
    th_xl = geom.tick_h(HMod.XL)
    th_lg = geom.tick_h(HMod.LG)
    th_sm = geom.tick_h(HMod.SM)
    th_xs = geom.tick_h(HMod.XS)
    th_dot = geom.tick_h(HMod.DOT)

    # Numeral offsets for Sin/Cosine and Tan/Cotangent self-folded scales:
    sym_off_rf = -1.4 / 2
    sym_off_lf = 1.2 / 2

    # Tick Placement (the bulk!)
    fonts_lbl = (f_lbl, None, None)
    fonts2 = (f_lbl, f_mdn, None)
    fonts_xl = (f_xln, None, None)
    ths1 = (th_med, th_xl, th_sm, th_xs)
    ths2 = (th_med, th_xl, th_xs, th_xs)
    ths3 = (th_med, th_sm, th_sm, th_xs)
    ths4 = (th_med, th_xl, th_sm, th_dot)
    fonts_no = (None, None, None)
    if (sc.scaler in {Scalers.Base, Scalers.Inverse}) and sc.shift == 0:  # C/D and CI/DI
        sf = 100
        fp1, fp2, fp4, fpe = (fp * sf for fp in (1, 2, 4, 10))
        grad_pat(r, geom, style, y_off, sc, al,
                 fp1, fp2, sf, (sf, 10, 5, 1), ths3, fonts2, True)
        grad_pat(r, geom, style, y_off, sc, al,
                 fp2, fp4, sf, (sf, sf // 2, 10, 2), ths1, fonts_lbl, False)
        grad_pat(r, geom, style, y_off, sc, al,
                 fp4, fpe + 1, sf, (sf, sf // 2, 10, 5), ths1, fonts_lbl, True)

        # Gauge Points
        Marks.pi.draw(r, geom, style, y_off, sc, f_lbl, al, col=sym_col, side=side)

        if y_off < geom.side_h + geom.oY:
            Marks.deg_per_rad.draw(r, geom, style, y_off, sc, f_lbl, al, col=sym_col, side=side)
            Marks.tau.draw(r, geom, style, y_off, sc, f_lbl, al, col=sym_col, side=side)

    elif sc.scaler == Scalers.Square:
        sf = 100
        for b in (10 ** n for n in range(0, 2)):
            fp1, fp2, fp3, fpe = (fp * b * sf for fp in (1, 2, 5, 10))
            steps = (sf * b, (sf * b) // 2, (sf * b) // 10)
            grad_pat(r, geom, style, y_off, sc, al,
                     fp1, fp2, sf, steps + (2 * b,), ths1, fonts_lbl, True)
            grad_pat(r, geom, style, y_off, sc, al,
                     fp2, fp3, sf, steps + (5 * b,), ths1, fonts_lbl, True)
            grad_pat(r, geom, style, y_off, sc, al,
                     fp3, fpe + 1, sf, steps + (steps[-1],), ths2, fonts_lbl, True)

        # Gauge Points
        Marks.pi.draw(r, geom, style, y_off, sc, f_lbl, al, side=side)
        Marks.pi.draw(r, geom, style, y_off, sc, f_lbl, al, shift_adj=0.5, side=side)

    elif sc == Scales.K:
        sf = 100
        for b in (10 ** n for n in range(0, 3)):
            fp1, fp2, fp3, fpe = (fp * b * sf for fp in (1, 3, 6, 10))
            steps = (sf * b, (sf * b) // 2, (sf * b) // 10)
            grad_pat(r, geom, style, y_off, sc, al, fp1, fp2, sf, steps + (5 * b,), ths1, fonts_xl, True)
            grad_pat(r, geom, style, y_off, sc, al, fp2, fp3, sf, steps + (10 * b,), ths2, fonts_xl, True)
            grad_pat(r, geom, style, y_off, sc, al, fp3, fpe + 1, sf, steps + (20 * b,), ths2, fonts_xl, True)

    elif sc == Scales.R1:
        sf = 1000
        fp1, fp2, fpe = (int(fp * sf) for fp in (1, 2, 3.17))
        grad_pat(r, geom, style, y_off, sc, al, fp1, fp2, sf, (sf // 10, sf // 20, sf // 100, sf // 200), ths1, fonts_no, True)
        grad_pat(r, geom, style, y_off, sc, al, fp2, fpe + 1, sf, (sf, sf // 10, sf // 20, sf // 100), (th_med, th_med, th_sm, th_xs), fonts2, True)

        # 1-10 Labels
        for x in range(1, 2):
            draw_numeral(r, geom, style, sym_col, y_off, scale_h, x, sc.pos_of(x, geom), th_med, f_lbl, al)
        # 1.1-1.9 Labels
        for x in range(11, 20):
            draw_numeral(r, geom, style, sym_col, y_off, scale_h, last_digit_of(x), sc.pos_of(x / 10, geom), th_med,
                         f_lgn, al)

        # Marks.sqrt_ten.draw(r, geom, style, y_off, sc, f_lgn, al, sym_col, side=side)

    elif sc in {Scales.W1, Scales.W1Prime}:
        sc.grad_pat_divided(r, geom, style, y_off, al, [2])
        Marks.sqrt_ten.draw(r, geom, style, y_off, sc, f_lgn, al, sym_col, side=side)

    elif sc in {Scales.W2, Scales.W2Prime}:
        sc.grad_pat_divided(r, geom, style, y_off, al, [5])
        Marks.sqrt_ten.draw(r, geom, style, y_off, sc, f_lgn, al, sym_col, side=side)

    elif sc == Scales.R2:
        sf = 1000
        fp1, fp2, fpe = (int(fp * sf) for fp in (3.16, 5, 10))
        grad_pat(r, geom, style, y_off, sc, al, fp1, fp2, sf, (sf, sf // 10, sf // 20, sf // 100), ths3, fonts2, True)
        grad_pat(r, geom, style, y_off, sc, al, fp2, fpe + 1, sf, (sf, sf // 2, sf // 10, sf // 50), ths1, fonts_lbl, True)

        # Marks.sqrt_ten.draw(r, geom, style, y_off, sc, f_lgn, al, sym_col, side=side)

    elif sc == Scales.H1:
        draw_numeral(r, geom, style, sym_col, y_off, scale_h, 1.005, sc.pos_of(1.005, geom), geom.tick_h(HMod.XL),
                     f_lgn, al)
        sc.grad_pat_divided(r, geom, style, y_off, al, [1.03, 1.1])

    elif sc == Scales.H2:
        draw_numeral(r, geom, style, sym_col, y_off, scale_h, 1.5, sc.pos_of(1.5, geom), geom.tick_h(HMod.XL), f_lgn,
                     al)
        sc.grad_pat_divided(r, geom, style, y_off, al, [4])

    elif (sc.scaler == Scalers.Base and sc.shift == pi_fold_shift) or sc == Scales.CIF:  # CF/DF/CIF
        is_cif = sc == Scales.CIF
        sf = 1000
        fp1 = 310 if is_cif else 314
        i1 = round(sf / TEN)
        fp2, fp3, fp4 = (fp * i1 for fp in (4, 10, 20))
        fpe = 3200 if is_cif else fp1 * TEN
        grad_pat(r, geom, style, y_off, sc, al, fp1, fp2, sf, (i1, i1 // 2, 10, 2), ths1, fonts_lbl, True)
        grad_pat(r, geom, style, y_off, sc, al, fp2, fp3, sf, (i1, i1 // 2, 10, 5), ths1, fonts_lbl, True)
        grad_pat(r, geom, style, y_off, sc, al, fp3, fp4, sf, (sf, i1, i1 // 2, i1 // 10), ths3, fonts2, True)
        grad_pat(r, geom, style, y_off, sc, al, fp4, fpe + 1, sf, (sf, sf // 2, 100, 20), ths1, fonts_lbl, True)

        # Gauge Points
        Marks.pi.draw(r, geom, style, y_off, sc, f_lbl, al, side=side)
        Marks.pi.draw(r, geom, style, y_off, sc, f_lbl, al, shift_adj=-1, side=side)

    elif sc == Scales.L:
        sf = 100
        grad_pat(r, geom, style, y_off, sc, al, 0, TEN * sf + 1, sf,
                 (100, 50, 10, 2), (th_lg, th_xl, th_med, th_xs), fonts_no, True)
        # Labels
        for x in range(0, 11):
            if x == 0:
                draw_numeral(r, geom, style, sym_col, y_off, scale_h, x, sc.pos_of(x, geom), th_med, f_lbl, al)
            if x == 10:
                draw_numeral(r, geom, style, sym_col, y_off, scale_h, 1, sc.pos_of(x, geom), th_med, f_lbl, al)
            elif x in range(1, 10):
                draw_numeral(r, geom, style, sym_col, y_off, scale_h, x / 10, sc.pos_of(x, geom), th_med, f_lbl, al)

    elif sc == Scales.Ln:
        grad_pat_auto(r, geom, style, y_off, sc, al, include_last=True)

    elif sc.scaler in {Scalers.Sin, Scalers.CoSin}:
        sf = 100
        fp1, fp2, fp3, fp4, fp5, fpe = (int(fp * sf) for fp in (5.7, 20, 30, 60, 80, 90))
        grad_pat(r, geom, style, y_off, sc, al, fp1, fp2, sf, (sf, sf // 2, sf // 10, sf // 10), (th_xl, th_sm, th_xs, th_xs), fonts_no, True)
        grad_pat(r, geom, style, y_off, sc, al, fp2, fp3, sf, (sf * 5, sf, sf // 2, sf // 5), (th_xl, th_sm, th_xs, th_xs), fonts_no, True)
        grad_pat(r, geom, style, y_off, sc, al, fp3, fp4, sf, (sf * 10, sf * 5, sf, sf // 2), (th_xl, th_xl, th_sm, th_xs), fonts_no, True)
        grad_pat(r, geom, style, y_off, sc, al, fp4, fp5, sf, (sf * 10, sf * 5, sf, sf), (th_xl, th_sm, th_xs, th_xs), fonts_no, True)
        grad_pat(r, geom, style, y_off, sc, al, fp5, fpe + 1, sf, (sf * 10, sf * 5, sf * 5, sf * 5), (th_med, th_sm, th_xs, th_xs), fonts_no, True)

        # Degree Labels
        for x in range(6, 16):
            x_coord = sc.pos_of(x, geom) + sym_off_lf * style.sym_width(str(x), f_mdn_i)
            draw_numeral(r, geom, style, sym_col, y_off, scale_h, x, x_coord, th_med, f_mdn, al)
            xi = angle_opp(x)
            x_coord_opp = sc.pos_of(x, geom) + sym_off_rf * style.sym_width(str(xi), f_mdn_i)
            draw_numeral(r, geom, style, dec_col, y_off, scale_h, xi, x_coord_opp, th_med, f_mdn_i, al)

        for x in range(16, 20):
            x_coord = sc.pos_of(x, geom) + sym_off_lf * style.sym_width(str(x), f_mdn_i)
            draw_numeral(r, geom, style, sym_col, y_off, scale_h, x, x_coord, th_med, f_mdn, al)

        for x in range(20, 71, 5):
            if (x % 5 == 0 and x < 40) or last_digit_of(x) == 0:
                x_coord = sc.pos_of(x, geom) + sym_off_lf * style.sym_width(str(x), f_mdn_i)
                draw_numeral(r, geom, style, sym_col, y_off, scale_h, x, x_coord, th_med, f_mdn, al)
                if x != 20:
                    xi = angle_opp(x)
                    x_coord = sc.pos_of(x, geom) + sym_off_rf * style.sym_width(str(xi), f_mdn_i)
                    if xi != 40:
                        draw_numeral(r, geom, style, dec_col, y_off, scale_h, xi, x_coord, th_med, f_mdn_i, al)
                    elif xi == 40:
                        draw_numeral(r, geom, style, dec_col, y_off + 11, scale_h, 40, x_coord, th_med, f_mdn_i, al)

        draw_numeral(r, geom, style, sym_col, y_off, scale_h, DEG_RIGHT_ANGLE, scale_w, th_med, f_lgn, al)

    elif sc in {Scales.T, Scales.T1}:
        sf = 100
        fp1, fp2, fp3, fpe = (int(fp * sf) for fp in (5.7, 10, 25, 45))
        grad_pat(r, geom, style, y_off, sc, al, fp1, fp2, sf, (sf, sf // 2, sf // 10, sf // 20), (th_xl, th_xl, th_sm, th_xs), fonts_no, True)
        grad_pat(r, geom, style, y_off, sc, al, fp2, fp3, sf, (sf, sf // 2, sf // 10, sf // 10), (th_xl, th_sm, th_xs, th_xs), fonts_no, True)
        grad_pat(r, geom, style, y_off, sc, al, fp3, fpe + 1, sf, (sf * 5, sf, sf // 5, sf // 5), (th_xl, th_med, th_xs, th_xs), fonts_no, True)

        # Degree Labels
        f = geom.STH * 1.1
        for x in range(6, 16):
            x_coord = sc.pos_of(x, geom) + sym_off_lf * style.sym_width(str(x), f_mdn_i)
            draw_numeral(r, geom, style, sym_col, y_off, scale_h, x, x_coord, f, f_mdn, al)
            xi = angle_opp(x)
            x_coord_opp = sc.pos_of(x, geom) + sym_off_rf * style.sym_width(str(xi), f_mdn_i)
            draw_numeral(r, geom, style, dec_col, y_off, scale_h, xi, x_coord_opp, f, f_mdn_i, al)

        for x in range(16, 21):
            x_coord = sc.pos_of(x, geom) + sym_off_lf * style.sym_width(str(x), f_mdn_i)
            draw_numeral(r, geom, style, sym_col, y_off, scale_h, x, x_coord, f, f_mdn, al)

        for x in range(25, 41, 5):
            x_coord = sc.pos_of(x, geom) + sym_off_lf * style.sym_width(str(x), f_mdn_i)
            draw_numeral(r, geom, style, sym_col, y_off, scale_h, x, x_coord, f, f_mdn, al)
            xi = angle_opp(x)
            x_coord_opp = sc.pos_of(x, geom) + sym_off_rf * style.sym_width(str(xi), f_mdn_i)
            draw_numeral(r, geom, style, dec_col, y_off, scale_h, xi, x_coord_opp,
                         f, f_mdn_i, al)

        draw_numeral(r, geom, style, sym_col, y_off, scale_h, 45, scale_w, f, f_lgn, al)

    elif sc == Scales.T2:
        # Ticks
        sf = 100
        fp1, fp2, fpe = (int(fp * sf) for fp in (45, 75, 84.5))
        grad_pat(r, geom, style, y_off, sc, al, fp1, fp2, sf, (sf * 5, sf, sf // 2, sf // 10), ths4, fonts_xl, False)
        grad_pat(r, geom, style, y_off, sc, al, fp2, fpe, sf, (sf * 5, sf, sf // 2, sf // 20), ths4, fonts_xl, False)

    elif sc == Scales.ST:
        # Ticks
        sf = 100
        fp1, fp2, fp3, fp4, fpe = (int(fp * sf) for fp in (0.58, 1, 2, 4, 5.8))
        grad_pat(r, geom, style, y_off, sc, al, fp1, fp2, sf, (sf, sf // 20, sf // 100, sf // 100), ths1, fonts_no, True)
        grad_pat(r, geom, style, y_off, sc, al, fp2, fp3, sf, (sf // 10, sf // 10, sf // 20, sf // 100), ths1, fonts_no, True)
        grad_pat(r, geom, style, y_off, sc, al, fp3, fp4, sf, (sf // 2, sf // 2, sf // 10, sf // 50), ths1, fonts_no, True)
        grad_pat(r, geom, style, y_off, sc, al, fp4, fpe + 1, sf, (sf, sf, sf // 10, sf // 20), ths1, fonts_no, True)

        for x in range(570, 1000):
            if x % 5 == 0 and last_digit_of(x) != 0:
                draw_tick(r, geom, sym_col, y_off, scale_h, sc.pos_of(x / 1000, geom), geom.tick_h(HMod.XS), al)

        # Degree Labels
        draw_sym_al(r, geom, style, sym_col, y_off, scale_h, '1°', sc.pos_of(1, geom), th_med, f_lbl, al)
        for x in range(6, 10):
            x_value = x / 10
            draw_numeral(r, geom, style, sym_col, y_off, scale_h, x_value, sc.pos_of(x_value, geom), th_med, f_lbl, al)
        for x in range(1, 4):
            x_value = x + 0.5
            draw_numeral(r, geom, style, sym_col, y_off, scale_h, x_value, sc.pos_of(x_value, geom), th_med, f_lbl, al)
        for x in range(2, 6):
            draw_numeral(r, geom, style, sym_col, y_off, scale_h, x, sc.pos_of(x, geom), th_med, f_lbl, al)

    elif sc == Scales.SRT:
        grad_pat_auto(r, geom, style, y_off, sc, al)

    elif sc == Scales.P:
        # Labels
        label_h = geom.tick_h(HMod.MED)
        font_s = f_smn
        for x in [0.995]:
            draw_numeral(r, geom, style, sym_col, y_off, scale_h, x, sc.pos_of(x, geom), label_h, font_s, al)
        sc.grad_pat_divided(r, geom, style, y_off, al, [0.3, 0.7, 0.9, 0.98],
                            start_value=0.1, end_value=.995)

    elif sc == Scales.Sh1:
        sc.grad_pat_divided(r, geom, style, y_off, al, [0.2, 0.4])

    elif sc == Scales.Sh2:
        grad_pat_auto(r, geom, style, y_off, sc, al, include_last=True)

    elif sc == Scales.Ch1:
        sc.grad_pat_divided(r, geom, style, y_off, al, [1, 2], start_value=0.01)

    elif sc == Scales.Th:
        sc.grad_pat_divided(r, geom, style, y_off, al, dividers=[0.2, 0.4, 1], end_value=3)
        # Labels
        label_h = geom.tick_h(HMod.MED)
        for x in [1, 1.5, 2, 3]:
            draw_numeral(r, geom, style, sym_col, y_off, scale_h, x, sc.pos_of(x, geom), label_h, f_smn, al)

    elif sc == Scales.Chi:
        grad_pat_auto(r, geom, style, y_off, sc, al, include_last=True)
        Marks.pi_half.draw(r, geom, style, y_off, sc, f_lgn, al, sym_col, side=side)

    elif sc == Scales.Theta:
        grad_pat_auto(r, geom, style, y_off, sc, al, include_last=True)

    elif sc == Scales.f_x:
        sc.grad_pat_divided(r, geom, style, y_off, al, [0.2, 0.5, 1])

    elif sc == Scales.L_r:
        sc.grad_pat_divided(r, geom, style, y_off, al, [0.05, 0.1, 0.2, 0.5, 1, 2],
                            start_value=0.025, end_value=2.55)

    elif sc == Scales.LL0:
        sc.grad_pat_divided(r, geom, style, y_off, al, [1.002, 1.005],
                            start_value=1.00095, end_value=1.0105)

    elif sc == Scales.LL1:
        sc.grad_pat_divided(r, geom, style, y_off, al, [1.02, 1.05],
                            start_value=1.0095, end_value=1.11)

    elif sc == Scales.LL2:
        sc.grad_pat_divided(r, geom, style, y_off, al, [1.2, 2],
                            start_value=1.1, end_value=3)
        Marks.e.draw(r, geom, style, y_off, sc, f_lgn, al, sym_col, side=side)

    elif sc == Scales.LL3:
        sc.grad_pat_divided(r, geom, style, y_off, al, [10, 50, 100, 1000, 10000],
                            start_value=2.5, end_value=60000)
        Marks.e.draw(r, geom, style, y_off, sc, f_lgn, al, sym_col, side=side)

    elif sc == Scales.LL03:
        sc.grad_pat_divided(r, geom, style, y_off, al, [0.001, 0.01, 0.1],
                            start_value=0.0001, end_value=0.39)
        Marks.inv_e.draw(r, geom, style, y_off, sc, f_smn, al, sym_col, side=side)

    elif sc == Scales.LL02:
        sc.grad_pat_divided(r, geom, style, y_off, al, [0.75],
                            start_value=0.35, end_value=0.91)
        Marks.inv_e.draw(r, geom, style, y_off, sc, f_smn, al, sym_col, side=side)

    elif sc == Scales.LL01:
        sc.grad_pat_divided(r, geom, style, y_off, al, [0.95, 0.98],
                            start_value=0.9, end_value=0.9906)

    elif sc == Scales.LL00:
        sc.grad_pat_divided(r, geom, style, y_off, al, [0.998],
                            start_value=0.989, end_value=0.9991)

    else:
        sc.grad_pat_divided(r, geom, style, y_off, al, None)


def first_digit_of(x) -> int:
    """First numeral in the digital representation of a number."""
    return int(str(x)[0])


def last_digit_of(x) -> int:
    """Last numeral in the digital representation of a number."""
    if int(x) == x:
        x = int(x)
    return int(str(x)[-1])


def sig_digit_of(num):
    """When only one digit will fit on a major scale's numerals, pick the most significant."""
    if not (num > 0 and math.log10(num).is_integer()):
        if num % 10 == 0:
            num = first_digit_of(num)
        elif num < 1:
            num = last_digit_of(num)
    else:
        num = first_digit_of(num)
    return num


# ----------------------4. Line Drawing Functions----------------------------

# These functions are unfortunately difficult to modify,
# since I built them with specific numbers rather than variables


def draw_borders(r, geom, y0, side, color=Colors.BLACK):
    """
    Place initial borders around scales
    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param y0: vertical offset
    :param Side side:
    :param string|tuple color:
    """

    # Main Frame

    total_w = geom.total_w
    side_h = geom.side_h
    stator_h = geom.stator_h
    y_offsets = [0, stator_h - 1, side_h - stator_h - 1, side_h - 2]
    for horizontal_y in [y0 + y_off for y_off in y_offsets]:
        r.rectangle((geom.oX, horizontal_y, total_w - geom.oX, horizontal_y + 2), outline=color)
    for vertical_x in [geom.oX, total_w - geom.oX]:
        r.rectangle((vertical_x, y0, vertical_x + 2, y0 + side_h), fill=color)

    # Top Stator Cut-outs
    # if side == SlideSide.FRONT:
    y_start = y0
    if side == Side.REAR:
        y_start += side_h - stator_h
    y_end = stator_h + y_start
    half_stock_height = stator_h // 2
    for horizontal_x in [half_stock_height + geom.oX, (total_w - half_stock_height) - geom.oX]:
        r.rectangle((horizontal_x, y_start, horizontal_x + 2, y_end), fill=color)


def draw_metal_cutoffs(r, geom, y0, side):
    """
    Use to temporarily view the metal bracket locations
    :param ImageDraw.Draw r:
    :param Geometry geom:
    :param int y0: vertical offset
    :param Side side:
    """

    # Initial Boundary verticals
    cutoff_w = geom.cutoff_w
    verticals = [cutoff_w + geom.oX, geom.total_w - cutoff_w - geom.oX]
    for i, start in enumerate(verticals):
        r.rectangle((start - 1, y0, start + 1, y0 + i), Colors.CUTOFF)

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

    side_h = geom.side_h
    total_w = geom.total_w
    half_cutoff_w = cutoff_w // 2
    b = 30  # offset of metal from boundary
    # Create the left piece using format: (x1,x2,y1,y2)
    x_left = b + geom.oX
    x_mid = x_left + half_cutoff_w
    x_right = cutoff_w - b + geom.oX
    y_top = b + y0
    y_mid = y_top + side_h - geom.stator_h
    y_bottom = side_h - b + y0
    coords = [[x_mid, x_right, y_top, y_top],  # 1
              [x_left, x_mid, y_mid, y_mid],  # 2
              [x_left, x_right, y_bottom, y_bottom],  # 3
              [x_mid, x_mid, y_top, y_mid],  # 4
              [x_left, x_left, y_mid, y_bottom],  # 5
              [x_right, x_right, y_top, y_bottom]]  # 6

    # Symmetrically create the right piece
    for i in range(0, len(coords)):
        (x1, x2, y1, y2) = coords[i]
        coords.append([total_w - x2, total_w - x1, y1, y2])

    # Transfer coords to points for printing (yeah I know it's dumb)
    points = coords
    # If backside, first apply a vertical reflection
    if side == Side.REAR:
        for coord in coords:
            (x1, x2, y1, y2) = coord
            mid_y = 2 * y0 + side_h
            points.append([x1, x2, mid_y - y2, mid_y - y1])
    for i in range(0, 12):
        (x1, x2, y1, y2) = points[i]
        r.rectangle((x1 - 1, y1 - 1, x2 + 1, y2 + 1), fill=Colors.CUTOFF2)


class Mode(Enum):
    RENDER = 'render'
    DIAGNOSTIC = 'diagnostic'
    STICKERPRINT = 'stickerprint'


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
    img_renderer.rectangle((x0, y0, x0 + dx, y0 + dy), outline=Colors.CUT)


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
    col = Colors.CUT
    r.line((x1 - arm_width, y1, x1 + arm_width, y1), col)
    r.line((x1 - arm_width, y2, x1 + arm_width, y2), col)
    r.line((x2 - arm_width, y1, x2 + arm_width, y1), col)
    r.line((x2 - arm_width, y2, x2 + arm_width, y2), col)
    # vertical cross arms at 4 corners:
    r.line((x1, y1 - arm_width, x1, y1 + arm_width), col)
    r.line((x1, y2 - arm_width, x1, y2 + arm_width), col)
    r.line((x2, y1 - arm_width, x2, y1 + arm_width), col)
    r.line((x2, y2 - arm_width, x2, y2 + arm_width), col)


def transcribe(src_img, dst_img, src_x, src_y, size_x, size_y, target_x, target_y):
    """
    Transfer a pixel rectangle from a SOURCE (for rendering) to DESTINATION (for stickerprint)
    :param src_img: SOURCE of pixels
    :param dst_img: DESTINATION of pixels
    :param src_x: First corner of SOURCE
    :param src_y: First corner of SOURCE
    :param size_x: Width of SOURCE chunk to transcribe
    :param size_y: Length of SOURCE chunk to transcribe
    :param target_x: Target corner of DESTINATION
    :param target_y: Target corner of DESTINATION
    """

    src_box = src_img.crop((src_x, src_y, src_x + size_x, src_y + size_y))
    dst_img.paste(src_box, (target_x, target_y))


def save_png(img_to_save: Image, basename: str, output_suffix=None):
    output_filename = f"{basename}{'.' + output_suffix if output_suffix else ''}.png"
    from os.path import abspath
    output_full_path = abspath(output_filename)
    img_to_save.save(output_full_path, 'PNG')
    print(f'Result saved to: file://{output_full_path}')


def main():
    """CLI processor for rendering the models in various modes."""
    import argparse
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--mode',
                             choices=[m.value for m in Mode],
                             help='What to render')
    args_parser.add_argument('--model',
                             choices=keys_of(Models),
                             default='Demo',
                             help='Which sliderule model')
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
    render_mode: Mode = next(mode for mode in Mode if mode.value == cli_args.mode)
    model_name = cli_args.model
    model: Model = getattr(Models, model_name)
    output_suffix = cli_args.suffix or ('test' if cli_args.test else None)
    render_cutoffs = cli_args.cutoffs
    global DEBUG
    DEBUG = cli_args.debug

    start_time = time.time()

    geom = model.geometry
    sliderule_img = Image.new('RGB', (geom.total_w, geom.print_height), model.style.bg)
    if render_mode in {Mode.RENDER, Mode.STICKERPRINT}:
        render_sliderule_mode(model, sliderule_img, render_mode, render_cutoffs=render_cutoffs)

    if render_mode == Mode.RENDER:
        save_png(sliderule_img, f'{model_name}.SlideRuleScales', output_suffix)

    if render_mode == Mode.DIAGNOSTIC:
        render_diagnostic_mode(model, model_name, output_suffix)

    if render_mode == Mode.STICKERPRINT:
        render_stickerprint_mode(model, model_name, output_suffix, sliderule_img)

    print(f'The program took {round(time.time() - start_time, 2)} seconds to run')


def render_sliderule_mode(model: Model, sliderule_img, render_mode: Mode, render_cutoffs: bool = False):
    r = ImageDraw.Draw(sliderule_img)
    geom = model.geometry
    layout = model.layout
    y_rear_start = geom.side_h + 2 * geom.oY
    if render_mode == Mode.RENDER:
        draw_borders(r, geom, geom.oY, Side.FRONT)
        if render_cutoffs:
            draw_metal_cutoffs(r, geom, geom.oY, Side.FRONT)
        draw_borders(r, geom, y_rear_start, Side.REAR)
        if render_cutoffs:
            draw_metal_cutoffs(r, geom, y_rear_start, Side.REAR)
    # Front Scale
    # Titling
    style = model.style
    f_lbl = style.font_for(FontSize.SC_LBL)
    side_w = geom.side_w
    li = geom.li
    y_off = y_side_start = geom.oY
    if model == Models.Demo:
        upper = Align.UPPER
        y_off_titling = 25 + y_off
        title_col = Colors.RED
        draw_sym_al(r, geom, style, title_col, y_off_titling, 0, model.name, side_w * 1 / 4 - li, 0, f_lbl, upper)
        draw_sym_al(r, geom, style, title_col, y_off_titling, 0, model.subtitle, side_w * 2 / 4 - li + geom.oX, 0,
                    f_lbl,
                    upper)
        draw_sym_al(r, geom, style, title_col, y_off_titling, 0, model.brand, side_w * 3 / 4 - li, 0, f_lbl, upper)
        y_off = y_off_titling + f_lbl.size
    # Scales
    for side in [Side.FRONT, Side.REAR]:
        for part, top in [(RulePart.STATOR, True), (RulePart.SLIDE, True), (RulePart.STATOR, False)]:
            part_scales = layout.scales_at(side, part, top)
            last_i = len(part_scales) - 1
            for i, sc in enumerate(part_scales):
                scale_h = geom.scale_h(sc, side)
                scale_al = layout.scale_al(sc, side, top)
                # Last scale per top part, align to bottom of part:
                if top and i == last_i and scale_al == Align.LOWER:
                    y_off = y_side_start + geom.stator_h + (geom.slide_h if part == RulePart.SLIDE else 0) - scale_h
                if i == 0 and scale_al == Align.UPPER:  # First scale, aligned to top edge
                    y_off = y_side_start + geom.stator_h + (geom.slide_h if part == RulePart.STATOR else 0)
                gen_scale(r, geom, style, y_off, sc, al=scale_al, side=side)
                y_off += scale_h

        y_off = y_rear_start
        y_side_start = y_rear_start
        y_off += geom.top_margin


def render_stickerprint_mode(model, model_name, output_suffix, sliderule_img):
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
    geom = model.geometry
    scale_w = 6500
    geom_s = Geometry(
        (scale_w, 1600),
        Geometry.NO_MARGINS,
        (scale_w, Geometry.SH),
        Geometry.DEFAULT_TICK_WH,
        640
    )
    style = model.style
    scale_h = geom_s.SH
    total_w = scale_w + 2 * o_x2
    total_h = 5075
    stickerprint_img = Image.new('RGB', (total_w, total_h), style.bg)
    r = ImageDraw.Draw(stickerprint_img)
    # fsUM,MM,LM:
    y = 0
    y += o_y2 + o_a
    x_off = 750
    x_left = geom.oX + x_off
    slide_h = geom_s.slide_h
    stator_h = geom_s.stator_h
    transcribe(sliderule_img, stickerprint_img, x_left, geom.oY, scale_w, stator_h, o_x2, y)
    extend(stickerprint_img, geom_s, y + stator_h - 1, BleedDir.DOWN, ext)
    draw_corners(r, o_x2, y - o_a, o_x2 + scale_w, y + stator_h)
    y += stator_h + o_a
    transcribe(sliderule_img, stickerprint_img, x_left, geom.oY + stator_h + 1, scale_w, slide_h, o_x2, y)
    extend(stickerprint_img, geom_s, y + 1, BleedDir.UP, ext)
    extend(stickerprint_img, geom_s, y + slide_h - 1, BleedDir.DOWN, ext)
    draw_corners(r, o_x2, y, o_x2 + scale_w, y + slide_h)
    y += slide_h + o_a
    transcribe(sliderule_img, stickerprint_img, x_left, geom.oY + geom.side_h - stator_h, scale_w, stator_h, o_x2,
               y)
    extend(stickerprint_img, geom_s, y + 1, BleedDir.UP, ext)
    extend(stickerprint_img, geom_s, y + stator_h - 1, BleedDir.DOWN, ext)
    draw_corners(r, o_x2, y, o_x2 + scale_w, y + stator_h + o_a)
    # bsUM,MM,LM:
    y += stator_h + o_a + o_a + o_a
    y_start = geom.oY + geom.side_h + geom.oY
    transcribe(sliderule_img, stickerprint_img, x_left, y_start, scale_w, stator_h, o_x2, y)
    extend(stickerprint_img, geom_s, y + stator_h - 1, BleedDir.DOWN, ext)
    draw_corners(r, o_x2, y - o_a, o_x2 + scale_w, y + stator_h)
    y += stator_h + o_a
    transcribe(sliderule_img, stickerprint_img, x_left, y_start + stator_h + 1 - 3, scale_w, slide_h, o_x2, y)
    extend(stickerprint_img, geom_s, y + 1, BleedDir.UP, ext)
    extend(stickerprint_img, geom_s, y + slide_h - 1, BleedDir.DOWN, ext)
    draw_corners(r, o_x2, y, o_x2 + scale_w, y + slide_h)
    y += slide_h + o_a
    transcribe(sliderule_img, stickerprint_img, x_left, y_start + geom_s.side_h - stator_h, scale_w, stator_h, o_x2,
               y)
    extend(stickerprint_img, geom_s, y + 1, BleedDir.UP, ext)
    extend(stickerprint_img, geom_s, y + stator_h - 1, BleedDir.DOWN, ext)
    y_bottom = y + stator_h + o_a
    draw_corners(r, o_x2, y, o_x2 + scale_w, y_bottom)
    y_b = y_bottom + 20
    box_w = 510
    boxes = [
        [o_a, y_b,
         box_w + o_a, stator_h + o_a],
        [box_w + 3 * o_a, y_b,
         x_off + o_a, slide_h],
        [box_w + x_off + 5 * o_a, y_b,
         x_off + o_a, stator_h + o_a]
    ]
    for box in boxes:
        (x0, y0, dx, dy) = box
        draw_box(r, x0, y0, dx, dy)
        draw_box(r, x0, y0 + slide_h + o_a, dx, dy)

        x0 = round(2 * (6.5 * o_a + box_w + 2 * x_off) - x0 - dx)

        draw_box(r, x0, y0, dx, dy)
        draw_box(r, x0, y0 + slide_h + o_a, dx, dy)
    points = [
        [2 * o_a + 120, y_b + o_a + scale_h],
        [6 * o_a + box_w + x_off + 2 * scale_h, y_b + scale_h],
        [6 * o_a + box_w + x_off + scale_h, y_b + 2 * scale_h],

        [2 * o_a + 120, y_b + slide_h + o_a + scale_h],
        [6 * o_a + box_w + x_off + scale_h, y_b + 640 + o_a + o_a + 2 * scale_h],
        [6 * o_a + box_w + x_off + 2 * scale_h, y_b + 640 + o_a + o_a + scale_h]
    ]
    hole_r = 34  # (2.5mm diameter screw holes) = math.ceil(0.25 * Geometry.PixelsPerCM / 2)
    for point in points:
        (p_x, p_y) = point
        r.ellipse((p_x - hole_r, p_y - hole_r,
                   p_x + hole_r, p_y + hole_r),
                  fill=style.bg,
                  outline=Colors.CUT)

        p_x = round(2 * (6.5 * o_a + box_w + 2 * x_off) - p_x)

        r.ellipse((p_x - hole_r, p_y - hole_r,
                   p_x + hole_r, p_y + hole_r),
                  fill=style.bg,
                  outline=Colors.CUT)
    save_png(stickerprint_img, f'{model_name}.StickerCut', output_suffix)


def render_diagnostic_mode(model: Model, model_name: str, output_suffix: str):
    """
    Diagnostic mode, rendering scales independently.
    Works as a test of tick marks, labeling, and layout. Also, regressions.
    If you're reading this, you're a real one
    +5 brownie points to you
    """
    scale_h = Geometry.SH
    k = 120 + scale_h
    is_demo = model == Models.Demo
    layout = model.layout
    style = model.style
    upper = Align.UPPER
    sh_with_margins = scale_h + (40 if is_demo else 10)
    scale_names = ['A', 'B', 'C', 'D',
                   'K', 'R1', 'R2', 'CI',
                   'DI', 'CF', 'DF', 'CIF', 'L',
                   'S', 'T', 'ST']
    for sc_name in layout.scale_names_in_order():
        if sc_name not in scale_names:
            scale_names.append(sc_name)
    total_h = k + (len(scale_names) + 1) * sh_with_margins + scale_h
    geom_d = Geometry(
        (6500, total_h),
        (250, 250),  # remove y-margin to stack scales
        (Geometry.SL, scale_h),
        Geometry.DEFAULT_TICK_WH,
        480
    )
    diagnostic_img = Image.new('RGB', (geom_d.total_w, total_h), style.bg)
    r = ImageDraw.Draw(diagnostic_img)
    title_x = round(geom_d.midpoint_x) - geom_d.li
    title = 'Diagnostic Test Print of Available Scales'
    draw_sym_al(r, geom_d, style, style.fg, 50, 0, title, title_x, 0,
                style.font_for(FontSize.TITLE), upper)
    draw_sym_al(r, geom_d, style, style.fg, 200, 0, ' '.join(scale_names), title_x, 0,
                style.font_for(FontSize.SUBTITLE), upper)
    for n, sc_name in enumerate(scale_names):
        sc = getattr(Scales, sc_name)
        al = Align.LOWER if is_demo else layout.scale_al(sc, Side.FRONT, True)
        try:
            gen_scale(r, geom_d, style, k + (n + 1) * sh_with_margins, sc, al=al)
        except Exception as e:
            print(f"Error while generating scale {sc.key}: {e}")
    save_png(diagnostic_img, f'{model_name}.Diagnostic', output_suffix)


if __name__ == '__main__':
    main()
