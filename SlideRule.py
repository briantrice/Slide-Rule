#!/usr/bin/env python3

"""
Slide Rule Scale Generator 2.0 by Javier Lopez 2020
v3.0 Brian T. Rice <briantrice@gmail.com> 2024
Available Scales: A B C D K R1 R2 CI DI CF DF CIF L S T ST

Table of Contents
   1. Setup
   2. Fundamental Functions
   3. Scale Generating Function
   4. Line Drawing Functions
   5. Stickers
   6. Models
   7. Commands
"""

# ----------------------1. Setup----------------------------

import math
import os
import re
import time
import unicodedata
from xml.dom import minidom
from xml.etree import ElementTree
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cache
from itertools import chain
from typing import Callable

import toml
from PIL import Image, ImageFont, ImageDraw
import drawsvg as svg
import ziamath as zm


def keys_of(obj: object):
    return [k for k, v in obj.__dict__.items() if not k.startswith('__')]


# Angular constants:
TAU = math.tau
PI = math.pi
PI_HALF = PI / 2
DEG_FULL = 360
DEG_SEMI = DEG_FULL // 2
DEG_RT = DEG_SEMI // 2

FF = 255
WH = tuple[int, int]


class Color(Enum):
    WHITE, BLACK = (FF, FF, FF), (0, 0, 0)
    RED, GREEN, BLUE = (FF, 0, 0), (0, FF, 0), (0, 0, FF)
    YELLOW, CYAN, MAGENTA = (FF, FF, 0), (0, FF, FF), (FF, 0, FF)

    CUT = BLUE  # color which indicates CUT
    CUTOFF = (230, 230, 230)
    CUTOFF2 = (234, 36, 98)
    SYM_GREEN = (34, 139, 30)  # Override PIL for green for slide rule symbol conventions
    FC_LIGHT_BLUE_BG = (194, 235, 247)  # Faber-Castell scale background
    FC_LIGHT_GREEN_BG = (203, 243, 225)  # Faber-Castell scale background
    PICKETT_EYE_SAVER_YELLOW = (253, 253, 150)  # pastel yellow
    LIGHT_BLUE = 'lightblue'

    RED_WHITE_1 = (FF, 224, 224)
    RED_WHITE_2 = (FF, 192, 192)
    RED_WHITE_3 = (FF, 160, 160)

    GREY = (127, 127, 127)

    @staticmethod
    @cache
    def to_pil(col_spec):
        return col_spec.value if isinstance(col_spec, Color) else col_spec

    @classmethod
    def to_str(cls, col):
        for name, val in cls._member_map_.items():
            if val == col:
                return name.lower()
        if isinstance(col, tuple):
            return f'rgb({col[0]},{col[1]},{col[2]})'
        elif isinstance(col, cls):
            return cls.to_str(col.value)

    @classmethod
    def from_str(cls, color: str):
        return getattr(cls, color.upper(), color)


class FontSize(Enum):
    TITLE = 140
    SUBTITLE = 120
    SC_LBL = 90
    N_XL = 75
    N_LG = 60
    N_MD = 55
    N_MD2 = 50
    N_SM = 45
    N_XS = 35


class FontStyle(Enum):
    REG, ITALIC, BOLD, BOLD_ITALIC = 0, 1, 2, 3


class OutFormat(Enum):
    PNG, SVG = 'png', 'svg'


class Font:
    """Fonts are families per https://cm-unicode.sourceforge.io/font_table.html"""
    Family = tuple[str, str, str, str]
    CMUTypewriter: Family = ('cmuntt', 'cmunit', 'cmuntb', 'cmuntx')
    # = CMUTypewriter-Regular, CMUTypewriter-Italic, CMUTypewriter-Bold, CMUTypewriter-BoldItalic
    CMUSansSerif: Family = ('cmunss', 'cmunsi', 'cmunsx', 'cmunso')
    # = CMUSansSerif, CMUSansSerif-Oblique, CMUSansSerif-Bold, CMUSansSerif-BoldOblique
    CMUConcrete: Family = ('cmunorm', 'cmunoti', 'cmunobx', 'cmunobi')
    # = CMUConcrete-Roman, CMUConcrete-Italic, CMUConcrete-Bold, CMUConcrete-BoldItalic
    CMUBright: Family = ('cmunbsr', 'cmunbso', 'cmunbsr', 'cmunbso')
    # = CMUBright-SemiBold, CMUBright-SemiBoldOblique, CMUBright-SemiBold, CMUBright-SemiBoldOblique
    # 'cmunbmr', 'cmunbmo',  # CMUBright-Roman, CMUBright-Oblique

    @classmethod
    @cache
    def get_truetype_font(cls, font_family: Family, fs: int, font_style: int):
        font_name = font_family[font_style]
        if not font_name.endswith('.ttf'): font_name += '.ttf'
        return ImageFont.truetype(font_name, fs)

    @classmethod
    @cache
    def get_svg_font_for(cls, font: ImageFont):
        font_name = os.path.basename(font.path)
        if font_name.endswith('.ttf'): font_name = font_name[:-4]
        if not font_name.endswith('.svg'): font_name += '.svg'
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fonts', font_name)) as font_file:
            return minidom.parse(font_file)

    @classmethod
    def font_for(cls, font_family: Family, font_size, font_style=FontStyle.REG, h_ratio: float = None):
        fs: int = font_size.value if isinstance(font_size, FontSize) else font_size
        if h_ratio and h_ratio != 1:
            fs = round(fs * h_ratio)
        return cls.get_truetype_font(font_family, fs, font_style.value)


@dataclass(frozen=True)
class Style:
    fg: Color = Color.BLACK
    """foreground color black"""
    bg: Color = Color.WHITE
    """background color white"""
    decreasing_color: Color = Color.RED
    """color for a decreasing value scale"""
    decimal_color: Color = Color.BLACK
    """color for sub-decimal points"""
    bg_colors: dict[str, Color] = field(default_factory=dict)
    """background color overrides for particular scale keys"""
    font_family: Font.Family = Font.CMUTypewriter
    overrides: dict[str, dict[str, object]] = field(default_factory=dict)
    right_sym: bool = True
    """Whether to draw the right legend, usually formula, omitted for pocket slide rules."""
    border_color: Color = Color.BLACK

    @classmethod
    def from_dict(cls, style_def: dict):
        if 'font_family' in style_def:
            style_def['font_family'] = getattr(Font, style_def['font_family'])
        for key in ('fg', 'bg', 'decreasing_color', 'decimal_color'):
            if key in style_def:
                style_def[key] = Color.from_str(style_def[key])
        if 'overrides' in style_def:
            for k, v in style_def['overrides'].items():
                for attr, value in v.items():
                    if attr == 'color':
                        style_def['overrides'][k][attr] = Color.from_str(value)
        if 'bg_colors' in style_def:
            for k, v in style_def['bg_colors'].items():
                style_def['bg_colors'][k] = Color.from_str(v)
        return cls(**style_def)

    def fg_col(self, element: str, is_increasing=True):
        return self.override_for(element, 'color',
                                 self.fg if is_increasing else self.decreasing_color)

    def bg_col(self, element: str):
        return self.bg_colors.get(element)

    def overrides_for(self, element: str) -> dict:
        return self.overrides.get(element)

    def override_for(self, element: str, key: str, default):
        sc_overrides = self.overrides_for(element)
        return sc_overrides.get(key, default) if sc_overrides else default

    def numeral_decimal_color(self):
        return self.decimal_color

    def font_for(self, font_size, h_ratio: float = None, italic: bool = False):
        return Font.font_for(self.font_family, font_size, FontStyle.ITALIC if italic else FontStyle.REG, h_ratio)

    @staticmethod
    def sym_dims(symbol: str, font: ImageFont) -> WH:
        """Gets the size dimensions (width, height) of the input text"""
        (x1, y1, x2, y2) = font.getbbox(symbol)
        return x2 - x1, y2 - y1 + 20

    @classmethod
    def sym_w(cls, symbol: str, font: ImageFont) -> int:
        (x1, _, x2, _) = font.getbbox(symbol)
        return x2 - x1


class HMod(Enum):
    """Tick height size factors (h_mod in pat)"""
    DOT, XS, SM, MED, LG, LG2, XL = 0.25, 0.5, 0.85, 1, 1.15, 1.2, 1.3


class Side(Enum):
    """Side of the slide (front or rear)"""
    FRONT, REAR = 'front', 'rear'


class RulePart(Enum):
    STATOR_TOP, SLIDE, STATOR_BOTTOM = 'stator_top', 'slide', 'stator_bottom'


class BraceShape(Enum):
    L, C = 'L', 'C'


@dataclass(frozen=True)
class Geometry:
    """Slide Rule Geometric Parameters"""
    oX: int = 100  # x margins
    oY: int = 100  # y margins
    side_w: int = 8000  # 30cm = 11.8in
    side_h: int = 1600  # 6cm = 2.36in
    slide_h: int = 640  # 2.4cm = 0.945in

    SH: int = 160  # 6mm
    """scale height"""
    SL: int = 5600  # 21cm = 8.27in
    """scale length"""
    SM: int = 0
    """Scale margin"""

    # Ticks, Labels, are referenced from li as to be consistent
    STH: int = 70  # 2.62mm
    """standard tick height"""
    STT: int = 3  # 0.1125mm
    """standard tick thickness"""
    PixelsPerCM = 1600 / 6
    PixelsPerIN = PixelsPerCM * 2.54

    top_margin: int = 110
    @staticmethod
    def _overrides_factory(): return {Side.FRONT: {}, Side.REAR: {}}
    scale_h_overrides: dict[Side, dict[str, int]] = field(default_factory=_overrides_factory)
    margin_overrides: dict[Side, dict[str, int]] = field(default_factory=_overrides_factory)
    overhang_overrides: dict[Side, dict[str, int]] = field(default_factory=_overrides_factory)

    brace_shape: BraceShape = BraceShape.L
    brace_offset: int = 30  # offset of metal from boundary
    brace_hole_r: int = 34  # screw hole diameter (2.5mm)

    NO_MARGINS = (0, 0)
    DEFAULT_SCALE_WH = (SL, SH)
    DEFAULT_TICK_WH = (STT, STH)

    @classmethod
    def flip_overrides(cls, overrides: dict[Side, dict[int: [str]]] = None):
        result = cls._overrides_factory()
        if overrides:
            for side in Side:
                side_overrides = overrides.get(side, {})
                for h, sc_keys in side_overrides.items():
                    for sc_key in sc_keys:
                        result[side][sc_key] = int(h)
        return result

    @classmethod
    def dim_to_pixels(cls, dim) -> int:
        if matches := re.match(r'^\s*([\d.]+)\s*(\w*)\s*$', dim) if isinstance(dim, str) else None:
            num, units = matches.group(1), matches.group(2)
            result = float(num) if '.' in num else int(num)
            if units == 'cm':
                result *= cls.PixelsPerCM
            elif units == 'mm':
                result *= cls.PixelsPerCM / 10
            elif units == 'in':
                result *= cls.PixelsPerIN
            elif units == 'pt':
                result *= cls.PixelsPerIN / 72
            return int(result)
        return dim

    @classmethod
    def make(cls, side_wh: WH, margins_xy: WH, scale_wh: WH = DEFAULT_SCALE_WH, tick_wh: WH = DEFAULT_TICK_WH,
             slide_h: int = slide_h, top_margin: int = top_margin,
             scale_h_overrides: dict[Side, dict[int: [str]]] = None,
             margin_overrides: dict[Side, dict[int: [str]]] = None,
             overhang_overrides: dict[Side, dict[int: [str]]] = None,
             brace_shape: str = brace_shape.value, brace_offset: int = brace_offset, brace_hole_r: int = brace_hole_r):
        return cls(oX=margins_xy[0], oY=margins_xy[1], side_w=side_wh[0], side_h=side_wh[1], slide_h=slide_h,
                   top_margin=top_margin, SH=scale_wh[1], SL=scale_wh[0], STH=tick_wh[1], STT=tick_wh[0],
                   scale_h_overrides=cls.flip_overrides(scale_h_overrides),
                   margin_overrides=cls.flip_overrides(margin_overrides),
                   overhang_overrides=cls.flip_overrides(overhang_overrides),
                   brace_shape=next((x for x in BraceShape if x.value == brace_shape), brace_shape),
                   brace_offset=brace_offset, brace_hole_r=brace_hole_r)

    @classmethod
    def from_dict(cls, geometry_def: dict):
        for key in ('scale_h_overrides', 'margin_overrides', 'overhang_overrides'):
            if key in geometry_def:
                result = {}
                for side in Side:
                    result[side] = geometry_def[key].get(side.value, {})
                geometry_def[key] = result
        for k, v in geometry_def.items():
            if isinstance(v, list):
                geometry_def[k] = [cls.dim_to_pixels(x) for x in v]
            elif isinstance(v, str):
                geometry_def[k] = cls.dim_to_pixels(v)
        if 'scale_wh' not in geometry_def:
            geometry_def['scale_wh'] = cls.DEFAULT_SCALE_WH
        if 'tick_wh' not in geometry_def:
            geometry_def['tick_wh'] = cls.DEFAULT_TICK_WH
        return cls.make(**geometry_def)

    @property
    def total_w(self):
        return self.side_w + 2 * self.oX

    @property
    def midpoint_x(self):
        return int(self.total_w // 2)

    @property
    def print_h(self):
        return self.side_h * 2 + 3 * self.oY

    @property
    def stator_h(self):
        return int((self.side_h - self.slide_h) // 2)

    @property
    def has_inset_stator(self):
        return self.brace_shape == BraceShape.L

    @property
    def short_stator_inset_w(self):
        return self.stator_h // 2

    @property
    def brace_w(self):
        """Brace width default, to ensure a square anchor piece."""
        return 0 if self.brace_shape is None else self.stator_h

    @property
    def li(self):
        """left index offset from left edge"""
        return (self.total_w - self.SL) // 2

    @property
    def min_tick_offset(self):
        """minimum tick horizontal offset"""
        return self.STT * 3  # separate each tick by at least the space of its width

    def tick_h(self, h_mod: HMod, h_ratio=None) -> int:
        result = self.STH * h_mod.value
        if h_ratio and h_ratio != 1:
            result *= h_ratio
        return round(result)

    def part_h(self, part: RulePart):
        return self.slide_h if part == RulePart.SLIDE else self.stator_h

    def edge_h(self, part: RulePart, top):
        if part == RulePart.STATOR_TOP and top:
            return 0
        if part == RulePart.SLIDE and top or part == RulePart.STATOR_TOP:
            return self.stator_h
        if part == RulePart.STATOR_BOTTOM and top:
            return self.stator_h + self.slide_h
        return self.side_h

    def scale_h(self, sc, side: Side = None, default: int = None) -> int:
        key = sc.key
        if side:
            return self.scale_h_overrides[side].get(key, default or self.SH)
        return self.scale_h_overrides[Side.FRONT].get(
            key, self.scale_h_overrides[Side.REAR].get(key, default or self.SH))

    def scale_margin(self, sc, side: Side = None, default=SM) -> int:
        key = sc.key
        if side:
            return self.margin_overrides[side].get(key, default or self.SM)
        return self.margin_overrides[Side.FRONT].get(
            key, self.margin_overrides[Side.REAR].get(key, default or self.SM))

    def scale_h_ratio(self, sc, side=None):
        if (scale_h := self.scale_h(sc, side=side)) != (default_scale_h := self.DEFAULT_SCALE_WH[1]):
            return scale_h / default_scale_h
        return None

    @staticmethod
    def label_offset_frac(sc) -> float:
        """fraction of total width to overhang each side to label"""
        return (0.05 if sc.can_overhang() or sc.can_spiral() else 0) + 0.02

    def scale_w(self, sc, with_labels=False) -> int:
        if isinstance(sc, Scale):
            return int(self.SL * (sc.overhang_ratio() + (self.label_offset_frac(sc) if with_labels else 0)))
        elif isinstance(sc, Ruler):
            return int(sc.scale_w(self))

    def part_bounds(self, part: RulePart, side: Side):
        """Creates and returns the front part outline as a rectangle of: (x1,y1,dx,dy)"""
        x0, x1 = self.oX, self.total_w - self.oX
        if self.has_inset_stator and part == (RulePart.STATOR_TOP if side == Side.FRONT else RulePart.STATOR_BOTTOM):
            x0 += self.short_stator_inset_w
            x1 -= self.short_stator_inset_w
        return x0, self.edge_h(part, True), x1 - x0, self.part_h(part)

    def brace_outline(self, y_off):
        """Creates and returns the left front brace piece outline in vectors as: (x1,x2,y1,y2)"""
        b = self.brace_offset  # offset of metal from boundary
        x_left = b + self.oX
        x_mid = x_left + self.brace_w // 2
        x_right = self.brace_w - b + self.oX
        y_top = b + y_off
        y_slide_top = y_off + self.stator_h - b
        y_slide_bottom = y_top + self.side_h - self.stator_h
        y_bottom = self.side_h - b + y_off
        if self.brace_shape == BraceShape.L:
            # x_left x_mid x_right
            # 0  ↓    ↓    ↓
            # ↓         1↓   ← 0
            #         ┌────┐ ← y_top
            #         │    │
            #         │    │
            #      4→ │    │ ←6
            #         │    │
            #         │    │
            #      2↓ │    │
            #    ┌────┘    │ ← y_slide_bottom
            # 5→ │         │
            #    │         │
            #    └─────────┘ ← y_bottom
            #        3↑      ← side_h
            return [(x_mid, x_right, y_top, y_top),  # 1
                    (x_left, x_mid, y_slide_bottom, y_slide_bottom),  # 2
                    (x_left, x_right, y_bottom, y_bottom),  # 3
                    (x_mid, x_mid, y_top, y_slide_bottom),  # 4
                    (x_left, x_left, y_slide_bottom, y_bottom),  # 5
                    (x_right, x_right, y_top, y_bottom)]  # 6
        elif self.brace_shape == BraceShape.C:
            return [(x_right, x_right, y_top, y_bottom),  # inside
                    (x_left, x_right, y_top, y_top),  # top
                    (x_left, x_right, y_bottom, y_bottom),  # bottom
                    (x_left, x_left, y_top, y_slide_top),  # outside top
                    (x_left, x_left, y_slide_bottom, y_bottom),  # outside bottom
                    # TODO extend Renderer to handle arcs + replace with arc:
                    (x_left, x_mid, y_slide_top, y_slide_top),  # arc top
                    (x_mid, x_mid, y_slide_top, y_slide_bottom),  # arc inside
                    (x_left, x_mid, y_slide_bottom, y_slide_bottom)]  # arc bottom
        return None

    def mirror_vectors_h(self, vectors: list[tuple[int, int, int, int]]):
        """(x1, x2, y1, y2) mirrored across the centerline"""
        total_w = self.total_w
        return [(total_w - x2, total_w - x1, y1, y2) for (x1, x2, y1, y2) in vectors]

    @staticmethod
    def mirror_vectors_v(vectors: list[tuple[int, int, int, int]], mid_y):
        """(x1, x2, y1, y2) mirrored across a horizontal line"""
        return [(x1, x2, mid_y - y2, mid_y - y1) for (x1, x2, y1, y2) in vectors]


class Align(Enum):
    """Scale Alignment (ticks and labels against upper or lower bounds)"""
    UPPER, LOWER = 'upper', 'lower'


class HAlign(Enum): L, C, R = 'left', 'center', 'right'


TEN = 10
HUNDRED = TEN * TEN


@dataclass(frozen=True)
class GaugeMark:
    sym: str
    value: float = None
    comment: str = None


class Marks:
    e = GaugeMark('e', math.e, comment='base of natural logarithms')
    inv_e = GaugeMark('1/e', 1 / math.e, comment='base of natural logarithms')
    tau = GaugeMark('\\tau', TAU, comment='ratio of circle circumference to radius')
    pi = GaugeMark('\\pi', PI, comment='ratio of circle circumference to diameter')
    pi_half = GaugeMark('\\pi/2', PI_HALF, comment='ratio of quarter arc length to radius')
    pi_quarter = GaugeMark('\\pi/4', PI / 4, comment='ratio of circle area to diameter²')
    inv_pi = GaugeMark('M', 1 / PI, comment='reciprocal of π')

    c = GaugeMark('c', math.sqrt(4 / PI), comment='ratio diameter to √area (square to base scale) √(4/π)')
    c1 = GaugeMark('c′', math.sqrt(4 * TEN / PI), comment='ratio diameter to √area (square to base scale) √(40/π)')

    deg_per_rad = GaugeMark('r', DEG_FULL / TAU / TEN, comment='degrees per radian')
    rad_per_deg = GaugeMark('ρ', TAU / DEG_FULL, comment='radians per degree')
    rad_per_min = GaugeMark('ρ′', TAU / DEG_FULL * 60, comment='radians per minute')
    rad_per_sec = GaugeMark('ρ″', TAU / DEG_FULL * 60 * 60, comment='radians per second')

    ln_over_log10 = GaugeMark('L', 1 / math.log10(math.e), comment='ratio of natural log to log base 10')

    sqrt_ten = GaugeMark('\\sqrt 10', math.sqrt(TEN), comment='square root of 10')
    cube_root_ten = GaugeMark('c', math.pow(TEN, 1 / 3), comment='cube root of 10')

    inf = GaugeMark('\\infty', math.inf, comment='infinity')


class ConversionMarks:
    cm_per_in = GaugeMark('in', 2.54, comment='cm per in')
    sq_cm_per_in = GaugeMark('sq in', cm_per_in.value**2, comment='cm² per in²')
    cu_cm_per_in = GaugeMark('cu in', cm_per_in.value**3, comment='cm³ per in³')
    ft_per_m = GaugeMark('ft', HUNDRED/(cm_per_in.value*12), comment='ft per m')
    yd_per_m = GaugeMark('yd', 3/ft_per_m.value, comment='yd per m')
    km_per_mi = GaugeMark('mi', cm_per_in.value*12*5280/1000, comment='mi per km')
    qt_per_l = GaugeMark('qt', 0.9463525, comment='US qt per l')
    gal_per_l = GaugeMark('gal', qt_per_l.value*4, 'US gal per l')
    lb_per_kg = GaugeMark('lb', 2.2046, comment='lbs per kg')
    hp_per_kw = GaugeMark('N', 1.341022, comment='mechanical horsepower per kW')
    g = GaugeMark('g', 9.80665, comment='gravity acceleration on Earth in m/s²')


# ----------------------2. Fundamental Functions----------------------------


TickFactors = tuple[int, int, int]
TF_BY_MIN: dict[int, TickFactors] = {  # Best tick subdivision pattern for a given minimum overall division
    1000: (TEN, TEN, TEN),
    500: (TEN, TEN, 5),
    250: (TEN, 5, 5),
    100: (TEN, 2, 5),
    50: (2, 5, 5),
    25: (1, 5, 5),
    20: (2, 5, 2),
    10: (2, 5, 1),
    5: (1, 5, 1),
    2: (1, 2, 1),
    1: (1, 1, 1)
}
TF_MIN = sorted(TF_BY_MIN.keys(), reverse=True)
TF_BIN: TickFactors = (4, 4, 4)


def t_s(s1: int, f: TickFactors):
    """tick iterative subdivision"""
    s2 = s1 if f[0] == 1 else s1 // f[0]
    s3 = s2 if f[1] == 1 else s2 // f[1]
    s4 = s3 if f[2] == 1 else s3 // f[2]
    return s1, s2, s3, s4


DEBUG = False


class Out:
    def __init__(self, r):
        self.r = r
    def draw_box(self, x0, y0, dx, dy, col, width=1): pass
    def fill_rect(self, x0, y0, dx, dy, col): pass
    def draw_cut(self, x0, y0, dx, dy, width, col):
        self.fill_rect(x0, y0, max(dx, width), max(dy, width), col)
    def draw_cut2(self, x0, y0, dx, dy, width, col): self.draw_cut(x0, y0, dx, dy, width, col)
    def draw_tick(self, x0, y0, h, width, col): pass
    def draw_circle(self, xc, yc, r, col): pass
    def draw_line(self, x0, y0, x1, y1, col): pass
    def draw_text(self, x_left, y_top, symbol: str, font, color): pass
    def draw_latex(self, x_left, y_top, latex: zm.Latex, font, color): pass

class RasterOut(Out):
    r: ImageDraw.ImageDraw = None

    @classmethod
    def for_image(cls, i: Image.Image):
        return cls(ImageDraw.Draw(i))

    def draw_box(self, x0, y0, dx, dy, col, width=1):
        self.r.rectangle((x0, y0, x0 + dx, y0 + dy), outline=Color.to_pil(col), width=width)

    def fill_rect(self, x0, y0, dx, dy, col):
        self.r.rectangle((x0, y0, x0 + dx, y0 + dy), fill=Color.to_pil(col))

    def draw_circle(self, xc, yc, r, col):
        self.r.circle((xc, yc), r, outline=Color.to_pil(col))

    def draw_line(self, x0, y0, x1, y1, col):
        self.r.line((x0, y0, x1, y1), fill=col)

    def draw_tick(self, x0, y0, h, width, col):
        x0 -= width // 2 + 1
        self.r.rectangle((x0, y0, x0 + width, y0 + h), fill=Color.to_pil(col))

    def draw_text(self, x_left, y_top, symbol: str, font, color):
        self.r.text((x_left, y_top), symbol, font=font, fill=Color.to_pil(color))

    def draw_latex(self, x_left, y_top, latex: zm.Latex, font, color):
        import cairosvg
        png_bytes = cairosvg.svg2png(latex.svg())
        self.r.bitmap((x_left, y_top), png_bytes)


class SVGOut(Out):
    r: svg.Drawing = None
    embed_fonts: bool = False

    cut_color = 'red'
    cut_color_2 = 'cyan'
    etch_color = 'blue'
    etch_color_2 = 'green'
    raster_color = 'black'

    families = {
        'CMU Typewriter Text': 'math',
        'CMU Sans Serif': 'monospace'
    }

    weights = {
        'Bold': 'bold',
        'BoldItalic': 'bold',
        'BoldOblique': 'bold',
    }

    styles = {
        'Italic': 'italic',
        'BoldItalic': 'italic',
        'Oblique': 'oblique',
        'BoldOblique': 'oblique',
    }

    math_font = None

    @classmethod
    def init(cls, debug=False):
        for op in ['\\log', '\\ln', '\\asin', '\\acos', '\\atan', '\\acot']:
            zm.declareoperator(op)
        if debug: zm.config.debug.on()
        if cls.math_font: zm.config.math.mathfont = cls.math_font
        zm.config.math.variant = 'typewriter'

    def __init__(self, r):
        super().__init__(r)
        self.font_ids = set()

    @classmethod
    def for_drawing(cls, i: svg.Drawing):
        return cls(i)

    @cache
    def color_str(self, col):
        if col == Color.BLACK or col == Color.BLACK.value: return None
        return Color.to_str(col) or col

    def add_font(self, font: ImageFont):
        svg_font = Font.get_svg_font_for(font)
        for font_elem in svg_font.getElementsByTagName('font'):
            font_id = font_elem.getAttribute('id')
            if font_id not in self.font_ids:
                self.r.append_def(svg.Raw(font_elem.toxml()))
                self.font_ids.add(font_id)

    def draw_box(self, x0, y0, dx, dy, col, width=1):
        self.r.append(svg.Rectangle(x0, y0, dx, dy, fill_opacity=0, stroke=self.color_str(col), stroke_width=width))

    def fill_rect(self, x0, y0, dx, dy, col):
        self.r.append(svg.Rectangle(x0, y0, dx, dy, fill=self.color_str(col)))

    def draw_cut(self, x0, y0, dx, dy, width, col):
        super().draw_cut(x0, y0, dx, dy, width, self.cut_color)
        # self.r.elements[-1].args['class'] = 'cut'

    def draw_cut2(self, x0, y0, dx, dy, width, col):
        super().draw_cut(x0, y0, dx, dy, width, self.cut_color_2)
        # self.r.elements[-1].args['class'] = 'cut2'

    def draw_circle(self, xc, yc, r, col):
        self.r.append(svg.Circle(xc, yc, r, stroke=self.color_str(col)))

    def draw_line(self, x0, y0, x1, y1, col):
        self.r.append(svg.Line(x0, y0, x1, y1, fill=self.color_str(col)))

    @cache
    def get_tick_ref(self, h, w, col, line_mode=False):
        color_str = self.color_str(col)
        svg_tick = svg.Line(0, 0, h, 0, fill=color_str)\
            if line_mode else svg.Rectangle(0, 0, w, h, fill=color_str)
        svg_tick.args['id'] = f'tick_{w}_{h}'
        # svg_tick.args['class'] = 'etch tick'
        return svg_tick

    def draw_tick(self, x0, y0, h, width, col):
        self.r.append(svg.Use(self.get_tick_ref(h, width, self.etch_color), x0, y0))

    def draw_text(self, x_left, y_top, symbol: str, font: ImageFont, col):
        font_family, font_style = font.getname()
        weight, style = self.weights.get(font_style), self.styles.get(font_style)
        if self.embed_fonts:
            self.add_font(font)
        color = self.etch_color if col == Color.BLACK else self.etch_color_2
        self.r.append(svg.Text(symbol, font.size, x_left, y_top,
                               font_family=font_family, font_weight=weight, font_style=style, fill=color,
                               text_anchor='start', dominant_baseline='hanging', word_spacing=-10))

    def draw_latex(self, x_left, y_top, latex: zm.Latex, font, color):
        latex_svg = latex.svgxml()
        latex_svg.set('x', str(x_left))
        latex_svg.set('y', str(y_top))
        col = self.color_str(color)
        if col: latex_svg.set('fill', col)
        desc = latex_svg.makeelement('desc', {})
        desc.text = latex.latex
        latex_svg.append(desc)
        latex_svg_str = ElementTree.tostring(latex_svg, encoding='unicode')
        self.r.append(svg.Raw(latex_svg_str))


@dataclass(frozen=True)
class Renderer:
    r: Out = None
    geometry: Geometry = None
    style: Style = None

    no_fonts = (None, None, None)

    @classmethod
    def to_image(cls, i, g: Geometry, s: Style):
        out = None
        if isinstance(i, Image.Image):
            out = RasterOut.for_image(i)
        elif isinstance(i, svg.Drawing):
            out = SVGOut.for_drawing(i)
        return cls(out, g, s)

    def draw_tick(self, y_off: int, x: int, h: int, col, scale_h: int, al: Align):
        """Places an individual tick, aligned to top or bottom of scale"""
        x0 = x + self.geometry.li
        y0 = y_off
        if al == Align.LOWER:
            y0 += scale_h - h - 1
        self.r.draw_tick(x0, y0, h, self.geometry.STT, col)

    def pat(self, y_off: int, sc, al: Align, i_start, i_end, i_sf, steps_i, steps_th, steps_font, digit1):
        """
        Place ticks in a graduated pattern. All options are given, not inferred.
        4 levels of tick steps and sizes needed, and three optional fonts for numerals.
        :param y_off: y pos
        :param Scale sc:
        :param Align al: alignment
        :param int i_start: index across the scale to start at
        :param int i_end: index across the scale to end at
        :param int i_sf: scale factor - how much to divide the inputs by before scaling (to generate fine decimals)
        :param tuple[int, int, int, int] steps_i: patterning steps, large to small
        :param tuple[int, int, int, int] steps_th: tick sizes, large to small, per step
        :param tuple[FreeTypeFont, FreeTypeFont, FreeTypeFont] steps_font: optional font sizes, for numerals above ticks
        :param tuple[bool, bool, bool] digit1: whether to show the numerals as the most relevant digit only
        """
        step1, step2, step3, step4 = steps_i
        th1, th2, th3, th4 = steps_th
        f1, f2, f3 = steps_font
        d1, d2, d3 = digit1
        scale_w, scale_h = self.geometry.SL, self.geometry.scale_h(sc)
        col = self.style.fg_col(sc.key, is_increasing=sc.is_increasing)
        tenth_col = self.style.decimal_color if sc.is_increasing else col
        for i in range(i_start, i_end, step4):
            n = i / i_sf
            x = sc.scale_to(n, scale_w)
            if i % step1 == 0:
                tick_h = th1
                if f1:
                    self.draw_numeral(Sym.sig_digit_of(n) if d1 else n, y_off, col, scale_h, x, tick_h, f1, al)
            elif i % step2 == 0:
                tick_h = th2
                if f2:
                    self.draw_numeral(Sym.sig_digit_of(n) if d2 else n, y_off, col, scale_h, x, tick_h, f2, al)
            elif i % step3 == 0:
                tick_h = th3
                if f3:
                    self.draw_numeral(Sym.sig_digit_of(n) if d3 else n, y_off, tenth_col, scale_h, x, tick_h, f3, al)
            else:
                tick_h = th4
            self.draw_tick(y_off, x, tick_h, col, scale_h, al)

    def pat_auto(self, y_off, sc, al, x_start=None, x_end=None, include_last=False):
        """
        Draw a graduated pattern of tick marks across the scale range, by algorithmic selection.
        From some tick patterns in order and by generic suitability, pick the most detailed with a sufficient tick gap.
        """
        if x_start is None:
            x_start = sc.value_at_start()
        if x_end is None:
            x_end = sc.value_at_end()
        if x_start > x_end:
            x_start, x_end = x_end, x_start
        elif x_start == x_end:
            return
        s, g = self.style, self.geometry
        min_tick_offset = g.min_tick_offset
        log_diff = abs(math.log10(abs((x_end - x_start) / max(x_start, x_end))))
        num_digits = math.ceil(log_diff) + 3
        scale_w = g.SL
        sf = 10 ** num_digits  # ensure enough precision for int ranges
        # Ensure a reasonable visual density of numerals
        frac_w = sc.offset_between(x_start, x_end, 1)
        step_num = 10 ** max(int(math.log10(x_end - x_start) - 0.5 * frac_w) + num_digits, 0)
        sub_div4 = next((TF_BY_MIN[i] for i in TF_MIN if i <= step_num
                         and sc.min_offset_for_delta(x_start, x_end, step_num / i / sf, scale_w) >= min_tick_offset),
                        TF_BY_MIN[1])
        (_, step2, step3, step4) = t_s(step_num, sub_div4)
        # Iteration Setup
        i_start = int(x_start * sf)
        if (i_offset := i_start % step4) > 0:  # Align to first tick on or after start
            i_start = i_start - i_offset + step4
        # Determine numeral font size
        h_ratio = g.scale_h_ratio(sc)
        num_font = s.font_for(FontSize.N_LG, h_ratio)
        numeral_tick_offset = sc.min_offset_for_delta(x_start, x_end, step_num / sf, scale_w)
        if (max_num_chars := numeral_tick_offset // s.sym_w('_', num_font)) < 2:
            num_font = s.font_for(FontSize.N_SM, h_ratio)
        # If there are sub-digit ticks to draw, and enough space for single-digit numerals:
        sub_num = (step4 < step3 < step_num) and max_num_chars > 8
        # Tick Heights:
        dot_th = g.tick_h(HMod.DOT, h_ratio)
        self.pat(y_off, sc, al,
                 i_start, int(x_end * sf + (1 if include_last else 0)), sf,
                 (step_num, step2, step3, step4),
                 (
                     g.tick_h(HMod.MED, h_ratio),
                     g.tick_h(HMod.XL if step3 == step_num // 2 and step4 < step3 else HMod.XS, h_ratio),
                     g.tick_h(HMod.XS, h_ratio) if step4 < step3 else dot_th,
                     dot_th),
                 (num_font,
                  s.font_for(FontSize.N_SM, h_ratio) if (sub_num and step2 == step_num // 10
                                                         or step2 == step_num // 2) else None,
                  s.font_for(FontSize.N_XS, h_ratio) if sub_num and step3 == step_num // 10 else None),
                 (max_num_chars < 3, max_num_chars < 16, max_num_chars < 128))

    def draw_symbol(self, symbol: str, color, x_left: float, y_top: float, font: ImageFont):
        symbol = symbol.translate(Sym.UNICODE_SUBS)
        if DEBUG:
            w, h = self.style.sym_dims(symbol, font)
            self.r.draw_box(x_left, y_top, w, h, Color.GREY)
        self.r.draw_text(x_left, y_top, symbol, font, color)

    def draw_expression(self, symbol: str, y_off: int, color, al_h: int, x: int, y: int, font: ImageFont, al: Align, h_al=HAlign.R):
        latex_result = zm.Latex(symbol, font.size, 'mathvariant', color=Color.to_pil(color), inline=True)
        w, h = latex_result.getsize()
        y_top = y_off
        if al == Align.UPPER:
            y_top += y
        elif al == Align.LOWER:
            y_top += al_h - 1 - y - h * 1.2
        x_left = x + self.geometry.li
        if h_al == HAlign.C: x_left -= w / 2
        elif h_al == HAlign.L: x_left -= w
        self.r.draw_latex(round(x_left), y_top, latex_result, font, color)

    def draw_sym_al(self, symbol: str, y_off: int, color, al_h: int, x: int, y: int, font: ImageFont, al: Align):
        """
        :param y_off: y pos
        :param color: color that PIL recognizes
        :param al_h: height (of scale or other bounding band) for alignment
        :param x: offset of centerline from left index (li)
        :param y: offset of base from baseline (LOWER) or top from upper line (UPPER)
        """
        if not symbol:
            return
        w, h = Style.sym_dims(symbol, font)

        y_top = y_off
        if al == Align.UPPER:
            y_top += y
        elif al == Align.LOWER:
            y_top += al_h - 1 - y - h * 1.2
        x_left = x + self.geometry.li - w / 2 + self.geometry.STT / 2
        self.draw_symbol(symbol, color, round(x_left), y_top, font)

    def draw_numeral(self, num, y_off: int, color, scale_h: int, x: int, y: int, font, al: Align):
        """Draw a numeric symbol for a scale"""
        self.draw_sym_al(Sym.num_sym(num), y_off, color, scale_h, x, y, font, al)

    def draw_numeral_sc(self, sc, num, y_off: int, color, scale_h: int, y: int, font, al: Align):
        self.draw_numeral(num, y_off, color, scale_h, sc.pos_of(num, self.geometry), y, font, al)

    def draw_mark(self, mark: GaugeMark, y_off: int, sc, font, al, col=None, shift_adj=0, side=None):
        s, g = self.style, self.geometry
        if col is None:
            col = s.fg_col(sc.key, is_increasing=sc.is_increasing)
        x = sc.scale_to(mark.value, g.SL, shift_adj=shift_adj)
        scale_h = g.scale_h(sc, side=side)
        tick_h = g.tick_h(HMod.XL if al == Align.LOWER else HMod.MED, h_ratio=g.scale_h_ratio(sc, side=side))
        self.draw_tick(y_off, x, tick_h, col, scale_h, al)
        if '\\' in mark.sym:
            self.draw_expression(mark.sym, y_off, col, scale_h, x, tick_h, font, al, h_al=HAlign.C)
        else:
            self.draw_sym_al(mark.sym, y_off, col, scale_h, x, tick_h, font, al)

    # ----------------------4. Line Drawing Functions----------------------------

    def draw_borders(self, y_off: int, side: Side):
        """Place borders around the parts"""
        color = self.style.border_color
        for i, part in enumerate(RulePart):  # per-part cuts
            (x0, y0, w, h) = self.geometry.part_bounds(part, side)
            y0 += y_off
            self.r.draw_cut(x0, y0, 0, h, 1, color) # left
            self.r.draw_cut(x0 + w, y0, 0, h, 1, color) # right
            if side == Side.FRONT or part == RulePart.STATOR_TOP:
                self.r.draw_cut(x0, y0 - (i + 1) // 2, w, 0, 1, color) # top
            if side == Side.REAR or part == RulePart.STATOR_BOTTOM:
                self.r.draw_cut(x0, y0 + h, w, 0, 1, color) # bottom

    def draw_brace_pieces(self, y_off: int, side: Side):
        """Draw the metal bracket locations for viewing"""
        # Initial Boundary verticals
        g = self.geometry
        verticals = [g.brace_w + g.oX, g.total_w - g.brace_w - g.oX]
        for i, start in enumerate(verticals):
            self.r.fill_rect(start - 1, y_off, 2, i, Color.CUTOFF.value)

        brace_fl = g.brace_outline(y_off)
        if brace_fl is None: return

        # Symmetrically create the right piece
        coords = brace_fl + g.mirror_vectors_h(brace_fl)

        # If backside, first apply a vertical reflection
        if side == Side.REAR:
            mid_y = 2 * y_off + g.side_h
            coords = g.mirror_vectors_v(coords, mid_y)
        for (x1, x2, y1, y2) in coords:
            self.r.fill_rect(x1 - 1, y1 - 1, x2 + 1, y2 + 1, Color.CUTOFF2.value)

    # ---------------------- 5. Stickers -----------------------------

    def draw_corners(self, x0: float, y0: float, dx: float, dy: float, col, arm_w=20):
        """Draw cross arms at each corner of the rectangle defined."""
        for (cx, cy) in ((x0, y0), (x0, y0 + dy), (x0 + dx, y0), (x0 + dx, y0 + dy)):
            self.r.draw_line(cx - arm_w, cy, cx + arm_w, cy, col)  # horizontal cross arm
            self.r.draw_line(cx, cy - arm_w, cx, cy + arm_w, col)  # vertical cross arm


class Sym:
    RE_EXPON_CARET = re.compile(r'^(.+)\^([-0-9.A-Za-z]+)$')
    RE_SUB_UNDERSCORE = re.compile(r'^(.+)_(\w+)$')
    RE_EXPON_UNICODE = re.compile(r'^([^⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)$')
    RE_SUB_UNICODE = re.compile(r'^([^₀₁₂₃]+)([₀₁₂₃]+)$')

    @staticmethod
    def num_char_convert(char):
        if char == '⁻':
            return '-'
        return unicodedata.digit(char)

    @classmethod
    def unicode_sub_convert(cls, symbol: str):
        return ''.join(map(str, map(cls.num_char_convert, symbol)))

    @classmethod
    def num_sym(cls, num):
        if isinstance(num, int):
            return str(num)
        elif num.is_integer():
            if num == 0:
                return '0'
            else:
                expon = math.log10(num)
                if expon.is_integer() and abs(expon) > 2:
                    return f'10^{int(expon)}'
                else:
                    return str(int(num))
        else:
            num_sym = str(num)
            if num_sym.startswith('0.'):
                expon = math.log10(num)
                if expon.is_integer() and expon < -2:
                    return f'10^{int(expon)}'
                else:
                    return num_sym[1:]  # Omit leading zero digit
            else:
                return num_sym

    @classmethod
    def split_by(cls, symbol: str, text_re: re.Pattern, unicode_re: re.Pattern):
        base_sym = symbol
        subpart_sym = None
        if matches := re.match(text_re, symbol):
            base_sym = matches.group(1)
            subpart_sym = matches.group(2)
        elif matches := re.match(unicode_re, symbol):
            base_sym = matches.group(1)
            subpart_sym = cls.unicode_sub_convert(matches.group(2))
        return base_sym, subpart_sym

    PRIMES = "'ʹʺ′″‴"
    UNICODE_SUBS = str.maketrans({  # Workaround for incomplete Unicode character support; needs font metadata.
        '′': "ʹ",
        '∡': 'a',
        '½': '1/2',
        '⅓': '1/3',
        '∛': '√',
        '∞': 'inf',
    })

    @classmethod
    def split_expon(cls, symbol: str):
        if len(symbol) > 1 and symbol[-1] in cls.PRIMES:
            return symbol[:-1], symbol[-1:]
        return cls.split_by(symbol, cls.RE_EXPON_CARET, cls.RE_EXPON_UNICODE)

    @classmethod
    def split_subscript(cls, symbol: str):
        return cls.split_by(symbol, cls.RE_SUB_UNDERSCORE, cls.RE_SUB_UNICODE)

    @classmethod
    def parts_of(cls, symbol: str):
        (base_sym, subscript) = cls.split_subscript(symbol)
        (base_sym, expon) = cls.split_expon(base_sym)
        return base_sym, expon, subscript

    @staticmethod
    def first_digit_of(x) -> int:
        return int(str(x)[0])

    @staticmethod
    def last_digit_of(x) -> int:
        if int(x) == x:
            x = int(x)
        return int(str(x)[-1])

    @classmethod
    def sig_digit_of(cls, num: float):
        """When only one digit will fit on a major scale's numerals, pick the most significant."""
        if not (num > 0 and math.log10(num).is_integer()):
            if num % 10 == 0:
                return cls.first_digit_of(num)
            else:
                return cls.last_digit_of(num)
        else:
            return cls.first_digit_of(num)


class BleedDir(Enum):
    UP, DOWN = 'up', 'down'


def extend(i, total_w: int, y: int, direction: BleedDir, amplitude: int):
    """
    Used to create bleed for sticker cutouts
    y: pixel row to duplicate
    amplitude: number of pixels to extend
    """
    assert y < i.height
    if isinstance(i, Image.Image):
        for x in range(0, int(total_w)):
            bleed_color = i.getpixel((x, y))
            for yi in range(y - amplitude, y) if direction == BleedDir.UP else range(y, y + amplitude):
                i.putpixel((x, yi), bleed_color)
    else:
        return  # TODO implement extend/bleed for SVG?


# ----------------------3. Scale Generating Function----------------------------


LN_TEN = math.log(TEN)
LOG10_E = math.log10(math.e)
LOG_0 = -math.inf
E0 = 1e-20
E1N = 1 - 1e-16
E1P = 1 + 1e-15


def unit(x): return x
def gen_base(x: float): return math.log10(x)
def pos_base(p: float): return math.pow(TEN, p)


def scale_sin_tan(x: float):
    return gen_base(HUNDRED * (math.sin(x) + math.tan(x)) / 2)


def angle_opp(x: float):
    """The opposite angle in degrees across a right triangle."""
    return DEG_RT - x


@dataclass(frozen=True)
class ScaleFN:
    """Encapsulates a generating function and its inverse.
    The generating function takes X and returns the fractional position in the unit output space it should locate.
    The inverse function takes a fraction of a unit output space, returning the value to indicate at that position.

    These should be monotonic over their intended range.
    """
    fn: Callable[[float], float]
    """Position of x: Returns the fractional position in the unit output space to put the input value."""
    inverse: Callable[[float], float]
    """Value at p: Returns the value to indicate at the fractional position in the output space."""
    is_increasing: bool = True
    min_x: float = -math.inf
    max_x: float = math.inf

    def __call__(self, x: float):
        return self.fn(x)

    def clamp_input(self, x: float):
        return max(min(x, self.max_x), self.min_x)

    def inverted(self):
        return ScaleFN(self.inverse, self.fn, not self.is_increasing)

    def position_of(self, value):
        return self.fn(value)

    def value_at(self, position):
        return self.inverse(position)

    def value_at_start(self):
        return self.value_at(0)

    def value_at_end(self):
        return self.value_at(1)


class ScaleFNs:
    Unit = ScaleFN(unit, unit)
    F_to_C = ScaleFN(lambda f: (f - 32) * 5 / 9, lambda c: (c * 9 / 5) + 32)
    neper_to_db = ScaleFN(lambda x_db: x_db / (20 / math.log(TEN)), lambda x_n: x_n * 20 / math.log(TEN))

    Base = ScaleFN(gen_base, pos_base, min_x=E0)
    Square = ScaleFN(lambda x: gen_base(x) / 2, lambda p: pos_base(p * 2))
    Cube = ScaleFN(lambda x: gen_base(x) / 3, lambda p: pos_base(p * 3))
    Inverse = ScaleFN(lambda x: 1 - gen_base(x), lambda p: pos_base(1 - p), is_increasing=False, min_x=E0)
    InverseSquare = ScaleFN(lambda x: 1 - gen_base(x) / 2, lambda p: pos_base(1 - p * 2), is_increasing=False, min_x=E0)
    InverseCube = ScaleFN(lambda x: 1 - gen_base(x) / 3, lambda p: pos_base(1 - p * 3), is_increasing=False, min_x=E0)
    SquareRoot = ScaleFN(lambda x: gen_base(x) * 2, lambda p: pos_base(p / 2), min_x=E0)
    CubeRoot = ScaleFN(lambda x: gen_base(x) * 3, lambda p: pos_base(p / 3), min_x=E0)
    Sin = ScaleFN(lambda x: gen_base(TEN * math.sin(math.radians(x))), lambda p: math.asin(pos_base(p)))
    CoSin = ScaleFN(lambda x: gen_base(TEN * math.cos(math.radians(x))), lambda p: math.acos(pos_base(p)),
                    is_increasing=False)
    Tan = ScaleFN(lambda x: gen_base(TEN * math.tan(math.radians(x))), lambda p: math.atan(pos_base(p)))
    SinTan = ScaleFN(lambda x: scale_sin_tan(math.radians(x)), lambda p: math.atan(pos_base(p)))
    SinTanRadians = ScaleFN(scale_sin_tan, lambda p: math.atan(pos_base(math.degrees(p))), min_x=1e-5)
    CoTan = ScaleFN(lambda x: gen_base(TEN * math.tan(math.radians(angle_opp(x)))),
                    lambda p: math.atan(pos_base(angle_opp(p))), is_increasing=False)
    SinH = ScaleFN(lambda x: gen_base(math.sinh(x)), lambda p: math.asinh(pos_base(p)), min_x=E0)
    CosH = ScaleFN(lambda x: gen_base(math.cosh(x)), lambda p: math.acosh(pos_base(p)), min_x=E0)
    TanH = ScaleFN(lambda x: gen_base(math.tanh(x)), lambda p: math.atanh(pos_base(p)), min_x=E0)
    Pythagorean = ScaleFN(lambda x: gen_base(math.sqrt(1 - (x ** 2))) + 1,
                          lambda p: math.sqrt(1 - (pos_base(p) / TEN) ** 2),
                          is_increasing=False, min_x=-E1N, max_x=E1N)
    LogLog = ScaleFN(lambda x: gen_base(math.log(x)), lambda p: math.exp(pos_base(p)), min_x=E1P)
    LogLogNeg = ScaleFN(lambda x: gen_base(-math.log(x)), lambda p: math.exp(pos_base(-p)),
                        is_increasing=False, min_x=E0, max_x=E1N)
    Hyperbolic = ScaleFN(lambda x: gen_base(math.sqrt((x ** 2) - 1)), lambda p: math.hypot(1, pos_base(p)), min_x=E1P)


@dataclass
class Scale:
    """Labeling and basic layout for a given invertible Scaler function."""
    left_sym: str
    """left scale symbol"""
    right_sym: str
    """right scale symbol"""
    scaler: ScaleFN
    gen_fn: Callable[[float], float] = None
    """generating function (producing a fraction of output width)"""
    pos_fn: Callable[[float], float] = None
    """positioning function (takes a proportion of output width, returning what value is there)"""
    shift: float = 0
    """scale shift from left index (as a fraction of output width)"""
    is_increasing: bool = None
    """whether the scale values increase as inputs increase (from left to right)"""
    key: str = None
    """non-unicode name for keying/lookup"""  # TODO extend for all alternate namings?
    on_slide: bool = False
    """whether the scale is meant to be on the slide; implying slide vs stator"""
    opp_key: str = None
    """which scale, if on an edge, it's aligned with"""
    mirror_key: str = None
    """which scale, mathematically, can be assigned numbers going the opposite direction on the ticks"""
    ex_start_value: float = None
    """which value, if the scale is displayed extended at start, to show first"""
    ex_end_value: float = None
    """which value, if the scale is displayed extended at end, to show last"""
    dividers: list[float] = None
    """which values should the automatic scale graduated patterns transition at"""
    marks: list[GaugeMark] = None
    """which marks deserve ticks and labels"""
    numerals: list = None
    """which numerals to label which otherwise wouldn't be"""
    comment: str = None
    """Explanation for the scale or its left symbol"""

    min_overhang_frac = 0.02

    def __post_init__(self):
        self.gen_fn = self.scaler.fn
        self.pos_fn = self.scaler.inverse
        if self.is_increasing is None:
            self.is_increasing = self.scaler.is_increasing
        if self.key is None:
            self.key = self.left_sym

    def __hash__(self):
        return hash(id(self))

    def displays_cyclic(self):
        return self.scaler in {ScaleFNs.Base, ScaleFNs.Inverse, ScaleFNs.Square, ScaleFNs.Cube}

    def can_spiral(self):
        return self.scaler in {ScaleFNs.LogLog, ScaleFNs.LogLogNeg}

    def can_overhang(self):
        return ((self.ex_end_value and self.frac_pos_of(self.ex_end_value) > 1 + self.min_overhang_frac)
                or (self.ex_start_value and self.frac_pos_of(self.ex_start_value) < -self.min_overhang_frac))

    def overhang_ratio(self):
        return max(1., self.frac_pos_of(self.ex_end_value)) - min(0., self.frac_pos_of(self.ex_start_value))\
            if self.can_overhang() else 1

    def frac_pos_of(self, x, shift_adj=0) -> float:
        """
        Generating Function for the Scales, scaled so 0..1 is the left to right of the scale
        :param float x: the dependent variable
        :param float shift_adj: how much the scale is shifted, as a fraction of the scale
        """
        return self.shift + shift_adj + self.gen_fn(x)

    def value_at_frac_pos(self, frac_pos: float, shift_adj=0) -> float:
        return self.pos_fn(frac_pos - self.shift - shift_adj)

    def value_at_start(self):
        return self.value_at_frac_pos(0)

    def value_at_end(self):
        return self.value_at_frac_pos(1)

    def value_range(self):
        return self.value_at_start(), self.value_at_end()

    def powers_of_ten_in_range(self):
        start_value, end_value = self.value_range()
        start_log = math.log10(start_value) if start_value > 0 else LOG_0
        end_log = math.log10(end_value) if end_value > 0 else LOG_0
        start_log, end_log = min(start_log, end_log), max(start_log, end_log)
        if end_log - start_log == math.inf:
            return None
        return range(math.ceil(start_log), math.ceil(end_log))

    def pos_of(self, x: float, g: Geometry) -> int:
        return round(g.SL * self.frac_pos_of(x))

    def offset_between(self, x_start: float, x_end: float, scale_w):
        return abs(self.frac_pos_of(self.scaler.clamp_input(x_end))
                   - self.frac_pos_of(self.scaler.clamp_input(x_start))) * scale_w

    def min_offset_for_delta(self, x_start: float, x_end: float, x_delta: float, scale_w=1):
        return min(
            self.offset_between(x_start, x_start + x_delta, scale_w),
            self.offset_between(x_end - x_delta, x_end, scale_w)
        )

    def scale_to(self, x, scale_w, shift_adj=0) -> int:
        """
        Generating Function for the Scales
        :param float x: the dependent variable
        :param float shift_adj: how much the scale is shifted, as a fraction of its width
        :param int scale_w: scale width in pixels
        :returns: The number of pixels across to the result position
        """
        return round(scale_w * self.frac_pos_of(x, shift_adj=shift_adj))

    def grad_pat_default(self, r: Renderer, y_off, al, extended=True):
        """graduated pattern, with as many defaults as can be inferred from the scale itself"""
        start_value = self.ex_start_value or self.value_at_start() if extended else None
        end_value = self.ex_end_value or self.value_at_end() if extended else None
        dividers = self.dividers
        if not dividers:
            powers = self.powers_of_ten_in_range()
            if powers:
                dividers = [10 ** n for n in powers]
        if dividers:
            r.pat_auto(y_off, self, al, x_start=start_value, x_end=dividers[0])
            last_i = len(dividers) - 1
            for i, di in enumerate(dividers):
                is_last = i >= last_i
                dj = end_value if is_last else dividers[i + 1]
                r.pat_auto(y_off, self, al, x_start=di, x_end=dj, include_last=is_last)
        else:
            r.pat_auto(y_off, self, al, x_start=start_value, x_end=end_value, include_last=True)

    def band_bg(self, r: Renderer, y_off, color, start_value=None, end_value=None):
        g = r.geometry
        li = g.li
        if start_value is None:
            start_value = self.value_at_start()
        if end_value is None:
            end_value = self.value_at_end()
        start_pos = self.pos_of(start_value, g)
        r.r.fill_rect(li + start_pos, y_off, self.pos_of(end_value, g) - start_pos, g.scale_h(self), color)

    def renamed(self, new_key: str, **kwargs):
        if 'left_sym' not in kwargs:
            kwargs['left_sym'] = new_key
        return replace(self, key=new_key, **kwargs)


pi_fold_shift = ScaleFNs.Inverse(PI)


class Scales:
    A = Scale('A', 'x^2', ScaleFNs.Square, opp_key='B', marks=[Marks.pi])
    AI = Scale('AI', '1/x^2', ScaleFNs.InverseSquare)
    B = Scale('B', 'x^2_y', ScaleFNs.Square, on_slide=True, opp_key='A', marks=A.marks)
    BI = Scale('BI', '1/x^2_y', ScaleFNs.InverseSquare, on_slide=True)
    C = Scale('C', 'x_y', ScaleFNs.Base, on_slide=True, opp_key='D', marks=[Marks.pi, Marks.deg_per_rad, Marks.tau])
    CF = Scale('CF', '\\pi x_y', ScaleFNs.Base, shift=pi_fold_shift, on_slide=True, opp_key='DF',
               marks=[Marks.pi, replace(Marks.pi, value=Marks.pi.value/TEN)])
    DF = Scale('DF', '\\pi x', ScaleFNs.Base, shift=pi_fold_shift, opp_key='CF', marks=CF.marks)
    DFM = Scale('DFM', 'x \\log e', ScaleFNs.Base, shift=ScaleFNs.Inverse(LOG10_E), marks=CF.marks)
    DF_M = Scale('DF/M', 'x \\ln 10', ScaleFNs.Base, shift=ScaleFNs.Inverse(LN_TEN), marks=CF.marks)
    CI = Scale('CI', '1/x_y', ScaleFNs.Inverse, on_slide=True, opp_key='DI', marks=CF.marks)
    CIF = Scale('CIF', '1/\\pi x_y', ScaleFNs.Inverse, shift=pi_fold_shift - 1, on_slide=True, marks=C.marks)
    D = Scale('D', 'x', ScaleFNs.Base, opp_key='C', marks=C.marks)
    DI = Scale('DI', '1/x', ScaleFNs.Inverse, opp_key='CI', marks=C.marks)
    DIF = CIF.renamed('DIF', right_sym='1/\\pi x', on_slide=False)
    K = Scale('K', 'x^3', ScaleFNs.Cube)
    KI = Scale('KI', '1/x^3', ScaleFNs.InverseCube)
    L = Scale('L', '\\log x', ScaleFN(lambda x: x / TEN, lambda p: p * TEN, min_x=E0))
    # TODO implement "folded-log" L scale with 1/2 log x with the rest of the range across backwards
    L_FOLDED = Scale('L', '\\tfrac {1}{2} \\log x', ScaleFN(lambda x: x / TEN / 2, lambda p: p * TEN * 2, min_x=E0))
    Ln = Scale('Ln', '\\ln x', ScaleFN(lambda x: x / LN_TEN, lambda p: p * LN_TEN, min_x=E0))
    LL0 = Scale('LL_0', 'e^{.001x}', ScaleFNs.LogLog, shift=3, key='LL0',
                dividers=[1.002, 1.004, 1.010], ex_start_value=1.00095, ex_end_value=1.0105)
    LL1 = Scale('LL_1', 'e^{.01x}', ScaleFNs.LogLog, shift=2, key='LL1',
                dividers=[1.02, 1.05, 1.10], ex_start_value=1.0095, ex_end_value=1.11)
    LL2 = Scale('LL_2', 'e^{.1x}', ScaleFNs.LogLog, shift=1, key='LL2',
                dividers=[1.2, 2], ex_start_value=1.1, ex_end_value=3, marks=[Marks.e])
    LL3 = Scale('LL_3', 'e^x', ScaleFNs.LogLog, key='LL3',
                dividers=[3, 6, 10, 50, 100, 1000, 10000], ex_start_value=2.5, ex_end_value=1e5, marks=[Marks.e])
    LG = L.renamed('LG')
    M = L.renamed('M', comment='M="mantissa"')
    E = LL3.renamed('E', comment='E="exponent"')
    LL00 = Scale('LL_00', 'e^{-.001x}', ScaleFNs.LogLogNeg, shift=3, key='LL00',
                 dividers=[0.998], ex_start_value=0.989, ex_end_value=0.9991)
    LL01 = Scale('LL_01', 'e^{-.01x}', ScaleFNs.LogLogNeg, shift=2, key='LL01',
                 dividers=[0.95, 0.98], ex_start_value=0.9, ex_end_value=0.9906)
    LL02 = Scale('LL_02', 'e^{-.1x}', ScaleFNs.LogLogNeg, shift=1, key='LL02',
                 dividers=[0.8, 0.9], ex_start_value=0.35, ex_end_value=0.91, marks=[Marks.inv_e])
    LL03 = Scale('LL_03', 'e^{-x}', ScaleFNs.LogLogNeg, key='LL03',
                 dividers=[5e-4, 1e-3, 1e-2, 0.1], ex_start_value=1e-4, ex_end_value=0.39, marks=[Marks.inv_e])
    LL_0, LL_1, LL_2, LL_3 = LL00.renamed('LL/0'), LL01.renamed('LL/1'), LL02.renamed('LL/2'), LL03.renamed('LL/3')
    P = Scale('P', '\\sqrt{1-(.1x)^2}', ScaleFNs.Pythagorean, key='P',
              dividers=[0.3, 0.6, 0.8, 0.9, 0.98, 0.99], ex_start_value=0.1, ex_end_value=.995)
    P1 = P.renamed('P_1')
    P2 = replace(P, left_sym='P_2', key='P_2', right_sym='\sqrt{1-(.01x)^2}', shift=1,
                 dividers=[0.999, 0.9998], ex_start_value=P1.ex_end_value, ex_end_value=0.99995, numerals=[0.99995])
    Q1 = Scale('Q_1', '\\sqrt[3]x', ScaleFNs.CubeRoot, marks=[Marks.cube_root_ten], key='Q1')
    Q2 = Scale('Q_2', '\\sqrt[3]{10x}', ScaleFNs.CubeRoot, shift=-1, marks=[Marks.cube_root_ten], key='Q2')
    Q3 = Scale('Q_3', '\\sqrt[3]{100x}', ScaleFNs.CubeRoot, shift=-2, marks=[Marks.cube_root_ten], key='Q3')
    R1 = Scale('R_1', '\\sqrt{x}', ScaleFNs.SquareRoot, key='R1', marks=[Marks.sqrt_ten])
    R2 = Scale('R_2', '\\sqrt{10x}', ScaleFNs.SquareRoot, key='R2', shift=-1, marks=R1.marks)
    Sq1, Sq2 = R1.renamed('Sq1'), R2.renamed('Sq2')
    S = Scale('S', '\\asin x°', ScaleFNs.Sin, mirror_key='CoS')
    CoS = Scale('C', '\\acos x°', ScaleFNs.CoSin, key='CoS', mirror_key='S')
    # SRT = Scale('SRT', '∡tan 0.01x', ScaleFNs.SinTanRadians)
    ST = Scale('ST', '\\atan 0.01x°', ScaleFNs.SinTan)
    T = Scale('T', '\\atan x°', ScaleFNs.Tan, mirror_key='CoT')
    CoT = Scale('CoT', '\\acot x°', ScaleFNs.CoTan, key='CoT', is_increasing=False, mirror_key='T', shift=-1)
    T1 = T.renamed('T1', left_sym='T₁')
    T2 = replace(T, left_sym='T₂', right_sym='∡tan 0.1x°', key='T2', shift=-1, mirror_key='CoT2')
    W1 = Scale('W_1', '\\sqrt x', ScaleFNs.SquareRoot, key='W1', opp_key="W1'", dividers=[1, 2],
               ex_start_value=0.95, ex_end_value=3.38, marks=[Marks.sqrt_ten])
    W1Prime = W1.renamed("W1'", left_sym="W'₁", opp_key='W1')
    W2 = Scale('W_2', '\\sqrt{10x}', ScaleFNs.SquareRoot, key='W2', shift=-1, opp_key="W2'", dividers=[5],
               ex_start_value=3, ex_end_value=10.66, marks=W1.marks)
    W2Prime = W2.renamed("W2'", left_sym="W'₂", opp_key='W2')

    H1 = Scale('H_1', '\\sqrt{1+(.1x)^2}', ScaleFNs.Hyperbolic, key='H1', shift=1, dividers=[1.03, 1.1], numerals=[1.005])
    H2 = Scale('H_2', '\\sqrt{1+x^2}', ScaleFNs.Hyperbolic, key='H2', dividers=[4])
    SH1 = Scale('Sh_1', '\\sinh x', ScaleFNs.SinH, key='Sh1', shift=1, dividers=[0.2, 0.4])
    SH2 = Scale('Sh_2', '\\sinh x', ScaleFNs.SinH, key='Sh2')
    CH1 = Scale('Ch', '\\cosh x', ScaleFNs.CosH, dividers=[1, 2], ex_start_value=0.01)
    TH = Scale('Th', '\\tanh x', ScaleFNs.TanH, shift=1, dividers=[0.2, 0.4, 1, 2], ex_end_value=3)

    Const = Scale('Const', '', ScaleFNs.Base,
                  marks=[Marks.e, Marks.pi, Marks.tau, Marks.deg_per_rad, Marks.rad_per_deg, Marks.c])


shift_360 = ScaleFNs.Inverse(3.6)
SquareRootNonLog = ScaleFN(lambda x: (x / TEN) ** 2, lambda p: TEN * math.sqrt(p), min_x=0.)
custom_scale_sets: dict[str, dict[str, Scale]] = {
    'Merchant': {  # scales from Aristo 965 Commerz II: KZ, %, Z/ZZ1/ZZ2/ZZ3 compound interest
        'Z': Scales.D.renamed('Z', right_sym='', opp_key='T1', dividers=[2, 4], comment='Z="Zins": interest'),
        'T1': Scales.D.renamed('T1', left_sym='T₁', right_sym='', opp_key='Z', on_slide=True),
        'P1': Scales.CI.renamed('P1', left_sym='P₁', right_sym='', on_slide=True, dividers=[2, 4]),
        'KZ': Scales.CF.renamed('KZ', right_sym='', shift=shift_360,
                                ex_start_value=0.3, ex_end_value=4, dividers=[0.4, 1, 2],
                                comment='KZ="Kapital Zins": interest on the principal, over 360 business days/year'),
        'T2': Scales.CF.renamed('T2', left_sym='T₂', shift=shift_360, on_slide=True,
                                ex_start_value=0.3, ex_end_value=4, dividers=[0.4, 1, 2]),
        'P2': Scales.CIF.renamed('P2', left_sym='P₂', right_sym='', shift=shift_360 - 1, on_slide=True,
                                 ex_start_value=0.25, ex_end_value=3.3, dividers=[0.4, 1, 2]),
        'P%': Scale(left_sym='', right_sym='', key='P%', on_slide=True, shift=shift_360,
                    scaler=ScaleFN(lambda x: gen_base((x + HUNDRED) / HUNDRED),
                                   lambda p: pos_base(p) * HUNDRED - HUNDRED),
                    dividers=[0], ex_start_value=-50, ex_end_value=100),
        # meta-scale showing % with 100% over 1/unity
        # special marks being 0,5,10,15,20,25,30,33⅓,40,50,75,100 in both directions
        'ZZ1': Scales.LL1.renamed('ZZ1', comment='ZZ="Zins Zins": compound interest'),
        'ZZ2': Scales.LL2.renamed('ZZ2', comment='ZZ="Zins Zins": compound interest'),
        'ZZ3': Scales.LL3.renamed('ZZ3', comment='ZZ="Zins Zins": compound interest'),
        'ZZ%': Scale(left_sym='ZZ%', right_sym='%', key='ZZ%', shift=2,
                     scaler=ScaleFN(lambda x: Scales.LL1.gen_fn((x + HUNDRED) / HUNDRED),
                                    lambda p: Scales.LL1.pos_fn(p) * HUNDRED - HUNDRED),
                     ex_start_value=1, ex_end_value=11, comment='Compound interest percentage'),
        '£': Scales.L.renamed('£', right_sym=''),
    },
    'Hemmi153': {
        'Chi': Scale('χ', '', ScaleFN(lambda x: x / PI_HALF, lambda p: p * PI_HALF), marks=[Marks.pi_half]),
        'K': replace(Scales.K, dividers=[2, 5, 10, 20, 50, 100, 200, 500], key='Hemmi153_K'),
        'θ': Scale('θ', '', ScaleFN(lambda x: math.sin(math.radians(x)) ** 2,
                                    lambda p: math.degrees(math.asin(math.sqrt(p)))), dividers=[5, 15, 75, 85]),
        'R_θ': Scale('R_θ', '', ScaleFN(lambda x: math.sin(x) ** 2, lambda p: math.asin(math.sqrt(p))),
                     dividers=[0.1, 0.2, 1, 1.4]),
        'P': Scale('P', 'SIN', SquareRootNonLog, opp_key='Q', dividers=[1, 2, 5]),
        'Q': Scale('Q', 'COS', SquareRootNonLog, opp_key='P', dividers=[1, 2, 5], on_slide=True),
        'Q′': Scale('Q′', '', SquareRootNonLog, shift=-1, on_slide=True),
        'T': Scale('T', '', ScaleFN(lambda x: math.sin(math.atan(x)) ** 2,
                                    lambda p: math.tan(math.asin(math.sqrt(p)))),
                   ex_end_value=10, dividers=[0.1, 0.2, 1.2, 1.5, 2, 3, 4, 5], key='Hemmi153_T'),
        # Gudermannian function:
        'G_θ': Scale('G_θ', '', ScaleFN(lambda x: math.tanh(x) ** 2,
                                        lambda p: math.atanh(math.sqrt(p))),
                     dividers=[0.1, 2, 3, 4], ex_end_value=0.999),
    },
    'PickettN515T': {
        'f_x': Scale('f_x', 'x/2π', ScaleFNs.Base, shift=gen_base(TAU), dividers=[0.2, 0.5, 1]),
        'L_r': Scale('L_r', '1/(2\pi x)²', ScaleFNs.InverseSquare, shift=gen_base(1 / TAU),
                     dividers=[0.05, 0.1, 0.2, 0.5, 1, 2], ex_start_value=0.025, ex_end_value=2.55),
    }
}


class Layout:
    def __init__(self, front: str, rear: str = None, scale_ns: dict = None, align_overrides=None):
        if align_overrides is None:
            align_overrides = {}
        if not rear and '\n' in front:
            (front, rear) = front.splitlines()
        self.sc_keys: dict[Side, dict[RulePart, list[str]]] = {
            Side.FRONT: self.parse_side_layout(front),
            Side.REAR: self.parse_side_layout(rear)
        }
        self.scale_ns: dict[str, Scale] = scale_ns or {}
        self.check_scales()
        self.scale_aligns: dict[Side, dict[str, Align]] = {
            Side.FRONT: align_overrides.get(Side.FRONT, {}), Side.REAR: align_overrides.get(Side.REAR, {})}
        self.infer_aligns()

    def __repr__(self):
        return f'Layout({self.sc_keys})'

    @classmethod
    def from_dict(cls, layout_def):
        if 'align_overrides' in layout_def:
            result = {Side.FRONT: {}, Side.REAR: {}}
            for k, v in layout_def['align_overrides'].items():
                for sc_key, al in v.items():
                    result[Side(k.lower())][sc_key] = Align[al.upper()]
            layout_def['align_overrides'] = result
        if 'scale_ns' in layout_def:
            layout_def['scale_ns'] = custom_scale_sets[layout_def['scale_ns']]
        return cls(**layout_def)

    @classmethod
    def parse_segment_layout(cls, segment_layout: str) -> [str]:
        if segment_layout:
            return re.split(r'[, ]+', segment_layout.strip(' '))
        return None

    @classmethod
    def parts_of_side_layout(cls, side_layout: str) -> [str]:
        side_layout = side_layout.strip(' |')
        parts = None
        if '[' in side_layout and ']' in side_layout:
            parts = re.split(r'[\[\]]', side_layout, 2)
        elif '/' in side_layout:
            parts = side_layout.split('/', 2)
        if parts:
            return [x.strip() for x in parts]
        return [side_layout, '', '']

    @classmethod
    def parse_side_layout(cls, layout: str) -> dict[RulePart, list[str]]:
        top_scales = None
        bottom_scales = None
        slide_scales = None
        if layout:
            major_parts = [cls.parse_segment_layout(x) for x in cls.parts_of_side_layout(layout.strip(' |'))]
            num_parts = len(major_parts)
            if num_parts == 1:
                (slide_scales) = major_parts
            elif num_parts == 3:
                (top_scales, slide_scales, bottom_scales) = major_parts
        return {RulePart.STATOR_TOP: top_scales, RulePart.SLIDE: slide_scales, RulePart.STATOR_BOTTOM: bottom_scales}

    def sc_keys_at(self, side: Side, part: RulePart, fallback=None):
        return self.sc_keys[side].get(part, fallback) or fallback

    def sc_keys_in_order(self):
        for side in Side:
            for part in RulePart:
                yield from self.sc_keys_at(side, part, [])

    def check_scales(self):
        for scale_name in self.sc_keys_in_order():
            if self.scale_named(scale_name) is None:
                raise ValueError(f'Unrecognized front scale name: {scale_name}')

    def scale_named(self, sc_name: str):
        sc_attr = sc_name.replace('/', '_') if '/' in sc_name else sc_name
        sc_attr = sc_name.replace("'", 'Prime') if "'" in sc_attr else sc_attr
        return self.scale_ns.get(sc_name, getattr(Scales, sc_attr, getattr(Rulers, sc_attr, None)))

    def all_scales(self):
        return (self.scale_named(sc_name) for sc_name in self.sc_keys_in_order())

    def scales_at(self, side: Side, part: RulePart) -> list[Scale]:
        return [self.scale_named(sc_name) for sc_name in self.sc_keys[side][part] or []]

    def infer_aligns(self):
        """Fill scale alignments per the layout into the overrides."""
        for side in Side:
            side_overrides = self.scale_aligns[side]
            side_seen = set()
            for part in RulePart:
                last_i = len(part_scales := self.scales_at(side, part)) - 1
                for i, sc in enumerate(part_scales):
                    if sc.key not in side_overrides:
                        if isinstance(sc, Ruler):  # rulers are always edge-aligned
                            side_overrides[sc.key] = Align.UPPER if part == RulePart.STATOR_TOP else Align.LOWER
                        elif i == 0 and part == RulePart.SLIDE:
                            side_overrides[sc.key] = Align.UPPER
                        elif i == last_i and part != RulePart.STATOR_BOTTOM:
                            side_overrides[sc.key] = Align.LOWER
                        elif sc.opp_key:
                            side_overrides[sc.key] = Align.UPPER if sc.opp_key in side_seen else Align.LOWER
                    side_seen.add(sc.key)

    def scale_al(self, sc: Scale, side: Side, part: RulePart):
        default_al = Align.UPPER if part == RulePart.STATOR_BOTTOM else Align.LOWER
        return self.scale_aligns[side].get(sc.key, default_al)


@dataclass(frozen=True)
class Ruler:
    """
    Rulers are geometry-dependent scales that appear only on the edges of a model per side.
    Like a scale, it has a tick pattern. Unlike a scale, it only aligns with the geometry itself.
    Also, tick placement is physically-oriented, so we need to avoid numeric error.
    """
    key: str = None
    unit: str = None
    tick_pattern: TickFactors = None
    left_offset: float = 0.0
    """margin from left side of Geometry in units"""
    num_units: int = 1000
    """number of units to show; should be generated or at least limited from geometry"""
    pixels_per_unit: float = 1

    def pos_of(self, x: float) -> int:
        return int((self.left_offset + x) * self.pixels_per_unit)

    def value_at(self, p: int) -> float:
        return (p / self.pixels_per_unit) - self.left_offset

    def scale_w(self, g: Geometry):
        return self.num_units_in(g) * self.pixels_per_unit

    def num_units_in(self, g: Geometry) -> int:
        return min(self.num_units, math.floor(g.side_w / self.pixels_per_unit - self.left_offset) if g else math.inf)

    @property
    def is_increasing(self):
        return True

    def pat(self, r: Renderer, y_off: int, al: Align):
        s, g = r.style, r.geometry
        li = g.li
        th1, th2, th3, th4 = (g.tick_h(HMod.LG), g.tick_h(HMod.MED), g.tick_h(HMod.XS), g.tick_h(HMod.DOT))
        font1 = s.font_for(FontSize.N_LG)
        scale_h = g.scale_h(self)
        if DEBUG:
            r.r.draw_box(self.pos_of(0), y_off, self.scale_w(g), scale_h, Color.GREY)
        sym_col = s.fg_col(self.key, is_increasing=self.is_increasing)
        i_sf = math.prod(self.tick_pattern)
        step1, step2, step3, _ = t_s(i_sf, self.tick_pattern)
        for i in range(0, self.num_units_in(g) * i_sf + 1):
            num = i / i_sf
            x = self.pos_of(num) - li
            tick_h = th4
            if i % step1 == 0:
                tick_h = th1
                r.draw_numeral(num, y_off, sym_col, scale_h, x, th1, font1, al)
            elif i % step2 == 0:
                tick_h = th2
            elif i % step3 == 0:
                tick_h = th3
            r.draw_tick(y_off, x, tick_h, sym_col, scale_h, al)

    def generate(self, r: Renderer, y_off: int, al: Align, side: Side = Side.FRONT):
        s, g = r.style, r.geometry
        # Units symbol on right
        scale_h = g.scale_h(self, side=side)
        sym_col = s.fg_col(self.key, is_increasing=self.is_increasing)
        f_lbl_r = s.font_for(FontSize.SC_LBL if scale_h > FontSize.SC_LBL.value * 1.5 else scale_h // 2)
        w2, h2 = s.sym_dims(right_sym := self.unit, f_lbl_r)
        y2 = (g.SH - h2) // 2  # Ignore custom height/spacing for legend symbols
        x_right = self.pos_of(self.num_units_in(g)) + w2
        r.draw_symbol(right_sym, sym_col, x_right,  y_off + y2, f_lbl_r)
        # Ticks
        self.pat(r, y_off, al)


class Rulers:
    PX = Ruler('px', '100px', TF_BY_MIN[10], 0, 100, 100)
    PT = Ruler('pt', 'pt', TF_BY_MIN[10], 0, 25 * 72, Geometry.PixelsPerIN // 72)
    CM = Ruler('cm', 'cm', TF_BY_MIN[20], 1.5, 30, Geometry.PixelsPerCM)
    IN_DEC = Ruler('in10', 'in', TF_BY_MIN[50], 0.5, 12, Geometry.PixelsPerIN)
    IN_BIN = Ruler('in2', 'in', TF_BIN, 0.5, 12, Geometry.PixelsPerIN)
    IN = IN_DEC


# --------------------------6. Models----------------------------


class Layouts:
    RegleDesEcoles = Layout('DF/CF C/D', '')
    Mannheim = Layout('A/B CI C/D K', '[S L T]')
    Rietz = Layout('K A/B CI C/D L', '[S ST T]')
    GenericDuplex = Layout('A [B, C] D', 'DF [CF, C] D')
    Darmstadt = Layout('K A/B K CI C/D P', '[LL1 LL2 LL3]')
    DarmstadtAdvanced = Layout('T K A/B BI CI C/D P S', '[ L LL0 LL1 LL2 LL3 ]')


@dataclass(frozen=True)
class Model:
    brand: str
    subtitle: str
    name: str
    geometry: Geometry
    layout: Layout
    style: Style = Style()

    @classmethod
    def from_dict(cls, model_def: dict):
        return cls(brand=model_def.get('brand'), subtitle=model_def.get('subtitle'), name=model_def.get('name'),
                   layout=Layout.from_dict(model_def['layout']), geometry=Geometry.from_dict(model_def['geometry']),
                   style=Style.from_dict(model_def.get('style', {})))

    @classmethod
    def from_toml_file(cls, toml_filename: str):
        return cls.from_dict(toml.load(toml_filename))

    example_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')

    @classmethod
    def from_example(cls, example_name: str):
        return cls.from_toml_file(os.path.join(cls.example_dir_path, f'Model-{example_name}.toml'))

    @classmethod
    def load(cls, model_name):
        if model_name == 'Demo':
            return DemoModel
        return cls.from_toml_file(model_name) if os.path.exists(model_name) else cls.from_example(model_name)

    @classmethod
    def example_names(cls):
        for fn in os.listdir(cls.example_dir_path):
            if match := re.match(r'Model-(.*)\.toml$', fn):
                yield match.group(1)

    def scale_h_per(self, side: Side, part: RulePart):
        result = 0
        for sc in self.layout.scales_at(side, part):
            result += self.geometry.scale_margin(sc, side)
            result += self.geometry.scale_h(sc, side)
        return result

    def max_scale_w(self):
        return max(self.geometry.scale_w(sc, with_labels=True) for sc in self.layout.all_scales())

    def auto_stock_h(self):
        result = 0
        for side in Side:
            for part in RulePart.STATOR_TOP, RulePart.STATOR_BOTTOM:
                result = max(result, self.scale_h_per(side, part))
        return result

    def auto_slide_h(self):
        result = 0
        for side in Side:
            result = max(result, self.scale_h_per(side, RulePart.SLIDE))
        return result


DemoModel = Model.from_example('Demo')


def gen_scale(r: Renderer, y_off: int, sc: Scale, al=None, side: Side = None):
    s, g = r.style, r.geometry
    if s.override_for(sc.key, 'hide', False):
        return

    h = g.scale_h(sc, side=side)
    h_ratio = g.scale_h_ratio(sc, side=side)

    # Place Index Symbols (Left and Right)
    f_lbl = s.font_for(FontSize.SC_LBL if h > FontSize.SC_LBL.value * 1.5 else h // 2)
    f_lbl_s = s.font_for(FontSize.N_LG if h > FontSize.N_LG.value * 2 else h // 2)
    f_xln = s.font_for(FontSize.N_XL, h_ratio)
    f_lgn = s.font_for(FontSize.N_LG, h_ratio)
    f_mdn = s.font_for(FontSize.N_MD, h_ratio)
    f_smn = s.font_for(FontSize.N_SM, h_ratio)
    f_mdn_i = s.font_for(FontSize.N_MD, h_ratio, italic=True)
    f_md2 = s.font_for(FontSize.N_MD2, h_ratio)
    f_md2_i = s.font_for(FontSize.N_MD2, h_ratio, italic=True)

    scale_w = g.SL
    if DEBUG:
        r.r.draw_box(g.li, y_off, scale_w, h, Color.GREY)

    sym_col = s.fg_col(sc.key, is_increasing=sc.is_increasing)
    bg_col = s.bg_col(sc.key)
    if bg_col:
        sc.band_bg(r, y_off, bg_col)

    label_offset_frac = g.label_offset_frac(sc)

    # Special Symbols for S, and T
    sc_alt: Scale = getattr(Scales, sc.mirror_key, None) if sc.mirror_key else None
    alt_col = s.fg_col(sc_alt.key, is_increasing=not sc_alt.is_increasing) if sc_alt else None

    # Right
    f_lbl_r = f_lbl_s if len(sc.right_sym) > 6 else f_lbl
    _, h2 = s.sym_dims(sc.right_sym, f_lbl_r)
    y2 = (g.SH - h2) // 2  # Ignore custom height/spacing for legend symbols
    if s.right_sym:
        x_right = round((1 + label_offset_frac) * scale_w)
        r.draw_expression(sc.right_sym, y_off, sym_col, h, x_right, y2, f_lbl_r, al, h_al=HAlign.R)
        if sc_alt or sc == Scales.ST:
            right_sym = sc_alt.right_sym if sc_alt else '\\asin 0.01x°'
            r.draw_expression(right_sym, y_off, alt_col or sym_col, h, x_right, round(y2 - h2 * 0.8), f_lbl_r, al, h_al=HAlign.R)

    # Left
    (left_sym, _, subscript) = Sym.parts_of(sc.left_sym)
    _, h1 = s.sym_dims(left_sym, f_lbl)
    y1 = (g.SH - h1) // 2  # Ignore custom height/spacing for legend symbols
    x_left = round((0 - label_offset_frac) * scale_w)
    r.draw_expression(sc.left_sym, y_off, sym_col, h, x_left, y1, f_lbl, al, h_al=HAlign.L)
    if alt_col:
        r.draw_expression(sc_alt.left_sym, y_off, alt_col, h, x_left - s.sym_w('__', f_lbl), y2, f_lbl, al, h_al=HAlign.L)

    th_xl, th_l, th, th_s, th_xs, th_dot = [
        g.tick_h(x, h_ratio) for x in (HMod.XL, HMod.LG, HMod.MED, HMod.SM, HMod.XS, HMod.DOT)]

    # Tick Placement (the bulk!)
    fonts1, fonts2, fonts_xl = (f_lbl, None, None), (f_lbl, f_mdn, None), (f_xln, None, None)
    ths1 = (th, th_xl, th_s, th_xs)
    ths2 = (th, th_xl, th_xs, th_xs)
    ths3 = (th, th_s, th_s, th_xs)
    ths4 = (th, th_xl, th_s, th_dot)
    ths5 = (th, th, th_s, th_xs)
    sf = 1000  # Reflects the minimum significant figures needed for standard scales to avoid FP inaccuracy
    d0, d1, d2 = (False, False, False), (True, False, False), (True, True, False)

    def pats(fp_v, tf_v, th_v, font_v, d_v):
        for i in range(0, len(fp_v) - 1):
            fp_next = fp_v[i + 1] + (1 if i == len(fp_v) - 2 else 0)  # Ensure the last fencepost value is drawn
            r.pat(y_off, sc, al, int(fp_v[i]), int(fp_next), sf, tf_v[i], th_v[i], font_v[i], d_v[i])

    if sc == Scales.Const:
        pass
    elif (sc.scaler in {ScaleFNs.Base, ScaleFNs.Inverse}) and sc.shift == 0:  # C/D and CI/DI
        pats([sf * fp for fp in (1, 2, 4, 10)],
             (t_s(sf, TF_BY_MIN[100]), t_s(sf, TF_BY_MIN[50]), t_s(sf, TF_BY_MIN[20])),
             (ths3, ths1, ths1), (fonts2, fonts1, fonts1), (d2, d0, d1))
    elif sc in {Scales.A, Scales.B, Scales.AI, Scales.BI}:
        for b in (sf * 10 ** n for n in range(0, 2)):
            pats([fp * b for fp in (1, 2, 5, 10)],
                 (t_s(b, TF_BY_MIN[50]), t_s(b, TF_BY_MIN[20]), t_s(b, TF_BY_MIN[10])),
                 (ths1, ths1, ths2), (fonts1, fonts1, fonts1), [d1] * 3)

    elif sc in {Scales.K, Scales.KI}:
        for b in (sf * (10 ** n) for n in range(0, 3)):
            pats([fp * b for fp in (1, 3, 6, 10)],
                 (t_s(b, TF_BY_MIN[20]), t_s(b, TF_BY_MIN[10]), t_s(b, TF_BY_MIN[5])),
                 (ths1, ths2, ths2), [fonts_xl] * 3, [d1] * 3)

    elif sc in {Scales.R1, Scales.Sq1}:
        pats([int(fp * sf) for fp in (1, 2, 3.17)],
             (t_s(sf // 10, TF_BY_MIN[20]), t_s(sf, TF_BY_MIN[100])),
             (ths1, ths5), (r.no_fonts, fonts2), (d1, d2))

        # 1-1.9 Labels
        r.draw_numeral_sc(sc, 1, y_off, sym_col, h, th, f_lbl, al)
        for x in (x / 10 for x in range(11, 20)):
            r.draw_numeral(Sym.last_digit_of(x), y_off, sym_col, h, sc.pos_of(x, g), th, f_lgn, al)

    elif sc in {Scales.R2, Scales.Sq2}:
        pats([int(fp * sf) for fp in (3.16, 5, 10)],
             (t_s(sf, TF_BY_MIN[100]), t_s(sf, TF_BY_MIN[50])),
             (ths3, ths1), (fonts2, fonts1), [d2] * 2)

    elif sc in {Scales.CF, Scales.DF, Scales.CIF}:
        is_cif = sc == Scales.CIF
        fp1 = int((0.31 if is_cif else 0.314) * sf)
        i1 = sf // TEN
        pats([fp1] + [fp * i1 for fp in (4, 10, 20)] + [int(3.2 * sf) if is_cif else fp1 * TEN],
             (t_s(i1, TF_BY_MIN[50]), t_s(i1, TF_BY_MIN[20]), t_s(sf, TF_BY_MIN[100]), t_s(sf, TF_BY_MIN[50])),
             (ths1, ths1, ths3, ths1), (fonts1, fonts1, fonts2, fonts1), [d2] * 4)

    elif sc == Scales.L:
        r.pat(y_off, sc, al, 0, TEN * sf + 1, sf, t_s(sf, TF_BY_MIN[50]),
              (th_l, th_xl, th, th_xs), r.no_fonts, d0)
        for x in range(0, 11):
            r.draw_numeral(x / 10, y_off, sym_col, h, sc.pos_of(x, g), th, f_lbl, al)

    elif sc.scaler in {ScaleFNs.Sin, ScaleFNs.CoSin} or sc in {Scales.T, Scales.T1, Scales.CoT}:
        is_tan = sc.scaler in {ScaleFNs.Tan, ScaleFNs.CoTan}
        ths_y = (th_xl, th_xl, th_s, th_xs)
        ths_z = (th_xl, th_s, th_xs, th_xs)
        sc_t = Scales.S if sc.scaler == ScaleFNs.CoSin else sc
        if is_tan:
            pats([int(fp * sf) for fp in (5.7, 10, 25, 45)],
                 (t_s(sf, TF_BY_MIN[20]), t_s(sf, TF_BY_MIN[10]), t_s(sf * 5, (5, 5, 1))),
                 (ths_y, ths_z, (th_xl, th, th_xs, th_xs)), [r.no_fonts] * 3, [d1] * 3)
        else:
            pats([int(fp * sf) for fp in (5.7, 20, 30, 60, 80, DEG_RT)],
                 (t_s(sf, TF_BY_MIN[10]), t_s(sf * 5, (5, 5, 1)), t_s(sf * 10, TF_BY_MIN[20]),
                  t_s(sf * 10, TF_BY_MIN[10]), t_s(sf * 10, TF_BY_MIN[2])),
                 (ths_z, ths_z, ths_y, ths_z, ths1), [r.no_fonts] * 5, [d1] * 5)

        # Degree Labels
        f = g.STH * 1.1 if is_tan else th
        range1, range2 = range(6, 16), range(16, 21)
        alt_col = s.fg_col(sc_alt.key, is_increasing=sc_alt.is_increasing)
        for x in chain(range1, range2, range(25, 41, 5), () if is_tan else range(50, 80, 10)):
            f_l = f_md2_i if x in range1 else f_mdn_i
            f_r = f_md2 if x in range1 else f_mdn
            x_coord = round(sc_t.pos_of(x, g) + 1.2 / 2 * s.sym_w(str(x), f_l))
            r.draw_numeral(x, y_off, sym_col, h, x_coord, f, f_r, al)
            if x not in range2:
                xi = angle_opp(x)
                x_coord_opp = round(sc_t.pos_of(x, g) - 1.4 / 2 * s.sym_w(str(xi), f_l))
                r.draw_numeral(xi, y_off, alt_col, h, x_coord_opp, f, f_l, al)

        r.draw_numeral(45 if is_tan else DEG_RT, y_off, sym_col, h, scale_w, f, f_lgn, al)

    elif sc == Scales.T2:
        pats([int(fp * sf) for fp in (45, 75, angle_opp(5.7))],
             (t_s(sf * 5, (5, 2, 5)), t_s(sf * 5, (5, 2, 10))),
             [ths4] * 2, [fonts_xl] * 2, [d0] * 2)

    elif sc == Scales.ST:
        pats([int(fp * sf) for fp in (0.57, 1, 2, 4, 5.8)],
             (t_s(sf, (20, 5, 2)), t_s(sf, TF_BY_MIN[100]), t_s(sf, TF_BY_MIN[50]), t_s(sf, TF_BY_MIN[20])),
             (ths1, ths5, ths5, ths5), [r.no_fonts] * 4, [d1] * 4)

        # Degree Labels
        r.draw_sym_al('1°', y_off, sym_col, h, sc.pos_of(1, g), th, f_lbl, al)
        for x in chain((x / 10 for x in range(6, 10)), (x + 0.5 for x in range(1, 4)), range(2, 6)):
            r.draw_numeral_sc(sc, x, y_off, sym_col, h, th, f_lbl, al)

    elif sc == custom_scale_sets['Merchant']['P%']:
        if DEBUG:
            sc.grad_pat_default(r, y_off, al)
        for pct_value in (0, 5, 10, 15, 20, 35, 30, 40):
            r.draw_numeral_sc(sc, pct_value, y_off, sym_col, h, 0, f_lgn, Align.LOWER)
            r.draw_numeral(pct_value, y_off, sym_col, h, sc.pos_of(-pct_value, g), 0, f_lgn, Align.LOWER)
        for sym, val in (('-50%', -50), ('50%', 50), ('-33⅓', -100/3), ('33⅓', 100/3), ('+100%', 100)):
            r.draw_sym_al(sym, y_off, sym_col, h, sc.pos_of(val, g), 0, f_lgn, Align.LOWER)

    elif sc == custom_scale_sets['Merchant']['ZZ%']:
        if DEBUG:
            sc.grad_pat_default(r, y_off, al)
        for x in range(1, 12):
            r.draw_numeral_sc(sc, x, y_off, sym_col, h, 0, f_lgn, Align.LOWER)
        for x in (x / 10 for x in range(15, 110, 10)):
            r.draw_numeral_sc(sc, x, y_off, sym_col, h, 0, f_lgn, Align.LOWER)

    else:  # Fallback to our generic algorithm, then fill in marks and numerals as helpful
        sc.grad_pat_default(r, y_off, al)

    if sc.marks:
        f_mark = f_lbl if sc.scaler in {ScaleFNs.Base, ScaleFNs.Inverse, ScaleFNs.Square} else f_lgn
        for mark in sc.marks:
            r.draw_mark(mark, y_off, sc, f_mark, al, sym_col, side=side)
    if sc.numerals:
        for x in sc.numerals:
            r.draw_numeral_sc(sc, x, y_off, sym_col, h, th, f_smn, al)


class Mode(Enum):
    RENDER, DIAGNOSTIC, STICKERPRINT = 'render', 'diagnostic', 'stickerprint'


def transcribe(src_img, dst_img, src_x: int, src_y: int, size_x: int, size_y: int, dst_x: int, dst_y: int):
    """Transfer a pixel rectangle from a SOURCE (for rendering) to DESTINATION (for stickerprint)"""
    assert size_x > 0
    assert size_y > 0
    if isinstance(src_img, svg.Drawing):
        for elem in src_img.elements:  # TODO: filter by the rectangle (as a viewport?)
            dst_img.append(elem)
    elif isinstance(src_img, Image.Image):
        src_box = src_img.crop((src_x, src_y, src_x + size_x, src_y + size_y))
        dst_img.paste(src_box, (dst_x, dst_y))


def image_for_rendering(model: Model, out_format: OutFormat, w=None, h=None):
    g = model.geometry
    if out_format == OutFormat.PNG:
        return Image.new('RGB', (int(w or g.total_w), int(h or g.print_h)), model.style.bg.value)
    elif out_format == OutFormat.SVG:
        drawing = svg.Drawing(int(w or g.total_w), int(h or g.print_h), id_prefix='def_')
        drawing.set_render_size(f'{drawing.width / Geometry.PixelsPerIN}in', f'{drawing.height / Geometry.PixelsPerIN}in')
        return drawing


def save_image(img_to_save, basename: str, output_suffix=None):
    output_filename = f"{basename}{'.' + output_suffix if output_suffix else ''}"
    output_full_path = os.path.abspath(output_filename)
    if isinstance(img_to_save, Image.Image):
        output_full_path += '.png'
        img_to_save.save(output_full_path, 'PNG')
    elif isinstance(img_to_save, svg.Drawing):
        output_full_path += '.svg'
        img_to_save.save_svg(output_full_path)
    print(f'Result saved to: file://{output_full_path}')


# ----------------------7. Commands------------------------------------------


def main():
    """CLI processor for rendering the models in various modes."""
    import argparse
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--mode',
                             choices=[m.value for m in Mode],
                             help='What to render')
    args_parser.add_argument('--model',
                             choices=list(Model.example_names()),
                             default='Demo',
                             help='Which sliderule model')
    args_parser.add_argument('--format',
                             default=OutFormat.PNG.value,
                             choices=[f.value for f in OutFormat],
                             help='Output format (PNG for sticker printing, SVG for laser tooling')
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
    mode: Mode = next(mode for mode in Mode if mode.value == cli_args.mode)
    out_format: OutFormat = next(f for f in OutFormat if f.value == cli_args.format)
    model_name = cli_args.model
    model = Model.load(model_name)
    output_suffix = cli_args.suffix or ('test' if cli_args.test else None)
    render_cutoffs = cli_args.cutoffs
    global DEBUG
    DEBUG = cli_args.debug
    SVGOut.init(debug=DEBUG)

    start_time = time.process_time()

    sliderule_img = None
    if mode in {Mode.RENDER, Mode.STICKERPRINT}:
        mode_render = mode == Mode.RENDER
        sliderule_img = render_sliderule_mode(model, out_format, sliderule_img,
                                              borders=mode_render, cutoffs=render_cutoffs)
        print(f'Slide Rule render finished at: {round(time.process_time() - start_time, 3)} seconds')
        if mode_render:
            save_image(sliderule_img, f'{model_name}.SlideRuleScales', output_suffix)

    if mode == Mode.DIAGNOSTIC:
        diagnostic_img = render_diagnostic_mode(model, out_format, all_scales=model != DemoModel)
        print(f'Diagnostic render finished at: {round(time.process_time() - start_time, 3)} seconds')
        save_image(diagnostic_img, f'{model_name}.Diagnostic', output_suffix)

    if mode == Mode.STICKERPRINT:
        stickerprint_img = render_stickerprint_mode(model, out_format, sliderule_img)
        print(f'Stickerprint render finished at: {round(time.process_time() - start_time, 3)} seconds')
        save_image(stickerprint_img, f'{model_name}.StickerCut', output_suffix)

    print(f'Program finished at: {round(time.process_time() - start_time, 3)} seconds')


def render_sliderule_mode(model: Model, out_format: OutFormat, sliderule_img=None, borders: bool = False, cutoffs: bool = False):
    if sliderule_img is None:
        sliderule_img = image_for_rendering(model, out_format)
    g, layout = model.geometry, model.layout
    y_front_start = g.oY
    r = Renderer.to_image(sliderule_img, g, model.style)
    y_rear_start = y_front_start + g.side_h + g.oY
    # Front Scale
    # Titling
    y_off = y_side_start = y_front_start
    if model == DemoModel:
        y_off_titling = 25 + y_off
        title_col = Color.RED
        f_lbl = model.style.font_for(FontSize.SC_LBL)
        side_w_q = g.side_w // 4
        for sym, x in ((model.name, side_w_q), (model.subtitle, side_w_q * 2 + g.oX), (model.brand, side_w_q * 3)):
            r.draw_sym_al(sym, y_off_titling, title_col, 0, x - g.li, 0, f_lbl, Align.UPPER)
        y_off = y_off_titling + f_lbl.size
    # Scales
    for side in Side:
        for part in RulePart:
            part_y_start = y_side_start + g.edge_h(part, True)
            part_y_end = part_y_start + g.part_h(part)
            part_scales = layout.scales_at(side, part)
            if not part_scales:
                y_off = part_y_end
                continue
            last_i = len(part_scales) - 1
            for i, sc in enumerate(part_scales):
                scale_h = g.scale_h(sc, side)
                scale_al = layout.scale_al(sc, side, part)
                # Handle edge-aligned scales:
                if i == 0 and scale_al == Align.UPPER:
                    y_off = part_y_start
                elif i == last_i and scale_al == Align.LOWER:
                    y_off = part_y_end - scale_h
                else:  # Incremental displacement by default
                    y_off += g.scale_margin(sc, side)
                if isinstance(sc, Scale):
                    gen_scale(r, y_off, sc, al=scale_al, side=side)
                elif isinstance(sc, Ruler):
                    sc.generate(r, y_off, scale_al, side)
                y_off += scale_h

        y_off = y_rear_start
        y_side_start = y_rear_start
        y_off += g.top_margin
    # Borders and (optionally) cutoffs
    if borders:
        for side in Side:
            y0 = y_front_start if side == Side.FRONT else y_rear_start
            r.draw_borders(y0, side)
            if cutoffs:
                r.draw_brace_pieces(y0, side)
    return sliderule_img


def render_stickerprint_mode(m: Model, out_format: OutFormat, sliderule_img: Image.Image):
    """Stickers break down by side, then part, then side cutoffs vs middle.
    18 total stickers: 2 sides x 3 parts x 3 bands."""
    o_x2, o_y2 = 50, 50  # margins
    o_a = 50  # overhang amount
    ext = 20  # extension amount
    g = m.geometry
    has_braces = isinstance(g.brace_shape, BraceShape)
    side_w, side_h, slide_h, stator_h = int(g.side_w), int(g.side_h), int(g.slide_h), int(g.stator_h)
    scale_x_margin = max(0, (side_w - (g.SL if m == DemoModel else m.max_scale_w())) // 2 - g.brace_w) + 30
    scale_w = side_w - scale_x_margin * 2
    total_w = scale_w + 2 * o_x2
    sticker_row_h = max(slide_h, stator_h) + o_a
    total_h = o_y2 * 2 + (side_h + 2 * o_a) * 2 + o_a * 3 + (sticker_row_h * 2 if has_braces else 0) + 45
    dst_img = image_for_rendering(m, out_format, w=total_w, h=total_h)
    r = Renderer.to_image(dst_img, replace(g, side_w=scale_w, oX=0, oY=0), m.style)
    y = o_y2
    # Middle band stickers:
    cut_col = Color.to_pil(Color.CUT)
    for (front, y_side) in ((True, g.oY), (False, g.oY + side_h + g.oY)):
        for (i, y_src, h) in ((0, 0, stator_h),
                              (1, stator_h + 1 - (0 if front else 3), slide_h),
                              (2, side_h - stator_h, stator_h)):
            y += o_a
            transcribe(sliderule_img, dst_img, g.oX + scale_x_margin, y_side + y_src, scale_w, h, o_x2, y)
            if i > 0:
                extend(dst_img, scale_w, y + 1, BleedDir.UP, ext)
            extend(dst_img, scale_w, y + h - 1, BleedDir.DOWN, ext)
            r.draw_corners(o_x2, y - o_a if i == 0 else y, scale_w, h if i == 1 else h + o_a, cut_col)
            y += h
        y += o_a + (o_a if front else 0)
    if has_braces:  # Side stickers:
        y_b = y + 20
        box_w = g.brace_w + 30
        boxes = [  # (x0, y0, dx, dy)
            (o_a,                              y_b, box_w + o_a,          stator_h + o_a),
            (box_w + 3 * o_a,                  y_b, scale_x_margin + o_a, slide_h),
            (box_w + 5 * o_a + scale_x_margin, y_b, scale_x_margin + o_a, stator_h + o_a)
        ]
        box_x_mirror = 13 * o_a // 2 + box_w + 2 * scale_x_margin
        for (x0, y0, dx, dy) in boxes:
            for x1 in (x0, 2 * box_x_mirror - x0 - dx):
                for y1 in (y0, y0 + slide_h + o_a):
                    r.r.draw_box(x1, y1, dx, dy, cut_col)
        scale_h = g.SH
        p1 = [
            (2 * o_a + 120,                                  y_b + o_a + scale_h),
            (6 * o_a + box_w + scale_x_margin + scale_h,     y_b + 2 * scale_h),
            (6 * o_a + box_w + scale_x_margin + 2 * scale_h, y_b + scale_h),
        ]
        points = p1 + [(cx, cy + sticker_row_h + (o_a if i > 0 else -o_a))
                       for (i, (cx, cy)) in enumerate(p1)]
        for (cx, cy) in points:
            for x in (cx, 2 * box_x_mirror - cx):
                r.r.draw_circle(x, cy, g.brace_hole_r, cut_col)
    return dst_img


def render_diagnostic_mode(model: Model, output_format: OutFormat, all_scales=False):
def render_diagnostic_mode(model: Model, out_format: OutFormat, all_scales=False):
    """
    Diagnostic mode, rendering scales independently.
    Works as a test of tick marks, labeling, and layout. Also, regressions.
    If you're reading this, you're a real one
    +5 brownie points to you
    """
    scale_h = model.geometry.SH
    k = 120 + scale_h
    layout, style = model.layout, model.style
    upper = Align.UPPER
    sh_with_margins = scale_h + (40 if model == DemoModel else 10)
    scale_names = ['A', 'B', 'C', 'D', 'K', 'R1', 'R2', 'CI', 'DI', 'CF', 'DF', 'CIF', 'L', 'S', 'T', 'ST']
    for sc_name in keys_of(Scales) + list(layout.sc_keys_in_order()) if all_scales else layout.sc_keys_in_order():
        if sc_name not in scale_names:
            scale_names.append(sc_name)
    total_h = k + (len(scale_names) + 1) * sh_with_margins + scale_h
    geom_d = Geometry.make(
        (6500, total_h),
        (250, 250),  # remove y-margin to stack scales
        (Geometry.SL, scale_h),
        slide_h=480
    )
    diagnostic_img = image_for_rendering(model, out_format, w=geom_d.total_w, h=total_h)
    r = Renderer.to_image(diagnostic_img, geom_d, style)
    title_x = geom_d.midpoint_x - geom_d.li
    title = 'Diagnostic Test Print of Available Scales'
    r.draw_sym_al(title, 50, style.fg, 0, title_x, 0, style.font_for(FontSize.TITLE), upper)
    sc_names_str = ' '.join(scale_names)
    r.draw_sym_al(sc_names_str, 200, style.fg, 0, title_x, 0,
                  style.font_for(FontSize.SUBTITLE, h_ratio=min(1.0, 100 / len(sc_names_str))), upper)
    if DEBUG:
        band_w = Scales.L.gen_fn(1) * geom_d.SL
        for i in range(-2, 12, 2):
            r.r.fill_rect(geom_d.li + band_w * i, k, band_w, total_h, Color.RED_WHITE_1)
    for n, sc_name in enumerate(scale_names):
        sc = layout.scale_named(sc_name)
        assert sc, f'Scale not found: {sc_name}'
        al = Align.LOWER if model == DemoModel else layout.scale_al(sc, Side.FRONT, RulePart.SLIDE)
        y_off = k + (n + 1) * sh_with_margins
        if isinstance(sc, Ruler):
            sc.generate(r, y_off, al)
        elif isinstance(sc, Scale):
            gen_scale(r, y_off, sc, al)
    return diagnostic_img


if __name__ == '__main__':
    main()
