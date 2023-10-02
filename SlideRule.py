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

BACKGROUND_COLOR = 'white'
CUT_COLOR = (0, 0, 255)  # color which indicates CUT (0,0,255 = blue)
BLACK = 'black'
RED = 'red'

oX = 100  # x margins
oY = 100  # y margins
total_width = 8000 + 2 * oX
sliderule_height = 1600 * 2 + 3 * oY

sliderule_img = Image.new('RGB', (total_width, sliderule_height), BACKGROUND_COLOR)
renderer = ImageDraw.Draw(sliderule_img)

SH = 160  # scale height

UPPER = 'upper'  # Lower alignment
LOWER = 'lower'  # Upper Alignment
SL = 5600  # scale length
li = round(total_width / 2 - SL / 2)  # left index offset from left edge

# Ticks, Labels, are referenced from li as to be consistent
STH = 70  # standard tick height
STT = 4  # standard tick thickness

# tick height scalars
XS = 0.5
SM = 0.85
MED = 1
MXL = 1.15
XL = 1.3

FRONT_SIDE = 'front'
REAR_SIDE = 'rear'
current_side = None
current_y0 = 0  # upper level starting point for various objects (ignore me)
# y_off = 100  # No longer global


BLACK_COLOR_RGB = (0, 0, 0)

# ----------------------2. Fundamental Functions----------------------------

def draw_tick(y_off, x, height, thickness, al):
    """
    Places an individual tick
    :param y_off: y pos
    :param x: offset of left edge from *left index*
    :param height: height of tickmark (measured from baseline or upperline)
    :param thickness: thickness of tickmark (measured from left edge)
    :param al: alignment
    """

    global current_y0, sliderule_img
    for T in range(0, thickness):
        for H in range(0, height + 1):
            if al == UPPER:
                current_y0 = H
            if al == LOWER:
                current_y0 = SH - 1 - H
            sliderule_img.putpixel((x + li + T - 2, current_y0 + y_off), BLACK_COLOR_RGB)
    if False:  # Replacement WIP
        global renderer
        renderer.rectangle(((x + li - 2, current_y0 - y_off), (x + li + thickness - 2, current_y0 + y_off)), fill=BLACK)


def pat(y_off, sc, S, iI, iF, a, b, e, a0, b0, shf, al):
    """
    Place ticks in a pattern
    a+bN (N ∈ Z) defines the patterning (which ticks to place)
    a0+b0N (N ∈ Z) defines the exclusion patterning (which ticks not to place)

    :param y_off: y pos
    :param str sc: scale key
    :param S: height modifier (input height scalar like xs, sm, med, lg)
    :param iI: starting index point (100 = left index)
    :param iF: ending index point (1001 = right index)
    :param a: offset from iI
    :param b: multiple value
    :param e: exclusion pattern? 1 = yes, 0 = no
    :param a0: offset from iI
    :param b0: multiple value; put placeholders like 1 & 1 in here if e == 0
    :param shf: scale shift amount
    :param al: alignment
    """

    for x in range(iI, iF):
        if e == 1:
            if x % b - a == 0 and x % b0 - a0 != 0:
                draw_tick(y_off, shf + scaling_fn(sc, x / 100), round(S * STH), STT, al)
        elif e == 0:
            if x % b - a == 0:
                draw_tick(y_off, shf + scaling_fn(sc, x / 100), round(S * STH), STT, al)


def font_for_family(font_style, font_size):
    """
    :param int font_style: font style (normal == 0 , italic == 1)
    :param int font_size: font size
    :return: FreeTypeFont
    """
    font_name = "cmunit.ttf" if font_style == 1 else "cmuntt.ttf"
    # font_name = "cmunrm.ttf" # mythical latex edition
    return ImageFont.truetype(font_name, font_size)


def get_size(symbol, font_size, font_style=0):
    """
    Gets the size dimensions (width, height) of the input text
    :param str symbol: the text
    :param int font_size: font size
    :param int font_style: font style (normal == 0 , italic == 1)
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
    :param int font_style: font style (normal == 0 , italic == 1)
    :return: int
    """
    w, h = get_size(s, font_size, font_style)
    return w


def get_height(s, font_size, font_style):
    """
    :param str s: symbol
    :param int font_size: font size
    :param int font_style: font style (normal == 0 , italic == 1)
    :return: int
    """
    w, h = get_size(s, font_size, font_style)
    return h


def draw_symbol(color, y_off, s, x, y, font_size, font_style, al):
    """
    :param str color: color name that PIL recognizes
    :param y_off: y pos
    :param str|int s: symbol
    :param x: offset of centerline from left index (li)
    :param y: offset of base from baseline (LOWER) or top from upperline (UPPER)
    :param int font_size: font size
    :param int font_style: font style
    :param str al: alignment
    """

    if color == 'green':  # Override PIL for green for slide rule symbol conventions
        color = '#228B1E'

    font = font_for_family(font_style, font_size)
    w, h = get_size(s, font_size, font_style)

    global current_y0, renderer
    if al == UPPER:
        current_y0 = y
    if al == LOWER:
        current_y0 = SH - 1 - y - h * 1.2
        # current_y0 = SH - 29 - y - h * 1.2  # FIXME hack to lower-align text better
    renderer.text((x + li - round(w / 2) + round(STT / 2), current_y0 + y_off), str(s), font=font, fill=color)


def extend(image, y, direction, A):
    """
    Used to create bleed for sticker cutouts
    :param image: e.g. img, img2, etc.
    :param int y: y pixel row to duplicate
    :param str direction: direction ('up','down')
    :param int A: Amplitude (# of pixels to extend)
    """

    for x in range(0, total_width):
        r, g, b = image.getpixel((x, y))

        if direction == DIR_UP:
            for yi in range(y - A, y):
                image.putpixel((x, yi), (r, g, b))

        if direction == DIR_DOWN:
            for yi in range(y, y + A):
                image.putpixel((x, yi), (r, g, b))


# ----------------------3. Scale Generating Function----------------------------

def gen_scale(y_off, sc, al):
    """
    :param int y_off: y pos
    :param str sc: scale; one of `SCALE_NAMES`
    :param str al: alignment; one of `UPPER` or `LOWER`
    """
    # Scale Symbol Labels
    shift = 0  # scale shift from left index
    left_sym = ""  # left scale symbol
    right_sym = ""  # right scale symbol
    col = BLACK  # symbol color

    if sc == 'A':
        shift = 0
        left_sym = 'A'
        right_sym = 'x²'
        col = BLACK
    if sc == 'B':
        shift = 0
        left_sym = 'B'
        right_sym = 'x²'
        col = BLACK
    if sc == 'C':
        shift = 0
        left_sym = 'C'
        right_sym = 'x'
        col = BLACK
    if sc == 'D':
        shift = 0
        left_sym = 'D'
        right_sym = 'x'
        col = BLACK
    if sc == 'K':
        shift = 0
        left_sym = 'K'
        right_sym = 'x³'
        col = BLACK
    if sc == 'R1':
        shift = 0
        left_sym = 'R'
        right_sym = '√x'
        col = BLACK
    if sc == 'R2':
        shift = -SL
        left_sym = 'R'
        right_sym = '√x'
        col = BLACK
    if sc == 'CI':
        shift = 0
        left_sym = 'CI'
        right_sym = '1/x'
        col = RED
    if sc == 'DI':
        shift = 0
        left_sym = 'DI'
        right_sym = '1/x'
        col = RED
    if sc == 'CF':
        shift = round(SL * (1 - math.log10(math.pi)))
        left_sym = 'CF'
        right_sym = 'πx'
        col = BLACK
    if sc == 'DF':
        shift = round(SL * (1 - math.log10(math.pi)))
        left_sym = 'DF'
        right_sym = 'πx'
        col = BLACK
    if sc == 'CIF':
        shift = 0
        left_sym = 'CIF'
        right_sym = '1/πx'
        col = RED
    if sc == 'L':
        shift = 0
        left_sym = 'L'
        right_sym = 'log x'
        col = BLACK
    if sc == 'S':
        shift = 0
        left_sym = 'S'
        right_sym = 'sin x'
        col = BLACK
    if sc == 'T':
        shift = 0
        left_sym = 'T'
        right_sym = 'tan x'
        col = BLACK
    if sc == 'ST':
        shift = 0
        left_sym = 'ST'
        right_sym = 'θ<5.7°'
        col = BLACK

    # Place Index Symbols (Left and Right)
    font_size = 90
    (w2, h2) = get_size(right_sym, font_size, REG)
    draw_symbol(col, y_off, right_sym, 102 / 100 * SL + 0.5 * w2, (SH - h2) / 2, font_size, REG, al)
    (w1, h1) = get_size(left_sym, font_size, REG)
    draw_symbol(col, y_off, left_sym, -2 / 100 * SL - 0.5 * w1, (SH - h1) / 2, font_size, REG, al)

    # Exceptions / Special Symbols for R1, R2, S, and T
    if sc == 'R1':
        if al == LOWER:
            draw_symbol(BLACK, y_off, 1, -2 / 100 * SL + 0.5 * w1,
                        SH - 1.3 * ((SH - h1) / 2 + 0.75 * h1), 60, 0, al)
        if al == UPPER:
            draw_symbol(BLACK, y_off, 1, -2 / 100 * SL + 0.5 * w1,
                        (SH - get_height(left_sym, font_size, REG)) / 2 + 0.75 * h1, 60, 0, al)
    if sc == 'R2':
        if al == LOWER:
            draw_symbol(BLACK, y_off, 2, -2 / 100 * SL + 0.5 * w1,
                        SH - 1.3 * ((SH - h1) / 2 + 0.75 * h1), 60, 0, al)
        if al == UPPER:
            draw_symbol(BLACK, y_off, 2, -2 / 100 * SL + 0.5 * w1,
                        (SH - h1) / 2 + 0.75 * h1, 60, 0, al)
    if sc == 'S':
        draw_symbol(RED, y_off, 'C', -2 / 100 * SL - 0.5 * w1 - get_width('_S', font_size, REG),
                    (SH - h2) / 2, font_size, REG, al)
    if sc == 'T':
        draw_symbol(RED, y_off, 'T', -2 / 100 * SL - 0.5 * w1 - get_width('_T', font_size, REG),
                    (SH - h2) / 2, font_size, REG, al)

    # Tick Placement (the bulk!)
    if sc == "C" or sc == "D" or sc == "CI" or sc == "DI":

        # Ticks
        pat(y_off, sc, MED, 100, 1001, 0, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, XL, 100, 1001, 50, 100, 1, 150, 1000, 0, al)
        pat(y_off, sc, SM, 100, 1001, 0, 10, 1, 150, 100, 0, al)
        pat(y_off, sc, SM, 100, 200, 5, 10, 0, 1, 1, 0, al)
        pat(y_off, sc, XS, 100, 200, 0, 1, 1, 0, 5, 0, al)
        pat(y_off, sc, XS, 200, 400, 0, 2, 1, 0, 10, 0, al)
        pat(y_off, sc, XS, 400, 1001, 0, 5, 1, 0, 10, 0, al)

        # 1-10 Labels
        for x in range(1, 11):
            if x == 10:
                draw_symbol(col, y_off, 1, scaling_fn(sc, x), STH, font_size, REG, al)
            else:
                draw_symbol(col, y_off, x, scaling_fn(sc, x), STH, font_size, REG, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            draw_symbol(col, y_off, x - 10, scaling_fn(sc, x / 10), round(STH * 0.85), 60, REG, al)

        # Gauge Points
        draw_tick(y_off, scaling_fn(sc, math.pi), round(STH), STT, al)
        draw_symbol(col, y_off, 'π', scaling_fn(sc, math.pi), round(STH), font_size, REG, al)

    if sc == "C" or sc == "D":
        if y_off < 1600 + oY:
            # r Gauge Point
            draw_tick(y_off, scaling_fn(sc, 18 / math.pi), round(STH), STT, al)
            draw_symbol(BLACK, y_off, 'r', scaling_fn(sc, 18 / math.pi), round(STH), font_size, REG, al)

    if sc == "A" or sc == "B":

        # Ticks
        pat(y_off, sc, MED, 100, 1001, 0, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, MED, 1000, 10001, 0, 1000, 0, 1, 1, 0, al)
        pat(y_off, sc, SM, 100, 501, 0, 10, 1, 50, 100, 0, al)
        pat(y_off, sc, SM, 1000, 5001, 0, 100, 1, 500, 1000, 0, al)
        pat(y_off, sc, XL, 100, 1001, 50, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, XL, 1000, 10001, 500, 1000, 0, 1, 1, 0, al)
        pat(y_off, sc, XS, 100, 200, 0, 2, 0, 1, 1, 0, al)
        pat(y_off, sc, XS, 1000, 2000, 0, 20, 0, 1, 1, 0, al)
        pat(y_off, sc, XS, 200, 500, 5, 10, 0, 1, 1, 0, al)
        pat(y_off, sc, XS, 2000, 5000, 50, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, XS, 500, 1001, 0, 10, 1, 0, 50, 0, al)
        pat(y_off, sc, XS, 5000, 10001, 0, 100, 1, 0, 500, 0, al)

        # 1-10 Labels
        for x in range(1, 11):
            if x == 10:
                draw_symbol(BLACK, y_off, 1, scaling_fn(sc, x), STH, font_size, REG, al)
                draw_symbol(BLACK, y_off, 1, scaling_fn(sc, x * 10), STH, font_size, REG, al)
            else:
                draw_symbol(BLACK, y_off, x, scaling_fn(sc, x), STH, font_size, REG, al)
                draw_symbol(BLACK, y_off, x, scaling_fn(sc, x * 10), STH, font_size, REG, al)

        # Gauge Points
        draw_tick(y_off, scaling_fn(sc, math.pi), round(STH), STT, al)
        draw_symbol(BLACK, y_off, 'π', scaling_fn(sc, math.pi), round(STH), font_size, REG, al)

    if sc == "K":
        for b in range(0, 3):
            # Ticks
            pat(y_off, sc, MED, 100 * (10 ** b), 1000 * (10 ** b) + 1, 0, 100 * (10 ** b), 0, 1, 1, 0, al)
            pat(y_off, sc, XL, 100 * (10 ** b), 600 * (10 ** b) + 1, 50 * (10 ** b), 100 * (10 ** b), 0, 1, 1, 0, al)
            pat(y_off, sc, SM, 100 * (10 ** b), 300 * (10 ** b) + 1, 0, 10 * (10 ** b), 0, 1, 1, 0, al)
            pat(y_off, sc, XS, 100 * (10 ** b), 300 * (10 ** b) + 1, 5 * (10 ** b), 10 * (10 ** b), 0, 1, 1, 0, al)
            pat(y_off, sc, XS, 300 * (10 ** b), 600 * (10 ** b) + 1, 0, 10 * (10 ** b), 0, 1, 1, 0, al)
            pat(y_off, sc, XS, 600 * (10 ** b), 1000 * (10 ** b) + 1, 0, 20 * (10 ** b), 0, 1, 1, 0, al)

        # 1-10 Labels
        f = 75
        for x in range(1, 11):
            if x == 10:
                draw_symbol(BLACK, y_off, 1, scaling_fn(sc, x), STH, f, 0, al)
                draw_symbol(BLACK, y_off, 1, scaling_fn(sc, x * 10), STH, f, 0, al)
                draw_symbol(BLACK, y_off, 1, scaling_fn(sc, x * 100), STH, f, 0, al)
            else:
                draw_symbol(BLACK, y_off, x, scaling_fn(sc, x), STH, f, 0, al)
                draw_symbol(BLACK, y_off, x, scaling_fn(sc, x * 10), STH, f, 0, al)
                draw_symbol(BLACK, y_off, x, scaling_fn(sc, x * 100), STH, f, 0, al)

    if sc == 'R1':

        # Ticks
        pat(y_off, sc, MED, 1000, 3200, 0, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, XL, 1000, 2000, 0, 50, 1, 0, 100, 0, al)
        pat(y_off, sc, SM, 2000, 3200, 0, 50, 0, 0, 1000, 0, al)
        pat(y_off, sc, SM, 1000, 2000, 0, 10, 1, 0, 50, 0, al)
        pat(y_off, sc, XS, 1000, 2000, 5, 10, 0, 1, 1, 0, al)
        pat(y_off, sc, XS, 2000, 3180, 0, 10, 1, 0, 50, 0, al)

        # 1-10 Labels
        for x in range(1, 4):
            draw_symbol(BLACK, y_off, x, scaling_fn(sc, 10 * x), STH, font_size, REG, al)

        # 0.1-3.1 Labels
        for x in range(11, 20):
            draw_symbol(BLACK, y_off, x - 10, scaling_fn(sc, x), STH, 60, 0, al)
        for x in range(21, 30):
            draw_symbol(BLACK, y_off, x - 20, scaling_fn(sc, x), STH, 60, 0, al)
        draw_symbol(BLACK, y_off, 1, scaling_fn(sc, 31), STH, 60, 0, al)

        # draw_tick(y_off,sl,round(sth),stt)

    if sc == 'R2':

        # Ticks
        pat(y_off, sc, MED, 4000, 10001, 0, 1000, 0, 1, 1, shift, al)
        pat(y_off, sc, XL, 5000, 10000, 500, 1000, 0, 1, 1, shift, al)
        pat(y_off, sc, SM, 3200, 10000, 0, 100, 1, 0, 1000, shift, al)
        pat(y_off, sc, SM, 3200, 5000, 0, 50, 0, 1, 1, shift, al)
        pat(y_off, sc, XS, 3160, 5000, 0, 10, 1, 0, 50, shift, al)
        pat(y_off, sc, XS, 5000, 10000, 0, 20, 1, 0, 100, shift, al)

        # 1-10 Labels
        for x in range(4, 10):
            draw_symbol(BLACK, y_off, x, scaling_fn(sc, 10 * x) + shift, STH, font_size, REG, al)
        draw_symbol(BLACK, y_off, 1, SL, STH, font_size, REG, al)

        # 0.1-3.1 Labels
        for x in range(32, 40):
            draw_symbol(BLACK, y_off, x % 10, scaling_fn(sc, x) + shift, STH, 60, REG, al)
        for x in range(41, 50):
            draw_symbol(BLACK, y_off, x % 10, scaling_fn(sc, x) + shift, STH, 60, REG, al)

    if sc == "CF" or sc == "DF":

        # Ticks
        pat(y_off, sc, MED, 100, 301, 0, 100, 0, 1, 1, shift, al)
        pat(y_off, sc, MED, 400, 1001, 0, 100, 0, 1, 1, -1 * SL + shift, al)
        pat(y_off, sc, XL, 200, 301, 50, 100, 0, 1, 1, shift, al)
        pat(y_off, sc, SM, 100, 201, 0, 5, 0, 1, 1, shift, al)
        pat(y_off, sc, SM, 200, 311, 0, 10, 0, 1, 1, shift, al)
        pat(y_off, sc, XL, 320, 1001, 50, 100, 0, 150, 1000, -1 * SL + shift, al)
        pat(y_off, sc, SM, 320, 1001, 0, 10, 1, 150, 100, -1 * SL + shift, al)
        pat(y_off, sc, XS, 100, 201, 0, 1, 1, 0, 5, shift, al)
        pat(y_off, sc, XS, 200, 314, 0, 2, 1, 0, 10, shift, al)
        pat(y_off, sc, XS, 316, 401, 0, 2, 1, 0, 10, -1 * SL + shift, al)
        pat(y_off, sc, XS, 400, 1001, 0, 5, 1, 0, 10, -1 * SL + shift, al)

        # 1-10 Labels
        for x in range(1, 4):
            draw_symbol(BLACK, y_off, x, scaling_fn(sc, x) + shift, STH, font_size, REG, al)
        for x in range(4, 10):
            draw_symbol(BLACK, y_off, x, scaling_fn(sc, x) - SL + shift, STH, font_size, REG, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            draw_symbol(BLACK, y_off, x - 10, scaling_fn(sc, x / 10) + shift, round(STH * 0.85), 60, REG, al)

            # Gauge Points
        draw_tick(y_off, scaling_fn(sc, math.pi) + shift, round(STH), STT, al)
        draw_symbol(BLACK, y_off, 'π', scaling_fn(sc, math.pi) + shift, round(STH), font_size, REG, al)
        draw_tick(y_off, scaling_fn(sc, math.pi) - SL + shift, round(STH), STT, al)
        draw_symbol(BLACK, y_off, 'π', scaling_fn(sc, math.pi) - SL + shift, round(STH), font_size, REG, al)

    if sc == 'CIF':

        # Ticks
        pat(y_off, sc, MED, 100, 301, 0, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, MED, 400, 1001, 0, 100, 0, 1, 1, SL, al)

        pat(y_off, sc, XL, 200, 301, 50, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, SM, 100, 201, 0, 5, 0, 1, 1, 0, al)
        pat(y_off, sc, SM, 200, 321, 0, 10, 0, 1, 1, 0, al)
        pat(y_off, sc, XL, 320, 1001, 50, 100, 0, 150, 1000, SL, al)
        pat(y_off, sc, SM, 310, 1001, 0, 10, 1, 150, 100, SL, al)
        pat(y_off, sc, XS, 100, 201, 0, 1, 1, 0, 5, 0, al)
        pat(y_off, sc, XS, 200, 321, 0, 2, 1, 0, 10, 0, al)
        pat(y_off, sc, XS, 310, 401, 0, 2, 1, 0, 10, SL, al)
        pat(y_off, sc, XS, 400, 1001, 0, 5, 1, 0, 10, SL, al)

        # 1-10 Labels
        for x in range(4, 10):
            draw_symbol(RED, y_off, x, scaling_fn(sc, x) + SL, STH, font_size, REG, al)
        for x in range(1, 4):
            draw_symbol(RED, y_off, x, scaling_fn(sc, x), STH, font_size, REG, al)

        # 0.1-0.9 Labels
        for x in range(11, 20):
            draw_symbol(RED, y_off, x - 10, scaling_fn(sc, x / 10), round(STH * 0.85), 60, REG, al)

    if sc == 'L':

        # Ticks
        pat(y_off, sc, MED, 0, 1001, 0, 10, 1, 50, 50, 0, al)
        pat(y_off, sc, XL, 1, 1001, 50, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, MXL, 0, 1001, 0, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, XS, 1, 1001, 0, 2, 1, 0, 50, 0, al)

        # Labels
        for x in range(0, 11):
            if x == 0:
                draw_symbol(BLACK, y_off, 0, scaling_fn(sc, x), STH, font_size, REG, al)
            if x == 10:
                draw_symbol(BLACK, y_off, 1, scaling_fn(sc, x), STH, font_size, REG, al)
            elif x in range(1, 10):
                draw_symbol(BLACK, y_off, '.' + str(x), scaling_fn(sc, x), STH, font_size, REG, al)

    if sc == 'S':

        # Ticks
        pat(y_off, sc, XL, 1000, 7001, 0, 1000, 0, 1, 1, 0, al)
        pat(y_off, sc, MED, 7000, 10001, 0, 1000, 0, 1, 1, 0, al)
        pat(y_off, sc, XL, 600, 2001, 0, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, SM, 600, 2000, 50, 100, 1, 0, 100, 0, al)
        pat(y_off, sc, XL, 2000, 6000, 500, 1000, 1, 0, 1000, 0, al)
        pat(y_off, sc, SM, 2000, 6000, 0, 100, 1, 0, 500, 0, al)
        pat(y_off, sc, XS, 570, 2000, 0, 10, 1, 0, 50, 0, al)
        pat(y_off, sc, XS, 2000, 3000, 0, 20, 1, 0, 100, 0, al)
        pat(y_off, sc, XS, 3000, 6000, 0, 50, 1, 0, 100, 0, al)
        pat(y_off, sc, SM, 6000, 8501, 500, 1000, 0, 1, 1, 0, al)
        pat(y_off, sc, XS, 6000, 8000, 0, 100, 0, 1, 1, 0, al)

        # Degree Labels

        for x in range(6, 16):
            width50italic = get_width(x, 50, ITALIC)
            draw_symbol(BLACK, y_off, str(x), scaling_fn(sc, x) + 1.2 / 2 * width50italic, STH, 50, REG, al)
            draw_symbol(RED, y_off, str(font_size - x), scaling_fn(sc, x) - 1.4 / 2 * get_width(90 - x, 50, ITALIC), STH, 50, ITALIC, al)

        for x in range(16, 20):
            draw_symbol(BLACK, y_off, str(x), scaling_fn(sc, x) + 1.2 / 2 * get_width(x, 55, ITALIC), STH, 55, REG, al)

        for x in range(20, 71, 5):
            if (x % 5 == 0 and x < 40) or x % 10 == 0:
                draw_symbol(BLACK, y_off, str(x), scaling_fn(sc, x) + 1.2 / 2 * get_width(x, 55, ITALIC), STH, 55, REG, al)
                if x != 20:
                    if font_size - x != 40:
                        draw_symbol(RED, y_off, str(font_size - x), scaling_fn(sc, x) - 1.4 / 2 * get_width(90 - x, 55, ITALIC), STH, 55, ITALIC, al)
                    if font_size - x == 40:
                        draw_symbol(RED, y_off + 11, str(40), scaling_fn(sc, x) - 1.4 / 2 * get_width(90 - x, 55, ITALIC), STH, 55, ITALIC, al)

        draw_symbol(BLACK, y_off, font_size, SL, STH, 60, 0, al)

    if sc == 'T':

        # Ticks
        pat(y_off, sc, XL, 600, 2501, 0, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, XL, 600, 1001, 50, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, XL, 2500, 4501, 0, 500, 0, 1, 1, 0, al)
        pat(y_off, sc, MED, 2500, 4501, 0, 100, 0, 1, 1, 0, al)
        draw_tick(y_off, SL, round(STH), STT, al)
        pat(y_off, sc, MED, 600, 951, 50, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, SM, 570, 1001, 0, 10, 1, 0, 50, 0, al)
        pat(y_off, sc, SM, 1000, 2500, 50, 100, 0, 1, 1, 0, al)
        pat(y_off, sc, XS, 570, 1001, 5, 10, 1, 0, 10, 0, al)
        pat(y_off, sc, XS, 1000, 2500, 0, 10, 1, 0, 50, 0, al)
        pat(y_off, sc, XS, 2500, 4501, 0, 20, 1, 0, 100, 0, al)

        # Degree Labels
        f = 1.1
        for x in range(6, 16):
            draw_symbol(BLACK, y_off, str(x), scaling_fn(sc, x) + 1.2 / 2 * get_width(x, 50, ITALIC), f * STH, 50, REG, al)
            draw_symbol(RED, y_off, str(font_size - x), scaling_fn(sc, x) - 1.4 / 2 * get_width(90 - x, 50, ITALIC), f * STH, 50, ITALIC, al)

        for x in range(16, 21):
            draw_symbol(BLACK, y_off, str(x), scaling_fn(sc, x) + 1.2 / 2 * get_width(x, 55, ITALIC), f * STH, 55, REG, al)

        for x in range(25, 41, 5):
            if x % 5 == 0:
                draw_symbol(BLACK, y_off, str(x), scaling_fn(sc, x) + 1.2 / 2 * get_width(x, 55, ITALIC), f * STH, 55, REG, al)
                draw_symbol(RED, y_off, str(font_size - x), scaling_fn(sc, x) - 1.4 / 2 * get_width(90 - x, 55, ITALIC), f * STH, 55, ITALIC, al)

        draw_symbol(BLACK, y_off, 45, SL, f * STH, 60, REG, al)

    if sc == 'ST':

        # Ticks
        pat(y_off, sc, MED, 100, 551, 0, 50, 0, 1, 1, 0, al)
        pat(y_off, sc, 1.2, 60, 100, 0, 10, 0, 1, 1, 0, al)
        pat(y_off, sc, XL, 60, 100, 5, 10, 0, 1, 1, 0, al)
        pat(y_off, sc, MED, 100, 200, 0, 10, 1, 0, 50, 0, al)
        pat(y_off, sc, SM, 200, 590, 0, 10, 0, 1, 1, 0, al)
        pat(y_off, sc, SM, 57, 100, 0, 1, 0, 1, 1, 0, al)
        pat(y_off, sc, SM, 100, 200, 0, 5, 0, 1, 1, 0, al)
        pat(y_off, sc, XS, 100, 200, 0, 1, 1, 0, 5, 0, al)
        pat(y_off, sc, XS, 200, 400, 0, 2, 1, 0, 10, 0, al)
        pat(y_off, sc, XS, 400, 585, 5, 10, 0, 1, 1, 0, al)

        for x in range(570, 1000):
            if x % 5 == 0 and x % 10 - 0 != 0:
                draw_tick(y_off, scaling_fn(sc, x / 1000), round(XS * STH), STT, al)

        # Degree Labels
        draw_symbol(BLACK, y_off, '1°', scaling_fn(sc, 1), STH, font_size, REG, al)
        for x in range(6, 10):
            draw_symbol(BLACK, y_off, "." + str(x), scaling_fn(sc, x / 10), STH, font_size, REG, al)
        for x in range(1, 4):
            draw_symbol(BLACK, y_off, str(x + 0.5), scaling_fn(sc, x + 0.5), STH, font_size, REG, al)
        for x in range(2, 6):
            draw_symbol(BLACK, y_off, str(x), scaling_fn(sc, x), STH, font_size, REG, al)


# ----------------------4. Line Drawing Functions----------------------------

# These functions are unfortunately difficult to modify,
# since I built them with specific numbers rather than variables

def draw_borders(y0):  # Place initial borders around scales y0 = vertical offset

    # Main Frame
    horizontals = [y0, 479 + y0, 1119 + y0, 1598 + y0]

    for i in range(0, 4):
        start = horizontals[i]
        for x in range(oX, total_width - oX):
            for y in range(start, start + 2):
                sliderule_img.putpixel((x, y), BLACK_COLOR_RGB)
    verticals = [oX, total_width - oX]
    for i in range(0, 2):
        start = verticals[i]
        for x in range(start, start + 2):
            for y in range(y0, 1600 + y0):
                sliderule_img.putpixel((x, y), BLACK_COLOR_RGB)

    # Top Stator Cut-outs
    verticals = [240 + oX, (total_width - 240) - oX]

    global current_side
    # if current_side == FRONT_SIDE:
    y_start = y0
    if current_side == REAR_SIDE:
        y_start = y_start + 1120
    y_end = 480 + y_start
    for i in range(0, 2):
        start = verticals[i]
        for x in range(start, start + 2):
            for y in range(y_start, y_end):
                sliderule_img.putpixel((x, y), BLACK_COLOR_RGB)


def draw_metal_cutoffs(y0):
    """
    Use to temporarily view the metal bracket locations
    :param y0: vertical offset
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
    if current_side == REAR_SIDE:
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

VALID_MODES = ['render', 'diagnostic', 'stickerprint']


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


def scaling_fn(sc, x):
    """
    Generating Function for the Scales
    :param Number x:
    :param str sc: one of `SCALE_NAMES`
    :return: int
    """
    assert sc in SCALE_NAMES
    if sc == 'C' or sc == 'D' or sc == 'CF' or sc == 'DF':
        return round(SL * math.log10(x))
    if sc == 'A' or sc == 'B':
        return round(1 / 2 * SL * math.log10(x))
    if sc == 'R1' or sc == 'R2':
        return round(2 * SL * math.log10(x / 10))
    if sc == 'K':
        return round(1 / 3 * SL * math.log10(x))
    if sc == 'CI' or sc == 'DI':
        return round(SL * (1 - math.log10(x)))
    if sc == 'CIF':
        return round(SL * (1 - math.log10(math.pi) - math.log10(x)))
    if sc == 'L':
        return round(SL * x / 10)
    if sc == 'S':
        return round(SL * math.log10(10 * math.sin(math.radians(x))))
    if sc == 'T':
        return round(SL * math.log10(10 * math.tan(math.radians(x))))
    if sc == 'ST':
        return round(SL * math.log10(100 * (math.sin(math.radians(x)) + math.tan(math.radians(x))) / 2))


# ---------------------- 6. Stickers -----------------------------


should_delineate: bool = True


def draw_box(image, x0, y0, dx, dy):
    """
    :param image:
    :param x0: First corner of box
    :param y0: First corner of box
    :param dx: width
    :param dy: height
    :return:
    """
    if should_delineate:
        # (x0,y0) First corner of box
        # dx, dy extension of box in positive direction

        for x in range(x0, x0 + dx):
            image.putpixel((x, y0), CUT_COLOR)
            image.putpixel((x, y0 + dy), CUT_COLOR)

        for y in range(y0, y0 + dy):
            image.putpixel((x0, y), CUT_COLOR)
            image.putpixel((x0 + dx, y), CUT_COLOR)


wE = 20  # width of extension cross arms


def draw_corners(image, x1, y1, x2, y2):
    if should_delineate:
        # (x1,y1) First corner of box
        # (x2,y2) Second corner of box

        for x in range(x1 - wE, x1 + wE):
            image.putpixel((x, y1), CUT_COLOR)
            image.putpixel((x, y2), CUT_COLOR)
        for x in range(x2 - wE, x2 + wE):
            image.putpixel((x, y1), CUT_COLOR)
            image.putpixel((x, y2), CUT_COLOR)
        for y in range(y1 - wE, y1 + wE):
            image.putpixel((x1, y), CUT_COLOR)
            image.putpixel((x2, y), CUT_COLOR)
        for y in range(y2 - wE, y2 + wE):
            image.putpixel((x1, y), CUT_COLOR)
            image.putpixel((x2, y), CUT_COLOR)


def transcribe(src_img, dest_img, x0, y0, dx, dy, xT, yT):
    """
    (x0,y0) First corner of SOURCE (rendering)
    (dx,dy) Width and Length of SOURCE chunk to transcribe
    (xT,yT) Target corner of DESTINATION; where to in-plop (into stickerprint)

    Note to self: this is such a bad way to do this, instead of
    transcribing over literally thousands of pixels I should have
    just generated the scales in the place where they are needed

    :param src_img: SOURCE of pixels
    :param dest_img: DESTINATION of pixels
    :param x0: First corner of SOURCE (rendering)
    :param y0: First corner of SOURCE (rendering)
    :param dx: Width of SOURCE chunk to transcribe
    :param dy: Length of SOURCE chunk to transcribe
    :param xT: Target corner of DESTINATION; where to in-plop (into stickerprint)
    :param yT: Target corner of DESTINATION; where to in-plop (into stickerprint)
    :return:
    """

    for x in range(0, dx):
        for y in range(0, dy):
            r, g, b = src_img.getpixel((x0 + x, y0 + y))
            dest_img.putpixel((xT + x, yT + y), (r, g, b))


def save_png(img_to_save, basename, output_suffix=None):
    output_filename = f"{basename}{'.'+output_suffix if output_suffix else ''}.png"
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
    args_parser.add_argument('--cutoffs',
                             action='store_true',
                             help='Render the metal cutoffs')
    cli_args = args_parser.parse_args()
    render_mode = cli_args.mode or prompt_for_mode()
    output_suffix = cli_args.suffix
    render_cutoffs = cli_args.cutoffs

    start_time = time.time()

    if render_mode == "render" or render_mode == "stickerprint":
        global current_side

        if render_mode == 'render':
            current_side = FRONT_SIDE
            draw_borders(oY)
            if render_cutoffs:
                draw_metal_cutoffs(oY)
            current_side = REAR_SIDE
            draw_borders(1600 + 2 * oY)
            if render_cutoffs:
                draw_metal_cutoffs(1600 + 2 * oY)

        # Front Scale
        gen_scale(110 + oY, 'L', LOWER)
        gen_scale(320 + oY, 'DF', LOWER)
        gen_scale(800 + oY, 'CI', LOWER)
        gen_scale(960 + oY, 'C', LOWER)

        gen_scale(480 + oY, 'CF', UPPER)
        gen_scale(640 + oY, 'CIF', UPPER)
        gen_scale(1120 + oY, 'D', UPPER)
        gen_scale(1280 + oY, 'R1', UPPER)
        gen_scale(1435 + oY, 'R2', UPPER)

        # These are my weirdo alternative universe "brand names", "model name", etc.
        # Feel free to comment them out
        global total_width, li
        draw_symbol(RED, 25 + oY, 'BOGELEX 1000', (total_width - 2 * oX) * 1 / 4 - li, 0, 90, REG, UPPER)
        draw_symbol(RED, 25 + oY, 'LEFT HANDED LIMAÇON 2020', (total_width - 2 * oX) * 2 / 4 - li + oX, 0, 90, REG, UPPER)
        draw_symbol(RED, 25 + oY, 'KWENA & TOOR CO.', (total_width - 2 * oX) * 3 / 4 - li, 0, 90, REG, UPPER)

        # Back Scale
        gen_scale(110 + 1600 + 2 * oY, 'K', LOWER)
        gen_scale(320 + 1600 + 2 * oY, 'A', LOWER)
        gen_scale(640 + 1600 + 2 * oY, 'T', LOWER)
        gen_scale(800 + 1600 + 2 * oY, 'ST', LOWER)
        gen_scale(960 + 1600 + 2 * oY, 'S', LOWER)

        gen_scale(480 + 1600 + 2 * oY, 'B', UPPER)
        gen_scale(1120 + 1600 + 2 * oY, 'D', UPPER)
        gen_scale(1360 + 1600 + 2 * oY, 'DI', UPPER)

    if render_mode == 'render':
        save_png(sliderule_img, 'SlideRuleScales', output_suffix)

    if render_mode == "diagnostic":
        global renderer
        # If you're reading this, you're a real one
        # +5 brownie points to you

        oX = 0  # x dir margins
        oY = 0  # y dir margins
        total_width = 7000
        total_height = 160 * 24
        li = round(total_width / 2 - SL / 2)  # update left index
        diagnostic_img = Image.new('RGB', (total_width, total_height), BACKGROUND_COLOR)
        renderer = ImageDraw.Draw(diagnostic_img)

        draw_symbol(BLACK, 50 + oY, 'Diagnostic Test Print of Available Scales', total_width / 2 - li, 0, 140, REG, UPPER)
        draw_symbol(BLACK, 200 + oY, ' '.join(SCALE_NAMES), total_width / 2 - li, 0, 120, REG, UPPER)
        k = 120 + SH

        for n, sc in enumerate(SCALE_NAMES):
            gen_scale(k + n * 200, sc, LOWER)

        save_png(diagnostic_img, 'Diagnostic', output_suffix)

    if render_mode == 'stickerprint':
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
        total_width = 6500 + 2 * oX2
        total_height = 5075

        stickerprint_img = Image.new('RGB', (total_width, total_height), BACKGROUND_COLOR)
        renderer = ImageDraw.Draw(stickerprint_img)

        # fsUM,MM,LM:
        l = 0

        l = oY2 + oA
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY, 6500, 480, oX2, l)
        extend(stickerprint_img, l + 480 - 1, DIR_DOWN, ext)
        draw_corners(stickerprint_img, oX2, l - oA, oX2 + 6500, l + 480)

        l = l + 480 + oA
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 481, 6500, 640, oX2, l)
        extend(stickerprint_img, l + 1, DIR_UP, ext)
        extend(stickerprint_img, l + 640 - 1, DIR_DOWN, ext)
        draw_corners(stickerprint_img, oX2, l, oX2 + 6500, l + 640)

        l = l + 640 + oA
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 1120, 6500, 480, oX2, l)
        extend(stickerprint_img, l + 1, DIR_UP, ext)
        extend(stickerprint_img, l + 480 - 1, DIR_DOWN, ext)
        draw_corners(stickerprint_img, oX2, l, oX2 + 6500, l + 480 + oA)

        # bsUM,MM,LM:

        l = l + 480 + oA + oA + oA

        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 1600 + oY, 6500, 480, oX2, l)
        extend(stickerprint_img, l + 480 - 1, DIR_DOWN, ext)
        draw_corners(stickerprint_img, oX2, l - oA, oX2 + 6500, l + 480)

        l = l + 480 + oA
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 1600 + oY + 481 - 3, 6500, 640, oX2, l)
        extend(stickerprint_img, l + 1, DIR_UP, ext)
        extend(stickerprint_img, l + 640 - 1, DIR_DOWN, ext)
        draw_corners(stickerprint_img, oX2, l, oX2 + 6500, l + 640)

        l = l + 640 + oA
        transcribe(sliderule_img, stickerprint_img, oX + 750, oY + 1600 + oY + 1120, 6500, 480, oX2, l)
        extend(stickerprint_img, l + 1, DIR_UP, ext)
        extend(stickerprint_img, l + 480 - 1, DIR_DOWN, ext)
        draw_corners(stickerprint_img, oX2, l, oX2 + 6500, l + 480 + oA)

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
            draw_box(stickerprint_img, box_i[0], box_i[1], box_i[2], box_i[3])
            draw_box(stickerprint_img, box_i[0], box_i[1] + 640 + oA, box_i[2], box_i[3])

            box_i[0] = round(2 * (6.5 * oA + 510 + 2 * 750) - box_i[0] - box_i[2])

            draw_box(stickerprint_img, box_i[0], box_i[1], box_i[2], box_i[3])
            draw_box(stickerprint_img, box_i[0], box_i[1] + 640 + oA, box_i[2], box_i[3])

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
                             fill=BACKGROUND_COLOR,
                             outline=CUT_COLOR)

            p_x = round(2 * (6.5 * oA + 510 + 2 * 750) - p_x)

            renderer.ellipse((p_x - r, p_y - r,
                              p_x + r, p_y + r),
                             fill=BACKGROUND_COLOR,
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