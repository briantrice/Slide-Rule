# Slide-Rule

This Python 3 application helps generate complete slide rule patterns.

[![Python application](https://github.com/briantrice/Slide-Rule/actions/workflows/python-app.yml/badge.svg)](https://github.com/briantrice/Slide-Rule/actions/workflows/python-app.yml)

## Running
To run the program, you can download the one file as:
- Download `SlideRule.py`
- Install PIL via:
```shell
python3 -m pip install Pillow
```

Or clone this git repository locally and install per the requirements file as:
```shell
python3 -m pip install -r requirements.txt
```

An installation of two fonts: `cmuntt.ttf` and `cmunit.ttf` (part of the infamous LaTeX fonts)
  - These can be found at: https://www.freebestfonts.com/computer-modern-font

```wp-cli
usage: SlideRule.py [-h] [--mode {render,diagnostic,stickerprint}] [--model {Aristo868,Demo,EV,FaberCastell283,FaberCastell283N,Graphoplex621,MannheimOriginal,PickettN515T}] [--suffix SUFFIX] [--test]
                    [--cutoffs] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --mode {render,diagnostic,stickerprint}
                        What to render
  --model {Aristo868,Demo,EV,FaberCastell283,FaberCastell283N,Graphoplex621,MannheimOriginal,PickettN515T}
                        Which sliderule model
  --suffix SUFFIX       Output filename suffix for variations
  --test                Output filename for test comparisons
  --cutoffs             Render the metal cutoffs
  --debug               Render debug indications (corners and bounding boxes)
```

The program has 3 rendering modes for any of the slide rule models defined:
- `render`: Prints a rendering of the full size scales in their correct positions
- `diagnostic`: Prints a rendering containing each available scale arranged in rows
- `stickerprint`: Prints an image which can be scaled to 677.33 ppi in Gimp and then Print+Cut in Silhouette Studio

There are multiple slide rule models available:
- `Demo` is the original slide rule made for instruction.
- `MannheimOriginal` is the simplest slide rule, with just 4 scales along the slide edges.
- `PickettN515T` is a slide rule made for electronics engineering calculations.
- `Aristo868` and `Graphoplex621` are European slide rules with a moderate number of compact scales.
- `FaberCastell283` and `FaberCastell283N` have the large number of scales.

The produced images are also saved.

## Troubleshooting
- If you have trouble finding them, look in the location where your program is (that seemed to work for me)
- `render` and `diagnostic` modes takes around 2 seconds to run, but the `stickerprint` mode usually takes 3 seconds to run.
- If you run across issues, comment on Instructables or here, and I will attempt to explain myself.

The Excel file attached can help convert between pixels, inches, and millimeters when the slide rule is being built out of physical material.

Enjoy!
