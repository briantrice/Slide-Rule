# Slide-Rule

This Python 3 application helps generate complete slide rule patterns.

This is via:
- [Javier Lopez: Slide Rule for the Modern Day](https://jlopezengineer.com/home/sliderule)
- [Instructables: A Vintage Calculator for the Modern Day](https://www.instructables.com/Slide-Rule-for-the-Modern-Day/)

[![Python application](https://github.com/briantrice/Slide-Rule/actions/workflows/python-app.yml/badge.svg)](https://github.com/briantrice/Slide-Rule/actions/workflows/python-app.yml)

## Running
To run the program, you can download the one file as:
- Download `SlideRule.py`
- Install PIL via:
```shell
python3 -m pip install Pillow
```
- Install TOML via:
```shell
python3 -m pip install toml
```

Or clone this git repository locally and install per the requirements file as:
```shell
python3 -m pip install -r requirements.txt
```

An installation of two fonts: `cmuntt.ttf` and `cmunit.ttf` (part of the infamous LaTeX fonts)
  - These can be found at: https://www.freebestfonts.com/computer-modern-font
  - Install all of them if you want different typesetting options.

```wp-cli
usage: SlideRule.py [-h] [--mode {render,diagnostic,stickerprint}]
                    [--model {Demo,MannheimOriginal,Ruler,MannheimWithRuler,Aristo868,Aristo965,PickettN515T,FaberCastell283,FaberCastell283N,Graphoplex621,Hemmi153,UltraLog}]
                    [--suffix SUFFIX] [--test] [--cutoffs] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --mode {render,diagnostic,stickerprint}
                        What to render
  --model {Demo,MannheimOriginal,Ruler,MannheimWithRuler,Aristo868,Aristo965,PickettN515T,FaberCastell283,FaberCastell283N,Graphoplex621,Hemmi153,UltraLog}
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

There are multiple slide rule models available, mostly defined in [TOML](https://toml.io):
- [Demo](examples/Model-Demo.toml) is the original slide rule made for instruction.
- [MannheimOriginal](examples/Model-MannheimOriginal.toml) is the simplest model, with just 4 scales along the edges.
- `Ruler` and `MannheimWithRuler` show the ruler scales, which can be set along the edge of a slide rule.
- [PickettN515T](examples/Model-PickettN515T.toml) is made for electronics engineering calculations.
- [Aristo868](examples/Model-Aristo868.toml) and [Graphoplex621](examples/Model-Graphoplex621.toml) are European slide rules with a moderate number of compact scales.
- [FaberCastell283](examples/Model-FaberCastell283.toml) and [FaberCastell283N](examples/Model-FaberCastell283N.toml) have the large number of scales.

The produced images are also saved. The images in [examples](examples) are refreshed by [generate_example_outputs.py](generate_example_outputs.py), a wrapper script you can run on new models, or to check for changes after making source code updates.

## Troubleshooting
- If you have trouble finding them, look in the location where your program is (that seemed to work for me)
- `render` and `diagnostic` modes takes around 2 seconds to run, but the `stickerprint` mode usually takes 3 seconds to run.
- If you run across issues, comment on [Javier Lopez Instructables article](https://www.instructables.com/Slide-Rule-for-the-Modern-Day/) or here, and I will attempt to explain myself.

To understand changes, the [imgdiff](https://github.com/n7olkachev/imgdiff) tool can highlight differences in image outputs (ignoring color changes with `-t 0.6`):
```shell
imgdiff -t 0.6 <original_image>.png <updated_image>.png <original_image>.diff.png
```

The [included Excel file](Slide%20Rule%20Proportion%20Calculator%20(Autosaved).xlsx) can help convert between pixels, inches, and millimeters when the slide rule is being built out of physical material.

Enjoy!
