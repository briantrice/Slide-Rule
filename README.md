# Slide-Rule

This repository helps generate complete slide rule patterns.

## Running
To run the program, you will need to:
- Download `SlideRule.py`
- Install PIL via:
  - `bash python3 -m pip install Pillow`
- An installation of two fonts: `cmuntt.ttf` and `cmunit.ttf` (part of the infamous LaTeX fonts)
  - These can be found at: https://www.freebestfonts.com/computer-modern-font

The program will prompt which mode you would like to run:
- `render`: Prints a rendering of the full size scales in their correct positions
- `diagnostic`: Prints a rendering containing each available scale arranged in rows
- `stickerprint`: Prints an image which can be scaled to 677.33 ppi in Gimp and then Print+Cut in Silhouette Studio

The produced images are also saved.

## Troubleshooting
- If you have trouble finding them, look in the location where your program is (that seemed to work for me)
- `render` and `diagnostic` modes takes around 15 seconds to run, but the `stickerprint` mode usually takes 1-2 minutes to run, so be patient.
- If you run across issues, comment on Instructables or here, and I will attempt to explain myself.

The Excel file attached can help convert between pixels, inches, and millimeters when the slide rule is being built out of physical material.

Enjoy!
