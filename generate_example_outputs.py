#!/usr/bin/env python3
import argparse
import os.path
import time

from SlideRule import (
    Model, Mode, render_diagnostic_mode, render_sliderule_mode, render_stickerprint_mode
)


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--mode',
                             choices=[m.value for m in Mode],
                             default=None,
                             help='Which mode to render (all by default)')
    example_models = list(Model.example_names())
    args_parser.add_argument('--model',
                             choices=example_models,
                             default=None,
                             help='Which sliderule model (all by default)')
    args_parser.add_argument('--all-scales',
                             action='store_true',
                             help='Whether to show every possible scale in diagnostic mode')
    cli_args = args_parser.parse_args()
    base_dir = os.path.relpath('examples/')
    os.makedirs(base_dir, exist_ok=True)
    modes = [next(m for m in Mode if m.value == cli_args.mode)] if cli_args.mode else Mode
    for model_name in ([cli_args.model] if cli_args.model else example_models):
        print(f'Building example outputs for: {model_name}')
        model = Model.load(model_name)

        start_time = time.process_time()
        if Mode.DIAGNOSTIC in modes:
            all_scales = cli_args.all_scales
            diagnostic_img = render_diagnostic_mode(model, all_scales=all_scales)
            suffix = 'allScales.png' if all_scales else 'png'
            diagnostic_filename = os.path.join(base_dir, f'{model_name}.Diagnostic.{suffix}')
            diagnostic_img.save(diagnostic_filename)
            print(f' Diagnostic output for: {model_name} at: {diagnostic_filename}')
        if Mode.RENDER in modes:
            sliderule_img = render_sliderule_mode(model, borders=True, cutoffs=True)
            sliderule_filename = os.path.join(base_dir, f'{model_name}.SlideRuleScales.png')
            sliderule_img.save(sliderule_filename, 'PNG')
            print(f' SlideRuleScales output for: {model_name} at: {sliderule_filename}')
        if Mode.STICKERPRINT in modes:
            sliderule_img = render_sliderule_mode(model, cutoffs=True)
            sliderule_stickers_img = render_sliderule_mode(model, sliderule_img)
            stickers_img = render_stickerprint_mode(model, sliderule_stickers_img)
            stickers_filename = os.path.join(base_dir, f'{model_name}.StickerCut.png')
            stickers_img.save(stickers_filename)
            print(f' StickerCut output for: {model_name} at: {stickers_filename}')
        print(f'Time elapsed: {round(time.process_time() - start_time, 3)}')


if __name__ == '__main__':
    main()
