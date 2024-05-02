#!/usr/bin/env python3
import argparse
import os.path

from SlideRule import (
    Models, Model, keys_of,
    render_diagnostic_mode, render_sliderule_mode, render_stickerprint_mode, Mode
)


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--mode',
                             choices=[m.value for m in Mode],
                             default=None,
                             help='Which mode to render (all by default)')
    args_parser.add_argument('--model',
                             choices=keys_of(Models),
                             default=None,
                             help='Which sliderule model (all by default)')
    cli_args = args_parser.parse_args()
    base_dir = os.path.relpath('examples/')
    os.makedirs(base_dir, exist_ok=True)
    modes = [next(m for m in Mode if m.value == cli_args.mode)] if cli_args.mode else Mode
    for model_name in ([cli_args.model] if cli_args.model else keys_of(Models)):
        print(f'Building example outputs for: {model_name}')
        model = getattr(Models, model_name) or Model.from_toml_file(os.path.join(base_dir, f'Model-{model_name}.toml'))

        if Mode.DIAGNOSTIC in modes:
            diagnostic_img = render_diagnostic_mode(model)
            diagnostic_filename = os.path.join(base_dir, f'{model_name}.Diagnostic.png')
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


if __name__ == '__main__':
    main()
