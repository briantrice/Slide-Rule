#!/usr/bin/env python3
import argparse
import os.path
import toml

from SlideRule import (
    Models, Model, keys_of,
    render_diagnostic_mode, render_sliderule_mode, render_stickerprint_mode
)


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--model',
                             choices=keys_of(Models),
                             default=None,
                             help='Which sliderule model (all by default)')
    cli_args = args_parser.parse_args()
    base_dir = os.path.relpath('examples/')
    os.makedirs(base_dir, exist_ok=True)
    for model_name in ([cli_args.model] if cli_args.model else keys_of(Models)):
        print(f'Building example outputs for: {model_name}')
        model = getattr(Models, model_name) or Model.from_toml_file(os.path.join(base_dir, f'Model-{model_name}.toml'))

        print(f'Building Diagnostic output for: {model_name}')
        diagnostic_img = render_diagnostic_mode(model)
        diagnostic_img.save(os.path.join(base_dir, f'{model_name}.Diagnostic.png'))

        print(f'Building SlideRuleScales output for: {model_name}')
        sliderule_img = render_sliderule_mode(model, borders=True, cutoffs=True)
        sliderule_img.save(os.path.join(base_dir, f'{model_name}.SlideRuleScales.png'), 'PNG')

        print(f'Building StickerCut output for: {model_name}')
        sliderule_stickers_img = render_sliderule_mode(model)
        stickers_img = render_stickerprint_mode(model, sliderule_stickers_img)
        stickers_img.save(os.path.join(base_dir, f'{model_name}.StickerCut.png'))


if __name__ == '__main__':
    main()
