#! /usr/bin/env python

import os
import sys
import subprocess
import click


@click.command()
@click.argument('setting_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--options', '-o', type=str, multiple=True)
@click.argument('target', type=str)
def main(setting_dir, target, options):
    """bb_make builds the TARGET with settings from SETTING_DIR."""
    make_file = os.path.join(os.path.dirname(__file__), "Makefile")
    setting_dir = os.path.abspath(setting_dir)
    cmd = ["make", "-f", make_file, "setting_dir={}".format(setting_dir)] \
        + list(options) + [target]
    print(" ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
