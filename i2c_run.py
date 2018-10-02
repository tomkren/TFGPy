#!/usr/bin/env python3

import i2c
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, default='')
    parser.add_argument('--img_paths', type=str, nargs='*')
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    i2c.i2c_run(args)


if __name__ == "__main__":
    main()
