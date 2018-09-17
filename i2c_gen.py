#!/usr/bin/env python3

import i2c
import argparse

GO_small = 'small'
GO_full = 'full'
GO_requested_128 = 'requested_128'
GO_003similar = '003similar'

GO_ALL = [GO_small, GO_full, GO_requested_128, GO_003similar]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_opts', type=str, choices=GO_ALL, default=GO_small)
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    i2c.i2c_gen(args.gen_opts)


if __name__ == "__main__":
    main()
