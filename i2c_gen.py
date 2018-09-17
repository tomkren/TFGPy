#!/usr/bin/env pypy3
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


if __name__ == "__main__":
    main()
