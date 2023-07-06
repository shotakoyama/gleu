from argparse import ArgumentParser
from .sent_main import sent_main, mean_main
from .corpus_main import corpus_main
from .count import set_tokenization


def gleu():
    main(corpus_main)


def sgleu():
    main(sent_main)


def mgleu():
    main(mean_main)


def main(handler):
    parser = ArgumentParser()
    add_args(parser)
    parser.set_defaults(handler = handler)
    args = parser.parse_args()
    set_tokenization(args.token)
    args.handler(args)


def add_args(parser):
    parser.add_argument(
            '-n', type = int, default = 4,
            help = 'maximum n for n-gram')
    parser.add_argument(
            '-s', '--src', '--source',
            required = True,
            dest = 'source_path')
    parser.add_argument(
            '-r', '--ref', '--references',
            required = True,
            nargs = '+',
            dest = 'ref_path_list')
    parser.add_argument(
            '-o', '--out', '--outputs',
            required = True,
            nargs = '+',
            dest = 'hyp_path_list')
    parser.add_argument(
            '-i', '--iter',
            type = int, default = 500,
            help = 'the number of iterations to run')
    parser.add_argument(
            '-d', '--digit', dest = 'digit',
            type = int, default = 2,
            help = 'digit of quantization')
    parser.add_argument(
            '-p', '--proc', dest = 'proc',
            type = int, default = 1)
    parser.add_argument(
            '-t', '--token', dest = 'token',
            choices = ['char', 'word'],
            default = 'word')
    parser.add_argument(
            '-m', '--max', dest = 'max',
            action = 'store_true')
    parser.add_argument(
            '-f', '--fix', '--fix-seed', dest = 'fix_seed',
            action = 'store_true')
    parser.add_argument(
            '-v', '--verbose', dest = 'verbose',
            action = 'store_true')

