import argparse

from robosat.tools import (
    compare,
    cover,
    dedupe,
    download,
    extract,
    export,
    features,
    masks,
    merge,
    predict,
    rasterize,
    serve,
    serve_dir,
    subset,
    train,
    weights,
    metrics,
)


def add_parsers():
    parser = argparse.ArgumentParser(prog="./rs")
    subparser = parser.add_subparsers(title="robosat tools", metavar="")

    # Add your tool's entry point below.

    extract.add_parser(subparser)
    cover.add_parser(subparser)
    download.add_parser(subparser)
    rasterize.add_parser(subparser)

    train.add_parser(subparser)
    export.add_parser(subparser)
    predict.add_parser(subparser)
    masks.add_parser(subparser)
    features.add_parser(subparser)
    merge.add_parser(subparser)
    dedupe.add_parser(subparser)

    serve.add_parser(subparser)
    serve_dir.add_parser(subparser)

    weights.add_parser(subparser)

    compare.add_parser(subparser)
    subset.add_parser(subparser)
    metrics.add_parser(subparser)

    # We return the parsed arguments, but the sub-command parsers
    # are responsible for adding a function hook to their command.

    subparser.required = True

    return parser.parse_args()


def main():
    """main entrypoint for robosat tools"""
    args = add_parsers()
    args.func(args)


if __name__ == "__main__":
    main()
