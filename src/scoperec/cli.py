import argparse
import importlib
import sys

COMMANDS = {
    "recommend": "scoperec.official_recommend",
    "recommend-temporal": "scoperec.recommend_v2",
    "results": "scoperec.release_results",
}


def main():
    parser = argparse.ArgumentParser(
        prog="scoperec",
        description="Frozen-backbone cold-start recommendation and final results.",
        epilog="Use scoperec <command> --help for command-specific options.",
    )
    parser.add_argument("command", choices=tuple(COMMANDS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    previous = sys.argv
    try:
        sys.argv = [f"scoperec {arguments.command}", *arguments.arguments]
        importlib.import_module(COMMANDS[arguments.command]).main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
