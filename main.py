import argparse
from game import Game
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NEAT xor experiment evaluated across multiple machines.")
    parser.add_argument(
        "level",
        help="level in the levels folder",
        type=str,
        action="store"
        )
    parser.add_argument(
        "--population",
        type=int,
        help="population size",
        action="store",
        default=50,
        dest="population"
    )
    parser.add_argument(
        "--generations",
        type=int,
        help="number of generation after stopping",
        action="store",
        default=1000,
        dest="gen"
    )
    parser.add_argument(
        "--filename",
        "-f",
        type=str,
        help="filename of the file generated after max generations",
        action="store",
        default=str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + ".neat",
        dest="filename"
    )
    parser.add_argument(
        "--make",
        help="launches the level editor",
        action="store_true",
        dest="make"
    )
    parser.add_argument(
        "--no-save",
        help="do not save the final genome",
        action="store_false",
        dest="save"
    )
    parser.add_argument(
        "--test",
        type=str,
        help="test best genome of .neat file",
        action="store",
        default=None,
        dest="test"
    )
    parser.add_argument(
        "--train",
        help="",
        action="store_true",
        dest="train"
    )
    parser.add_argument(
        "-i",
        "--interactive",
        help="makes testing interactive by setting checkpoints",
        action="store_true",
        dest="interactive"
    )

    ns = parser.parse_args()
    if ns.test is None:
        test = False
    else:
        test = True
    game = Game(ns.level, population=ns.population, max_gen=ns.gen,
                filename=ns.filename, save=ns.save, test=test, neatfile=ns.test,
                train=ns.train, make=ns.make, interactive=ns.interactive)

