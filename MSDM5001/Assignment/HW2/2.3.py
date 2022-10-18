from multiprocessing import Pool
import random
import numpy as np
from argparse import ArgumentParser


def f(seed):
    random.seed(seed)
    x = random.random()
    y = random.random()
    if x**2 + y**2 < 1:
        return 1
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=114)
    args = parser.parse_args()
    np.random.seed(args.seed)
    seeds = np.random.random(size=args.n)
    with Pool() as p:
        result = p.map(f, seeds)
    print(4*np.mean(result))
