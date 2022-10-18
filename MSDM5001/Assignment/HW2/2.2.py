from multiprocessing import Pool
from argparse import ArgumentParser


def f(x):
    return 4/(1+x**2)


def x(args):
    a, h, k = args
    return a+h*k


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", type=int, default=1000)
    args = parser.parse_args()
    n = args.n
    a, b = 0, 1
    h = (b-a)/n
    with Pool() as pool:
        xs = pool.map(x, [(a, h, k) for k in range(1, n)])
        fs = pool.map(f,xs)
    pi = h/2*(f(0)+f(1)+2*sum(fs))
    print(pi)
