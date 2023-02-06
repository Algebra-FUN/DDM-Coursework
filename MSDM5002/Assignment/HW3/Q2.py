import functools
import random
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
from tqdm import tqdm

style.use('seaborn')


random.seed(20912792)


def repeat(trials=1000, summary=np.mean):
    def decor(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(trials):
                results.append(func(*args, **kwargs))
            return summary(results)
        return wrapper
    return decor


def return_in_list(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return list(func(*args, **kwargs))
    return wrapper


DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class Player:
    def __init__(self, init_pos, look_ai, walk_ai):
        self.init_pos = init_pos
        self.look_ai = look_ai
        self.walk_ai = walk_ai

    def config(self, market, other):
        self.market = market
        self.other = other

    def reset(self):
        self.pos = self.init_pos
        self.previous = None

    def look(self):
        return self.look_ai(self)

    def walk(self):
        self.next = self.walk_ai(self)

    def update(self):
        self.previous = self.pos
        self.pos = self.next

    @return_in_list
    def available_pos(self):
        x, y = self.pos
        n = self.market.n
        for dx, dy in DIRECTIONS:
            x_, y_ = x+dx, y+dy
            # backward is always disabled
            if 0 <= x_ <= n and 0 <= y_ <= n and (x_, y_) != self.previous:
                yield (x_, y_)


class SuperMarket:
    def __init__(self, n, p0: Player, p1: Player):
        self.n = n
        self.p0 = p0
        self.p1 = p1
        self.p0.config(self, p1)
        self.p1.config(self, p0)

    def reset(self):
        self.p0.reset()
        self.p1.reset()

    @repeat(trials=1000)
    def go(self):
        self.reset()
        t = 0
        while self.p0.pos != self.p1.pos:
            self.p0.walk()
            self.p1.walk()
            self.p0.update()
            self.p1.update()
            t += 5
        return t


def stand(self: Player):
    return self.pos


def blind(self: Player):
    return False


def line_search(self: Player):
    if self.pos[0] == self.other.pos[0] or self.pos[1] == self.other.pos[1]:
        return True
    return False


def rand_find(self: Player):
    if self.look():
        x, y = self.pos
        Dx = self.other.pos[0] - x
        Dy = self.other.pos[1] - y
        dx = Dx//abs(Dx) if Dx else 0
        dy = Dy//abs(Dy) if Dy else 0
        return (x+dx, y+dy)
    return random.choice(self.available_pos())


# a)
ns = [*range(2, 21)]

plt.cla()
ts = []
for n in tqdm(ns, desc="2-a-only Alice walks"):
    Alice = Player((0, n), line_search, rand_find)
    Bob = Player((n, 0), blind, stand)

    supermarket = SuperMarket(n, Alice, Bob)
    ts.append(supermarket.go())
plt.plot(ns, ts, marker='s', label="only Alice walks")

ts = []
for n in tqdm(ns, desc="2-a-both Alice and Bob walk"):
    Alice = Player((0, n), line_search, rand_find)
    Bob = Player((n, 0), line_search, rand_find)

    supermarket = SuperMarket(n, Alice, Bob)
    ts.append(supermarket.go())
plt.plot(ns, ts, marker='o', label="both Alice and Bob walk")


plt.legend()
plt.title("a) Average finding time over 1000 trials")
plt.grid(True)
plt.xlabel('n')
plt.ylabel('finding time')
plt.savefig('out/q2-a.png')
plt.show()


# b)
n = 19
ks = [*range((n+1)//2)]

plt.cla()
ts = []
for k in tqdm(ks, desc="2-b-only Alice walks"):
    a = (n-1)//2-k
    b = (n+1)//2+k
    Alice = Player((a, b), line_search, rand_find)
    Bob = Player((b, a), blind, stand)

    supermarket = SuperMarket(n, Alice, Bob)
    ts.append(supermarket.go())
plt.plot(ks, ts, marker='s', label="only Alice walks")

ts = []
for k in tqdm(ks, desc="2-b-both Alice and Bob walk"):
    a = (n-1)//2-k
    b = (n+1)//2+k
    Alice = Player((a, b), line_search, rand_find)
    Bob = Player((b, a), line_search, rand_find)

    supermarket = SuperMarket(n, Alice, Bob)
    ts.append(supermarket.go())
plt.plot(ks, ts, marker='o', label="both Alice and Bob walk")

plt.legend()
plt.title("b) Average finding time over 1000 trials")
plt.grid(True)
plt.xlabel('k')
plt.ylabel('finding time')
plt.savefig('out/q2-b.png')
plt.show()

# c)


def poor_forward_search(self: Player):
    for pos in self.available_pos():
        if self.other.pos == pos:
            return True
    return False


# c) -a
ns = [*range(2, 21)]

plt.cla()
ts = []
for n in tqdm(ns, desc="2-c-a-only Alice walks"):
    Alice = Player((0, n), poor_forward_search, rand_find)
    Bob = Player((n, 0), blind, stand)

    supermarket = SuperMarket(n, Alice, Bob)
    ts.append(supermarket.go())
plt.plot(ns, ts, marker='s', label="only Alice walks")

ts = []
for n in tqdm(ns, desc="2-c-a-both Alice and Bob walk"):
    Alice = Player((0, n), poor_forward_search, rand_find)
    Bob = Player((n, 0), poor_forward_search, rand_find)

    supermarket = SuperMarket(n, Alice, Bob)
    ts.append(supermarket.go())
plt.plot(ns, ts, marker='o', label="both Alice and Bob walk")


plt.legend()
plt.title("c)-a: Average finding time over 1000 trials")
plt.grid(True)
plt.xlabel('n')
plt.ylabel('finding time')
plt.savefig('out/q2-c-a.png')
plt.show()

# c) -b
n = 19
ks = [*range((n+1)//2)]

plt.cla()
ts = []
for k in tqdm(ks, desc="2-c-b-only Alice walks"):
    a = (n-1)//2-k
    b = (n+1)//2+k
    Alice = Player((a, b), poor_forward_search, rand_find)
    Bob = Player((b, a), blind, stand)

    supermarket = SuperMarket(n, Alice, Bob)
    ts.append(supermarket.go())
plt.plot(ks, ts, marker='s', label="only Alice walks")

ts = []
for k in tqdm(ks, desc="2-c-b-both Alice and Bob walk"):
    a = (n-1)//2-k
    b = (n+1)//2+k
    Alice = Player((a, b), poor_forward_search, rand_find)
    Bob = Player((b, a), poor_forward_search, rand_find)

    supermarket = SuperMarket(n, Alice, Bob)
    ts.append(supermarket.go())
plt.plot(ks, ts, marker='o', label="both Alice and Bob walk")

plt.legend()
plt.title("c)-b: Average finding time over 1000 trials")
plt.grid(True)
plt.xlabel('k')
plt.ylabel('finding time')
plt.savefig('out/q2-c-b.png')
plt.show()