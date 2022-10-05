# %%0
# import all needed module at this cell
from matplotlib import pyplot as plt
from matplotlib import style
import shutil
import re
import os
import time
import random
import numpy as np
import functools
from tqdm import tqdm

style.use('seaborn')

# %%1
# Russian roulette

np.random.seed(20912792)


def A_win_1_bullet(num_game):
    num_pos = 10
    B_round = [1, 2, 3, 4, 5]
    B_lose = 0
    for i in range(num_game):
        gun = np.zeros(num_pos, bool)
        gun[np.random.randint(num_pos)] = True
        if any(gun[B_round]):
            B_lose += 1
    return B_lose/num_game


def A_win_k_bullet_fast(num_game, k=1):
    num_pos = 10
    B_round = [1, 2, 3, 4, 5]
    B_killed = 0
    alive_games = num_game
    for i in range(max(B_round)+1):
        if not alive_games:
            break
        p = k/(num_pos - i)
        # killed_games = np.random.binomial(1,p,size=alive_games).sum() if p < 1 else alive_games
        killed_games = np.random.binomial(
            alive_games, p) if p < 1 else alive_games
        if i in B_round:
            B_killed += killed_games
        alive_games -= killed_games
    return B_killed/num_game


def A_win_1_bullet_fast(num_game):
    return A_win_k_bullet_fast(num_game, k=1)

# def A_win_1_bullet_fast(num_game):
#     num_pos = 10
#     gun_pos = np.random.randint(0,num_pos,size=num_game)
#     B_lose = (1<=gun_pos)*(gun_pos<=5)
#     return B_lose.mean()


def A_win_3_bullet(num_game):
    num_pos = 10
    B_round = [1, 2, 3, 4, 5]
    B_lose = 0

    for i in range(num_game):
        gun = np.zeros(num_pos)
        b1 = np.random.randint(num_pos)
        gun[b1] = 1

        b2 = b1
        while b2 == b1:
            b2 = np.random.randint(num_pos)
        gun[b2] = 1

        b3 = b1
        while b3 == b1 or b3 == b2:
            b3 = np.random.randint(num_pos)
        gun[b3] = 1

        for n in range(num_pos):
            if gun[n] == 1:
                if n in B_round:
                    B_lose += 1
                break
    return B_lose/num_game


def A_win_3_bullet_fast(num_game):
    return A_win_k_bullet_fast(num_game, k=3)


# 1_bullet, 1_bullet_fast, 3_bullet, 3_bullet_fast
T1, T2, T3, T4 = 0, 0, 0, 0     # cumulative running time
P1, P2, P3, P4 = [], [], [], []  # records of winning probability

num_test = 10
print('test', end=' ')
for n in range(num_test):
    print(n, end=' ')
    num_game = 10**5

    start_time = time.time()
    P1.append(A_win_1_bullet(num_game))
    T1 += (time.time()-start_time)

    start_time = time.time()
    P2.append(A_win_1_bullet_fast(num_game))
    T2 += (time.time()-start_time)

    start_time = time.time()
    P3.append(A_win_3_bullet(num_game))
    T3 += (time.time()-start_time)

    start_time = time.time()
    P4.append(A_win_3_bullet_fast(num_game))
    T4 += (time.time()-start_time)

print()
print('T2/T1 =', T2/T1)
print('T4/T3 =', T4/T3)
print(min(P1), '<= P1 <=', max(P1))
print(min(P2), '<= P2 <=', max(P2))
print(min(P3), '<= P3 <=', max(P3))
print(min(P4), '<= P4 <=', max(P4))

# %% 2
# Supermarket

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

    def not_meet(self):
        return self.p0.pos != self.p1.pos

    @repeat(trials=1000)
    def go(self):
        self.reset()
        t = 0
        while self.not_meet():
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
plt.savefig('./img/2-a.png')
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
plt.savefig('./img/2-b.png')
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
plt.savefig('./img/2-c-a.png')
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
plt.savefig('./img/2-c-b.png')
plt.show()


# %%3
# File organization
WORK_DIR = f"{os.getcwd()}\\materials\\question3"

MONTH = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
         "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
MONTH_MAP = {month: str(i).zfill(2) for i, month in enumerate(MONTH, 1)}


def ckdir(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


def organize(work_dir=WORK_DIR):
    for file in os.listdir(work_dir):
        day, month, year = re.findall(
            "(\d{2})-([A-Z]{3})-(\d{4})\.txt", file)[0]
        target_dir = f"{work_dir}\\{year}"
        new_file = f"{MONTH_MAP[month]}{day}.txt"
        ckdir(target_dir)
        shutil.move(f"{work_dir}\\{file}", f"{target_dir}\\{new_file}")


# organize(WORK_DIR)

# %%4
# Frequency analysis


def f_analysis(string, f_key=True):
    f_dict = {}
    for char in string.lower():
        if char.isalpha():
            f_dict[char] = f_dict.get(char, 0) + 1
    f_stats = {}
    for char, f in f_dict.items():
        if f not in f_stats:
            f_stats[f] = []
        f_stats[f].append(char)

    print(string)
    print("=== Frequency table ===")
    for f, chars in sorted(f_stats.items(), key=lambda item: item[0], reverse=True):
        print(f"{f}\t{', '.join(sorted(chars))}")

    return f_stats if f_key else f_dict

# %% 5
# Bracket checking


LEFTS = '{[('
RIGHTS = '}])'
MAPPING = {left: right for left, right in zip(LEFTS, RIGHTS)}


def bracket_check(string):
    stack = []
    num_pairs = 0
    for char in string:
        if char in LEFTS:
            stack.append(char)
        elif char in RIGHTS:
            if char == MAPPING[stack[-1]]:
                stack.pop()
                num_pairs += 1
            else:
                return -1
    return num_pairs


# print(bracket_check(("{m(s[d]m(5}0[c)02]")))
# print(bracket_check("p(y[th[on]{c(our)s}e])"))


# %% 6
# Band matrix

def checkband(A, n, sign=1):
    for k in range(n-1, -1, -1):
        if np.count_nonzero(np.diag(A, sign*k)):
            return k


def bandwidth(A):
    assert A.ndim == 2
    m, n = A.shape
    bL = checkband(A, m, -1)
    bU = checkband(A, n, 1)
    return (bU, bL)


X = np.array([[1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
Y = np.array([[1, 0], [1, 1], [1, 1]])

print('X', bandwidth(X))
print('Y', bandwidth(Y))


# %% 7
# Coupon collector

np.random.seed(20912792)


class Box:
    def __init__(self, N):
        self.coupon = np.random.randint(N)

    def open(self):
        return self.coupon


def repeat_within(time_tol=0.1, summary=np.mean):
    def decor(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            used_time = 0
            results = []
            while used_time < time_tol:
                start_time = time.time()
                result = func(*args, **kwargs)
                used_time += time.time()-start_time
                results.append(result)
            return summary(results)
        return wrapper
    return decor


@repeat_within(time_tol=0.1, summary=np.mean)
def box_number(N, k):
    coupon_collection = set()
    coupon = 0
    box = 0
    while coupon < k:
        item = Box(N).open()
        box += 1
        if item not in coupon_collection:
            coupon_collection.add(item)
            coupon += 1
    return box

# theoretical value of expected time to collect k type from N
# decompose RV T(N,k) into a series of RV of gemotric distribution to calculate its expectation.


def ET(N, k):
    return N*sum(1/(N-t) for t in range(k))


# a)
plt.cla()
N = 10
ks = range(1, N+1)
ys = [ET(N, k) for k in ks]
plt.plot(ks, ys, label='theoretical')
ys = [box_number(N, k) for k in ks]
plt.plot(ks, ys, marker='s', linestyle=':', color='r', label='simulation')
plt.legend()
plt.title("number of boxes to get k types of coupons(N=10)")
plt.grid(True)
plt.xlabel('k')
plt.ylabel('expected number of boxes')
plt.savefig('./img/7-a.png')
plt.show()

# b)
plt.cla()
Nmax = 10
Ns = range(1, Nmax+1)
ys = [ET(N, N) for N in Ns]
plt.plot(ks, ys, label='theoretical')
ys = [box_number(N, N) for N in Ns]
plt.plot(Ns, ys, marker='s', linestyle=':', color='r', label='simulation')
plt.legend()
plt.title("number of boxes to get k types of coupons(k=N)")
plt.grid(True)
plt.xlabel('N')
plt.ylabel('expected number of boxes')
plt.savefig('./img/7-b.png')
plt.show()

# %% 8
# Tic-tac-toe

random.seed(20912792)


class Player:
    def __init__(self, ai):
        self.ai = ai

    def config(self, mark=0, rival=None):
        self.mark = mark
        self.rival = rival

    def reset(self):
        self.row_score = [0, 0, 0]
        self.col_score = [0, 0, 0]
        self.diag_score = 0
        self.sub_diag_score = 0

    def select(self, vacancies):
        return self.ai(vacancies, self, self.rival)

    def score(self, pos):
        i, j = pos
        self.row_score[i] += 1
        self.col_score[j] += 1
        if i == j:
            self.diag_score += 1
        elif i+j == 2:
            self.sub_diag_score += 1

    def is2win(self, pos):
        i, j = pos
        if self.row_score[i] == 2 or self.col_score[j] == 2:
            return True
        if i == j and self.diag_score == 2:
            return True
        if i+j == 2 and self.sub_diag_score == 2:
            return True
        return False


def Tictactoe_summary(results):
    turns = len(results)
    results = np.array(results)
    # [p0 win, p1 win, draw, turns]
    return [*(np.sum(results == x)/turns for x in (0, 1, -1)), turns]


def repeat_within(time_tol, summary):
    def decor(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            used_time = 0
            results = []
            while used_time < time_tol:
                start_time = time.time()
                result = func(*args, **kwargs)
                used_time += time.time()-start_time
                results.append(result)
            return summary(results)
        return wrapper
    return decor


class Tictactoe:
    def __init__(self, p0: Player, p1: Player):
        self.players = (p0, p1)
        p0.config(mark=0, rival=p1)
        p1.config(mark=1, rival=p0)

    def reset(self):
        for player in self.players:
            player.reset()
        self.vacancies = [(i, j) for j in range(3) for i in range(3)]

    @repeat_within(time_tol=1, summary=Tictactoe_summary)
    def __call__(self):
        self.reset()
        who = random.randint(0, 1)
        while self.vacancies:
            player = self.players[who % 2]
            pos, winner = player.select(self.vacancies)
            if winner is not None:
                return winner
            if player.is2win(pos):
                return player.mark
            player.score(pos)
            self.vacancies.remove(pos)
            who += 1
        return -1


def rand_choice(vacancies, *args):
    return random.choice(vacancies), None


Alice = Player(ai=rand_choice)
Bob = Player(ai=rand_choice)

tictactoe = Tictactoe(Alice, Bob)
# >>> tictactoe()
# [0.3979030224764013, 0.39796451741844235, 0.20413246010515634, 65046]


def bob_ai(vacancies, *args):
    # 1. pick centre
    if (1, 1) in vacancies:
        return (1, 1), None
    # 2. pick randomly
    return rand_choice(vacancies)


def alice_ai(vacancies, self: Player, rival: Player):
    # 1. alice try to win
    for pos in vacancies:
        if self.is2win(pos):
            return pos, self.mark

    # 2. try to let bob win in next turn
    posxbob = None
    # (n-1) vacancy for bob to win
    # only 1 vacancy for bob not win
    for pos in vacancies:
        if rival.is2win(pos):
            continue

        # this position not for bob to win
        if posxbob:
            # at least 2 vacancies for bob not win
            break
        posxbob = pos
    else:
        if posxbob:
            return posxbob, rival.mark

    # 3. pick randomly
    return rand_choice(vacancies)


Alice = Player(ai=alice_ai)
Bob = Player(ai=bob_ai)

tictactoe_AI = Tictactoe(Alice, Bob)
# >>> tictactoe_AI()
# [0.54723829451075, 0.39095159809552277, 0.06181010739372718, 46837]


# %% 9
# Magic command


class TimeitCustom:
    def __init__(self, repeat=7, loops=10000000):
        self.repeat = repeat
        self.loops = loops
        self.average = 0
        self.stdev = 0

    def __call__(self, target, *args, **kwargs):
        running_time = []
        for _ in range(self.repeat):
            start_time = time.time()
            for _ in range(self.loops):
                target(*args, **kwargs)
            running_time.append(1e9*(time.time() - start_time)/self.loops)

        self.average = np.mean(running_time)
        self.stdev = np.std(running_time)
        print(f"{self.average:.2f} ns ± {self.stdev:.4f} ns per loop (mean ± std. dev. of {self.repeat} runs, {self.loops} loops each)")
        return self


timeit_custom = TimeitCustom(repeat=7, loops=10000000)
timeit_custom(lambda x: x*x, 1)

# %% 10
# Quadratic equation


def quadratic(a,  b,  c):
    delta = b**2-4*a*c
    if delta >= 0:
        return (-b+delta**.5)/2/a, (-b-delta**.5)/2/a
    return (-b+(-delta)**.5*1j)/2/a, (-b-(-delta)**.5*1j)/2/a


def quadratic_(a,  b,  c):
    delta = b**2-4*a*c
    return (-b+delta**.5)/2/a, (-b-delta**.5)/2/a

# it's quite confuzed that:
# >>> quadratic(1,-2,2)
# ((1+1j), (1-1j))
# >>> quadratic_(1,-2,2)
# ((1+1j), (0.9999999999999999-1j))
# wondering why?
