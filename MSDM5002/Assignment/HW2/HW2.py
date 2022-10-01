# %%0
# import all needed module at this cell
from matplotlib import pyplot as plt
import shutil
import re
import os
import time
import numpy as np
from functools import wraps

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
    for i in range(num_pos):
        p = k/(num_pos - i)
        # killed_games = (np.random.random(size=alive_games)<p).sum()
        # killed_games = np.random.binomial(1,p,size=alive_games) if p < 1 else alive_games
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
#

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


organize(WORK_DIR)

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


print(bracket_check(("{m(s[d]m(5}0[c)02]")))
print(bracket_check("p(y[th[on]{c(our)s}e])"))


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
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            results = []
            while (time.time() - start_time) <= time_tol:
                results.append(func(*args, **kwargs))
            return summary(results)
        return wrapper
    return decor


@repeat_within(time_tol=0.1, summary=np.mean)
def box_number(N, k):
    coupon_collection = {}
    coupon = 0
    box = 0
    while coupon < k:
        item = Box(N).open()
        box += 1
        if not item in coupon_collection:
            coupon_collection[item] = 1
            coupon += 1
    return box


# -------------------- #
N = 10
ks = range(1, N+1)
ys = [*map(lambda k: box_number(N, k), ks)]
plt.plot(ks, ys, marker='s')
plt.legend()
plt.title("number of boxes to get k types of coupons(N=10)")
plt.grid(linestyle=':')
plt.xlabel('k')
plt.ylabel('expected number of boxes')
plt.savefig('./img/7-a.png')
plt.cla()

# -------------------- #
Nmax = 10
Ns = range(1, Nmax+1)
ys = [*map(lambda N: box_number(N, N), Ns)]
plt.plot(Ns, ys, marker='s')
plt.legend()
plt.title("number of boxes to get k types of coupons(k=N)")
plt.grid(linestyle=':')
plt.xlabel('N')
plt.ylabel('expected number of boxes')
plt.savefig('./img/7-b.png')
plt.cla()

# %% 8
# Tic-tac-toe


class Player:
    def __init__(self, ai):
        self.ai = ai

    def reset(self, mark=0, rival=None):
        self.mark = mark
        self.rival = rival
        self.row_score = [0, 0, 0]
        self.col_score = [0, 0, 0]
        self.diag_score = 0
        self.sub_diag_score = 0

    def select(self, vacancies):
        return self.ai(self, self.rival, vacancies)

    def iswin(self):
        if 3 in (*self.row_score, *self.col_score, self.diag_score, self.sub_diag_score):
            return True
        return False

    def score(self, pos):
        i, j = pos
        self.row_score[i] += 1
        self.col_score[j] += 1
        if i == j:
            self.diag_score += 1
        elif i+j == 2:
            self.sub_diag_score += 1
        return self.iswin()


def Tictactoe_summary(results):
    turns = len(results)
    results = np.array(results)
    # [p1 win, p2 win, draw, turns]
    return [*(np.sum(results == x)/turns for x in (0, 1, -1)), turns]


@repeat_within(time_tol=1, summary=Tictactoe_summary)
def tictactoe(p0, p1):
    players = (p0, p1)
    p0.reset(mark=0,rival=p1)
    p1.reset(mark=1,rival=p0)
    vacancies = [(i, j) for j in range(3) for i in range(3)]
    who = np.random.randint(2)
    while vacancies:
        player = players[who]
        pos, winner = player.select(vacancies)
        if winner is not None or player.score(pos):
            return winner or who
        vacancies.remove(pos)
        who = (who+1) % 2
    return -1

# %% 8-a


def rand_choice(self: Player, rival: Player, vacancies):
    choice = np.random.randint(len(vacancies))
    return vacancies[choice], None


Alice = Player(ai=rand_choice)
Bob = Player(ai=rand_choice)

tictactoe(Alice, Bob)

# %% 8-b


def bob_ai(self, rival, vacancies):
    # 1. pick centre
    if (1, 1) in vacancies:
        return (1, 1), None
    # 2. pick randomly
    return rand_choice(self, rival, vacancies)


def alice_ai(self: Player, rival: Player, vacancies):
    # 1. alice try to win
    vrs, vcs = zip(*vacancies)
    for r, s in enumerate(self.row_score):
        if s != 2:
            continue
        if r in vrs:
            return None, self.mark
    for c, s in enumerate(self.col_score):
        if s != 2:
            continue
        if c in vcs:
            return None, self.mark

    if self.diag_score == 2:
        for r, c in vacancies:
            if r == c:
                return None, self.mark

    if self.sub_diag_score == 2:
        for r, c in vacancies:
            if r+c == 2:
                return None, self.mark

    # 2. try to let bob win in next turn
    vacxbob = False
    # (n-1) vacancy for bob to win
    # only 1 vacancy for bob not win
    for r,c in vacancies:
        if r==c and rival.diag_score == 2:
            continue
        if r+c==2 and rival.sub_diag_score == 2:
            continue
        if rival.row_score[r] == 2:
            continue
        if rival.col_score[c] == 2:
            continue
        # this (r,c) not for bob to win
        if vacxbob:
            # at least 2 vacancy for bob not win
            break
        vacxbob = True
    else:
        if vacxbob:
            return None, rival.mark

    # 3. pick randomly
    return rand_choice(self, rival,vacancies)

Alice = Player(ai=alice_ai)
Bob = Player(ai=bob_ai)

tictactoe(Alice, Bob)


# %% 9
# Magic command


class TimeitCustom:
    def __init__(self, repeat=7, loops=10000000):
        self.repeat = repeat
        self.loops = loops
        self.average = 0
        self.stdev = 0

    def __call__(self, target, *args, **kwds):
        running_time = []
        for _ in range(self.repeat):
            start_time = time.time()
            for _ in range(self.loops):
                target(*args, **kwds)
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
    delta_sq = b**2-4*a*c
    return (-b+delta_sq**.5)/2/a, (-b-delta_sq**.5)/2/a
