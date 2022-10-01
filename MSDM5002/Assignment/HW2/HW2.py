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


def A_win_k_bullet_fast(num_game, k=3):
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

# %% 7
# Coupon collector

np.random.seed(20912792)


class Box:
    def __init__(self, N):
        self.coupon = np.random.randint(N)

    def open(self):
        return self.coupon

def repeat_in(time_tol=0.1,summary=np.mean):
    def decor(func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            start_time = time.time()
            results = []
            while (time.time() - start_time) <= time_tol:
                results.append(func(*args,**kwargs))
            return summary(results)
        return wrapper
    return decor

@repeat_in(time_tol=0.1,summary=np.mean)
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
plt.savefig('7-a')
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
plt.savefig('7-b')
plt.cla()

# %% 8
# Tic-tac-toe



# %% 10
# Quadratic equation


def quadratic(a,  b,  c):
    delta_sq = b**2-4*a*c
    return (-b+delta_sq**.5)/2/a, (-b-delta_sq**.5)/2/a
