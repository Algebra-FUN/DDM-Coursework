# %%0
# import all needed module at this cell
from unittest import result
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

def repeat_within(time_tol=0.1,summary=np.mean):
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

@repeat_within(time_tol=0.1,summary=np.mean)
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
    def __init__(self,name,ai):
        self.name = name
        self.ai = ai
        self.mark = None

    def __repr__(self):
        return self.name

    def play(self,game):
        return self.ai(self,game)

    def assign(self,number):
        self.mark = number

def Tictactoe_summary(results):
    turns = len(results)
    results = np.array(results)
    return [*(np.sum(results==x)/turns for x in (1,-1,0)),turns]

class Tictactoe:
    def __init__(self,player1,player2):
        self.broad = np.zeros((3,3))
        self.posp = player1
        self.posp.assign(1)
        self.negp = player2
        self.negp.assign(-1)

    @repeat_within(time_tol=1,summary=Tictactoe_summary)
    def __call__(self):
        self.broad = np.zeros((3,3))
        who = (-1)**np.random.randint(2)
        while 0 in self.broad:
            player = self.player(who)
            pos = player.play(self)
            self.broad[pos] = who
            if (status:=self.check()) != 0:
                return status
            who *= -1
        else:
            return 0

    def player(self,who):
        return self.posp if who == 1 else self.negp

    def pos(self,mark):
        return list(zip(*np.where(self.broad==mark)))

    def empty_pos(self):
        return self.pos(0)

    def check(self):
        for who in (-1,1):
            # check row
            for r in range(3):
                if all(self.broad[r,:] == who):
                    return who
            
            # check col
            for c in range(3):
                if all(self.broad[:,c] == who):
                    return who
            
            # check diag
            if all(np.diag(self.broad) == who):
                return who

        # check sub-diag
        if self.broad[0,2]==self.broad[1,1]==self.broad[2,0]:
            return self.broad[1,1]
        return 0

# %% 8-a
# 8-a

def rand_choice(self,game):
    empty_pos = game.empty_pos()
    choice = np.random.randint(len(empty_pos))
    return empty_pos[choice]

alice = Player('Alice',ai=rand_choice)
bob = Player('Bob',ai=rand_choice)

tictactoe = Tictactoe(alice,bob)
result = tictactoe()
print(result)

# %% 8-b
# 8-b

def bob_ai(self,game):
    # 1.picks the centre
    if (1,1) in game.empty_pos():
        return (1,1)

    # 2.picks any available at random
    return rand_choice(self,game)

def alice_ai(self,game):
    empty = game.empty_pos()
    # 1. helps her win immediately
    for pos in empty:
        game.broad[pos] = self.mark
        # alice's win pos
        if game.check() == self.mark:
            return pos
        game.broad[pos] = 0

    # 2. helps Bob win immediately
    # n-1 position is Bob's win pos
    # 1 position is not Bob's win pos
    n = len(empty)
    bob_win_pos_num = 0
    not_bob_win_pos = None
    for pos in empty:
        game.broad[pos] = -self.mark
        # bob's win pos
        if game.check() == -self.mark:
            bob_win_pos_num += 1
        else:
            not_bob_win_pos = pos
        game.broad[pos] = 0
    if bob_win_pos_num == n-1:
        return not_bob_win_pos

    # 3. picks any available at random
    return rand_choice(self,game)


alice = Player('Alice',ai=alice_ai)
bob = Player('Bob',ai=bob_ai)

tictactoe = Tictactoe(alice,bob)
result = tictactoe()
print(result)

# %% 10
# Quadratic equation


def quadratic(a,  b,  c):
    delta_sq = b**2-4*a*c
    return (-b+delta_sq**.5)/2/a, (-b-delta_sq**.5)/2/a
