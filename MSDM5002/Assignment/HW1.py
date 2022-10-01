#%% 1
# A falling ball

def get_input(question, max_input_times, default_value):
    for times in range(max_input_times+1):
        uncheck_value = input(question)
        try:
            val = float(uncheck_value)
        except ValueError:
            times < max_input_times and print("Your input should be a number")
            continue
        if val < 0:
            times < max_input_times and print("Your input should be a POSITIVE number")
            continue
        return val
    print(f"You are so stupid! I have to stop you and set its value to {default_value}.")
    return default_value

#%% 2
# Copyright statements

from datetime import date
TODAY = str(date.today())

def find_last(str,target=' '):
    return len(str)-str[::-1].index(target) - 1

def cut_at(str,max_len):
    if str[max_len-1].isalpha():
        if str[max_len].isalpha() or str[max_len] in ',.':
            return find_last(str[:max_len])
    return max_len

def get_email_lines(str,max_len,email):
    while len(str) > max_len:
        cut_idx = cut_at(str,max_len)
        yield str[:cut_idx].ljust(max_len)
        str = str[cut_idx:].strip()
    str += f' "{email}"'
    while len(str) > max_len:
        yield str[:max_len]
        str = str[max_len:]
    yield str.ljust(max_len)

def my_copyright4(name,email,date=TODAY):
    name_content_line = f"programmed by {name} for MSDM5002"
    content_width = len(name_content_line)
    date_content_line = f"date: {date}".center(content_width)
    email_content = f"You can use it as you like, but there might be many bugs. If you find some bugs, please send them to"
    email_lines = [f"***  {_}  ***" for _ in get_email_lines(email_content,content_width,email)]
    print(f"*****{content_width*'*'}*****")
    print(f"***  {name_content_line}  ***")
    print(f"***  {date_content_line}  ***")
    print(f"***--{content_width*'-'}--***")
    print(*email_lines,sep='\n')
    print(f"*****{content_width*'*'}*****")

my_copyright4('IA', 'ia@ust.hk')
my_copyright4('Alice & Bob', 'alice@wonder.land','2022-12-31')
my_copyright4('A', '0123456789'*10, '2022-09-01')

#%% 3
# Capitalization 

excluding = ("a", "an", "and", "as", "at", "but", "by", "for", "in", "nor", "of", "on", "or", "the", "to", "up")
spliters = ' -@&./'

def multi_split(string):
    word = ''
    for char in string:
        if char in spliters:
            yield word
            word = ''
            yield char
        else:
            word += char
    if word != '':
        yield word

def word_cap(word):
    return (word[0].upper()+word[1:]) if word not in excluding else word

def capitalize(sentence):
    return ''.join(map(word_cap,multi_split(sentence)))

string = input("Enter a sentence: ")
# Welcome to the world of data-driven modeling for master students offered by dept-of-phys&math@UST.hk on a great and green campus on a hill of@hill.
print(capitalize(string))

#%% 4
# Welcome

def welcome():
    names = []
    while (name := input("Enter a student’s name (Enter q/Q to stop): ")) not in 'qQ':
        names.append(name)
    for name in names:
        print(f"Hello {name}, welcome to the course 5002.")

welcome()

#%% 5
# Mountain patterns

def peak(h):
    max_len = 2*h-1
    for k in range(h-1):
        yield f"#{(max_len-2*k-2)*' '}#".center(max_len)
    yield '#'.center(max_len)
    while True:
        yield ''.center(max_len)

def peak_join(strs):
    return ''.join(str if i == 0 else str[1:] for i,str in enumerate(strs))

def mountain(hs):
    H = max(hs)
    for k,strs in enumerate(zip(*map(peak,hs))):
        if k >= H:
            return 
        yield peak_join(strs)

def show_mountain(hs):
    print(*list(mountain(hs))[::-1],sep='\n')

from random import randint 
heights = [randint(4, 10) for _ in range(3)] 
show_mountain(heights)

#%% 6
# Guess my number

from random import randint

def guess_number(lower = 1,upper = 100):
    target = randint(lower+1, upper-1)
    times = 0
    while True:
        guess = int(input(f"Guess my number: {lower} to {upper}!"))
        times += 1

        if guess <= lower or guess >= upper:
            print("Number out of range! Try again! ")
            continue

        if target == guess:
            print(f"Bingo! You got it in {times} guesses! ") 
            return
            
        if target < guess:
            upper = guess
        else: 
            # target > guess
            lower = guess

guess_number(1,100)

#%% 7
# Quiz 4
# count 1,0,-1 in 3d matrix

from random import randint, seed
import numpy as np

n=20;m=30;p=40;

student_id = 12345678

seed(student_id) ### replace 10 by your student ID

A =np.array([[[randint(-1,1) for x in range(n)] for y in range(m)] for z in range(p)])

# method one: using np.sum
def count(element):
    return np.sum(A == element)
print([student_id,*map(count,(1,0,-1))])