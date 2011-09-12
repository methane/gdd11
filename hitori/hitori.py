#!/usr/bin/env python
# coding: utf-8

# ただの深さ優先探索.
# 無限ループしないように、 /2 操作で変化がない場合を切る.

def read_input():
    problem = []
    with open('input.txt', 'r') as f:
        N = int(f.readline().strip())
        i = 0
        while True:
            l = f.readline().strip()
            if not l:
                if i != N:
                    raise Exception("Can't read N problems.")
                return problem
            k = int(l)
            l = f.readline().strip()
            p = map(int, l.split())
            assert k == len(p)
            problem.append(p)
            i += 1

def op_x(n):
    return [i//2 for i in n]

def op_y(n):
    return [i for i in n if i%5!=0]

def solve(numbers):
    if not numbers:
        return 0
    
    x = op_x(numbers)
    if x == numbers: # They're all 0
        return 1
    xstep = solve(x)

    y = op_y(numbers)
    if not y:
        return 1
    if y == numbers:
        return xstep+1

    ystep = solve(y)
    return min(xstep, ystep) + 1

def main():
    problems = read_input()
    for p in problems:
        print solve(p)

if __name__ == '__main__':
    main()
