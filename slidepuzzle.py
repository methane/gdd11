#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

from cStringIO import StringIO
from collections import defaultdict, deque
import os
import sys
import json
from pprint import pprint
import pickle
import time
import random

# 循環参照はつくらないようにしているので、無駄なGCを止める.
import gc
gc.disable()

def debug(*args):
    print(*args, file=sys.stderr)

PLATES = '123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0'

class Board(object):
    __slots__ = ['w', 'h', 'state']

    def __init__(self, w, h, state):
        self.w = w
        self.h = h
        self.state = state

    def __str__(self):
        return print_board(self.w, self.h, self.state)

def print_board(w, h, state):
    buf = StringIO()
    for i in xrange(0, w*h, w):
        l = state[i:i+w].replace('0', ' ')
        buf.write(l+'\n')
    return buf.getvalue()

def check_route(board, route):
    w = board.w
    h = board.h
    state = bytearray(board.state)
    pos = board.state.index(b'0')

    try:
        for c in route:
            if c == 'D':
                npos = pos+w
                if npos >= w*h:
                    return False
            elif c == 'U':
                npos = pos-w
                if npos < 0:
                    return False
            elif c == 'L':
                if pos%w == 0:
                    return False
                npos = pos-1
            elif c == 'R':
                if (pos+1)%w == 0:
                    return False
                npos = pos+1
            else:
                return False
            if state[npos] == b'=' or state[pos] == b'=':
                return False
            state[pos], state[npos] = state[npos], state[pos]
            pos = npos
    except IndexError:
        return False

    expected = make_goal(board.state)
    result = bytes(state)
    return expected == result


DATAFILE = 'slide.data'

def load_data():
    if not os.path.exists(DATAFILE):
        return {}
    with open(DATAFILE, 'rb') as f:
        return pickle.load(f)

def save_data(data):
    if os.path.exists(DATAFILE):
        os.rename(DATAFILE, DATAFILE + '.old')
    with open(DATAFILE, 'wb') as f:
        pickle.dump(data, f)


def trace_route(board, route):
    w = board.w
    h = board.h
    state = bytearray(board.state)

    print("initial state")
    print(print_board(w, h, board.state))

    pos = board.state.index('0')

    for i in xrange(0, len(route), 2):
        subr = route[i:i+2]
        print("trace:", subr)
        for c in route[i:i+2]:
            if c == 'D':
                npos = pos+w
            elif c == 'U':
                npos = pos-w
            elif c == 'L':
                npos = pos-1
            else: #R
                npos = pos+1

            state[pos], state[npos] = state[npos], state[pos]
            pos = npos
        print(print_board(w, h, bytes(state)))



# limits are L,R,U,D

def read_problem():
    with open('problems.txt') as f:
        L = f.readline().strip()
        LIMITS = map(int, L.split())

        num_problems = int(f.readline().strip())

        boards = []
        for L in f:
            L = L.strip()
            w, h, state = L.split(',')
            boards.append(Board(int(w), int(h), state))

        assert len(boards) == num_problems

    return LIMITS, boards


def make_goal(state):
    N = len(state)
    state = bytearray(state)
    for i in xrange(N):
        if state[i] != ord('='):
            state[i] = PLATES[i]
    state[-1] = '0'
    return bytes(state)


def better_route(L, R):
    lc = L.count
    rc = R.count
    return (lc('L') <= rc('L') and
            lc('R') <= rc('R') and
            lc('U') <= rc('U') and
            lc('D') <= rc('D'))

def cmd_test(args):
    test_board = Board(3,2,b"012453")
    #test_board = Board(6,6,b"71=45=28B0AID=CF9OJ===GHWVRSNZQP==UT")
    debug(str(test_board))
    print(iterative_deeping(test_board))

def solve_inner(problem):
    import _slide
    i,b = problem
    debug("start solving", i)
    routes = _slide.iterative_deeping(b.w, b.h, b.state)
    #routes = _slide.solve_slide(b.w, b.h, b.state)
    #routes = _slide.solve2(b.w, b.h, b.state)
    #routes = _slide.solve_brute_force(b.w, b.h, b.state)
    #routes = _slide.solve_brute_force2(b.w, b.h, b.state)
    #routes = _slide.solve_combined(b.w, b.h, b.state)
    return i,routes

def solve(which=None):
    of = sys.stdout
    limits, boards = read_problem()
    if which is None:
        which = range(len(boards))

    procs = os.environ.get('SLIDE_PROCS')
    if procs:
        # parallel processing
        from multiprocessing import Pool
        pool = Pool(int(procs))
        problems = [(i,boards[i]) for i in which]
        for i, routes in pool.imap_unordered(solve_inner, problems):
            if routes:
                print(i, repr(routes), file=of)
                of.flush()
    else:
        # single processing
        for i in which:
            b = boards[i]
            i, routes = solve_inner((i, b))
            if routes:
                print(i, repr(routes), file=of)
                of.flush()

def merge_result(l, r):
    for k in r:
        if k not in l:
            l[k] = r[k]
        else:
            l[k].extend(r[k])

def read_routes(fn='routes.txt'):
    routes = {}
    with open(fn, 'r') as f:
        for L in f:
            n,r = L.split(None, 1)
            n = int(n)
            r = eval(r)
            if r:
                X = routes.setdefault(n, [])
                X.extend(r)
    return routes

def print_routes(routes):
    for i in xrange(5000):
        L = routes.get(i)
        if not L:
            print()
            continue
        print(L[0])

def check_routes(boards, routes):
    for k,v in routes.items():
        for r in v:
            if not check_route(boards[k], r):
                debug("Checking", k, "route:", r)
                trace_route(boards[k], r)

def cmd_solve(args):
    if args:
        which = []
        for arg in args:
            if '-' in arg:
                s,e = arg.split('-')
                if s and e:
                    which.extend(range(int(s), int(e)))
                elif s:
                    which.extend(range(int(s), 5000))
                elif e:
                    which.extend(range(0, int(e)))
            else:
                which.append(int(arg))
        solve(which)
    else:
        solve()

def cmd_check(args):
    limits, boards = read_problem()

    if args:
        for fn in args:
            data = read_routes(fn)
            check_routes(boards, data)
    else:
        data = load_data()
        check_routes(boards, data)

def cmd_load(args):
    limits, boards = read_problem()
    data = load_data()
    for f in args:
        debug("Loading:", f)
        newdata = read_routes(f)
        for k,routes in newdata.iteritems():
            L = data.setdefault(k, [])
            for r in routes:
                if not isinstance(r, bytes):
                    continue
                if r not in L and check_route(boards[k], r):
                    L.append(r)
            L.sort(key=len)
    save_data(data)

def cmd_trace(args):
    n = int(args[0])
    data = load_data()
    L, B = read_problem()
    trace_route(B[n], data[n][0])

def cmd_dump(args):
    data = load_data()
    pprint(data)

def cmd_missing(args):
    data = load_data()
    for i in xrange(5000):
        if i not in data or not(data[i]):
            print(i)

def cmd_solve_missing(args):
    which = map(int, open(args[0]))
    random.shuffle(which)
    solve(which)

def cmd_answer(args):
    LIMITS, BOARDS = read_problem()
    LX, RX, UX, DX = LIMITS
    L = R = U = D = 0
    data = load_data()
    answered = 0
    # todo: 手順の短い順に出力する.
    for i in xrange(5000):
        d = data.get(i, [])
        for r in d: # todo: 残り手数を考慮してソートする.
            lc = r.count('L'); rc = r.count('R')
            uc = r.count('U'); dc = r.count('D')
            if lc+L > LX or rc+R > RX or uc+U > UX or dc+D > DX:
                continue
            print(r)
            L += lc; R += rc
            U += uc; D += dc
            answered += 1
            break
        else:
            print()

    TOTAL = L+R+U+D
    TOTALX= LX+RX+UX+DX
    remans = 5000 - answered
    remstep = TOTALX-TOTAL
    debug("answerd: {answered} L={L}/{LX}, R={R}/{RX}, U={U}/{UX}, D={D}/{DX}\n"
          "TOTAL={TOTAL}/{TOTALX} remain={remans}/{remstep}".format(**vars()))

def shorten(r):
    while True:
        s = r
        s = s.replace('UD', '').replace('DU', '')
        s = s.replace('LR', '').replace('RL', '')
        if r == s:
            return s
        r = s

def cmd_shorten(_):
    data = load_data()
    for i, L in data.items():
        for j, r in enumerate(L):
            s = shorten(r)
            if s != r:
                debug("Shorten", i)
                debug("FROM:", r)
                debug("TO  :", s)
                L[j]=s
            u = list(set(L))
            u.sort(key=len)
            L[:]=u
    save_data(data)

def main():
    random.seed(int(time.time()))
    if len(sys.argv) < 2:
        debug("commands: solve load dump answer missing")
        return

    cmd = sys.argv[1]
    args = sys.argv[2:]

    fun = globals()['cmd_' + cmd]
    fun(args)

if __name__ == '__main__':
    main()
