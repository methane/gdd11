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
    pos = board.state.index('0')

    try:
        for c in route:
            if c == 'D':
                npos = pos+w
            elif c == 'U':
                npos = pos-w
            elif c == 'L':
                if pos%w == 0:
                    return False
                npos = pos-1
            else: #R
                if (pos+1)%w == 0:
                    return False
                npos = pos+1
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
        #debug('LX={0}, RX={1}, UX={2}, DX={3}'.format(*LIMITS))

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

def append_route(routes, new_route):
    for old_route in routes:
        if better_route(old_route, new_route):
            return
        if better_route(new_route, old_route):
            routes[routes.index(old_route)] = new_route
            return
    routes.append(new_route)

def join_route(route, state, remain):
    for p, s, r in remain:
        if state == s:
            return route + r[::-1]

def join_route_back(broute, state, remain):
    for p, s, r in remain:
        if state == s:
            return r + broute[::-1]

def dist(w,h,from_,to_):
    dist = 0
    for i in xrange(w*h):
        c = from_[i]
        if c in '=0':
            continue
        index = to_.index(c)
        dist += abs(i//w - index//w)
        dist += abs(i%w - index%w)
    return dist

def make_dist_fun(w, h, target):
    s = """\
def dist(state):
    d = 0
    abs_ = abs
    ind = state.index

"""
    for i in xrange(w*h):
        c = target[i]
        if c in '0=':
            continue
        x = i%w
        y = i//w
        s += """\
    pos = ind('{c}')
    d += abs_(pos//{w} - {y})
    d += abs_(pos% {w} - {x})
""".format(**vars())
    s += """\
    return d
"""
    #print("dist function")
    #print(s)
    #print("end")
    env = {}
    exec s in env
    return env['dist']

def solve_slide(board):
    W = board.w
    H = board.h
    Z = W*H
    QMAX = 100000

    S = board.state
    G = make_goal(S)
    debug("Goal:", G)

    fwd_dist = make_dist_fun(W, H, G)
    back_dist = make_dist_fun(W, H, S)

    dist_limit_b = dist_limit = fwd_dist(S) + (W+H)*2

    Q = deque()
    state = board.state
    pos = state.index(b'0')
    Q.append((pos, state, ''))
    visited = set()
    visited.add(bytes(state))
    old_v2 = old_v1 = set()

    BQ = deque()
    pos = G.index(b'0')
    BQ.append((pos, G, ''))
    bvisited = set()
    bvisited.add(G)
    old_bv2 = old_bv1 = set()

    goal_route = []

    def trymove(from_, to_, d, visited, q, rvisited, remain,
                joinfun, distfun, limit):
        if not(0 <= to_ < Z) or state[to_] == '=':
            return
        newstate = bytearray(state)
        newstate[from_], newstate[to_] = state[to_], state[from_]
        newstate = bytes(newstate)

        if newstate in visited:
            return
        if newstate in rvisited:
            ans = joinfun(route + d, newstate, remain)
            debug("Goal:", ans)
            goal_route.append(ans)
            return

        if distfun(newstate) < limit:
            visited.add(newstate)
            q.append((to_, newstate, route+d))
            return newstate

    for step in xrange(1,100):
        if len(Q) > QMAX:
            hist = defaultdict(int)
            for pos,state,route in Q:
                hist[fwd_dist(state)] += 1
            z = 0
            for i in sorted(hist):
                z += hist[i]
                if z > QMAX:
                    dist_limit = i
                    break

        nq = deque()
        old_v1 = set(visited)
        debug("fwrd step:", step, "visited:", len(visited), "queue:", len(Q),
              "limit:", dist_limit)
        while Q:
            pos, state, route = Q.popleft()
            if fwd_dist(state) > dist_limit:
                continue
            if state in bvisited:
                answer = join_route(route, state, BQ)
                goal_route.append(answer)
                debug("match:", answer)
                continue
            trymove(pos, pos-W, 'U', visited, nq, bvisited, BQ,
                    join_route, fwd_dist, dist_limit)
            if pos%W:
                trymove(pos, pos-1, 'L', visited, nq, bvisited, BQ,
                        join_route, fwd_dist, dist_limit)
            trymove(pos, pos+W, 'D', visited, nq, bvisited, BQ,
                    join_route, fwd_dist, dist_limit)
            if (pos+1)%W:
                trymove(pos, pos+1, 'R', visited, nq, bvisited, BQ,
                        join_route, fwd_dist, dist_limit)
        if goal_route: return goal_route
        visited -= old_v2
        old_v2 = old_v1
        Q = nq

        if len(BQ) > QMAX:
            hist = defaultdict(int)
            for pos,state,route in BQ:
                hist[back_dist(state)] += 1
            z = 0
            for i in sorted(hist):
                z += hist[i]
                if z > QMAX:
                    dist_limit_b = i
                    break
        debug("back step:", step, "visited:", len(bvisited), "queue:", len(BQ),
              "limit:", dist_limit_b)
        nq = deque()
        old_bv1 = set(bvisited)
        while BQ:
            pos, state, route = BQ.popleft()
            if back_dist(state) > dist_limit_b:
                continue
            if state in visited:
                answer = join_route_back(route, state, Q)
                goal_route.append(answer)
                debug("back match:", answer)
                continue
            trymove(pos, pos-W, 'D', bvisited, nq, visited, Q,
                    join_route_back, back_dist, dist_limit_b)
            if pos%W:
                trymove(pos, pos-1, 'R', bvisited, nq, visited, Q,
                        join_route_back, back_dist, dist_limit_b)
            trymove(pos, pos+W, 'U', bvisited, nq, visited, Q,
                    join_route_back, back_dist, dist_limit_b)
            if (pos+1)%W:
                trymove(pos, pos+1, 'L', bvisited, nq, visited, Q,
                        join_route_back, back_dist, dist_limit_b)
        if goal_route: return goal_route
        bvisited -= old_bv2
        old_bv2 = old_bv1
        BQ = nq
    return goal_route

def test():
    #test_board = Board(3,3,b"168452=30")
    test_board = Board(6,6,b"71=45=28B0AID=CF9OJ===GHWVRSNZQP==UT")

    debug(str(test_board))
    solve_slide(test_board)

def solve(which=None):
    from _slide import solve_slide
    #of = open('routes-1.txt', 'w')
    of = sys.stdout
    limits, boards = read_problem()
    if which is None:
        which = range(len(boards))

    for i in which:
        b = boards[i]
        debug("start solving", i)
        routes = solve_slide(b)
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
    debug("answerd: {answered} L={L}/{LX}, R={R}/{RX}, U={U}/{UX}, D={D}/{DX}".
          format(**vars()))


def main():
    if len(sys.argv) < 2:
        debug("commands: solve load dump answer missing")
        return

    cmd = sys.argv[1]
    args = sys.argv[2:]

    fun = globals()['cmd_' + cmd]
    fun(args)

if __name__ == '__main__':
    main()
