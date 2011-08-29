from __future__ import print_function

from cStringIO import StringIO
from collections import defaultdict
import sys

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
        w = self.w
        h = self.h
        buf = StringIO()
        for i in xrange(0, w*h, w):
            l = self.state[i:i+w].replace('0', ' ')
            buf.write(l+'\n')
        return buf.getvalue()

# limits are L,R,U,D

def read_problem():
    with open('problems.txt') as f:
        L = f.readline().strip()
        LIMITS = map(int, L.split())
        debug('LX={0}, RX={1}, UX={2}, DX={3}'.format(*LIMITS))

        num_problems = int(f.readline().strip())

        boards = []
        for L in f:
            L = L.strip()
            w, h, state = L.split(',')
            boards.append(Board(int(w), int(h), state))

        assert len(boards) == num_problems

    return LIMITS, boards


from collections import deque

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

def solve_slide(board):
    W = board.w
    H = board.h
    Z = W*H

    G = make_goal(board.state)
    debug("Goal:", G)

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

    # moving functions.
    # `pos`, `state` and `route` are enclosed.
    def trymove(from_, to_, d, visited, q):
        if not(0 <= to_ < Z) or state[to_] == '=':
            return
        newstate = bytearray(state)
        newstate[from_], newstate[to_] = state[to_], state[from_]
        newstate = bytes(newstate)
        #if newstate == G:
        #    debug("Goal!:", route+d)
        #    goal_route.append(route+d)
        if newstate in visited:
            return
        visited.add(newstate)
        q.append((to_, newstate, route+d))
        return newstate

    for step in xrange(1,15):
        nq = deque()
        old_v1 = set(visited)
        debug("step:", step, "visited:", len(visited), "queue:", len(Q))
        while Q:
            pos, state, route = Q.popleft()
            if state in bvisited:
                answer = join_route(route, state, BQ)
                goal_route.append(answer)
                debug("match:", answer)
                continue
            trymove(pos, pos-W, 'U', visited, nq)
            trymove(pos, pos-1, 'L', visited, nq)
            trymove(pos, pos+W, 'D', visited, nq)
            trymove(pos, pos+1, 'R', visited, nq)
        if goal_route: return goal_route
        visited -= old_v2
        old_v2 = old_v1
        Q = nq

        debug("back step:", step, "visited:", len(bvisited), "queue:", len(BQ))
        nq = deque()
        old_bv1 = set(bvisited)
        while BQ:
            pos, state, route = BQ.popleft()
            if state in visited:
                answer = join_route_back(route, state, Q)
                goal_route.append(answer)
                debug("back match:", answer)
                continue
            trymove(pos, pos-W, 'D', bvisited, nq)
            trymove(pos, pos-1, 'R', bvisited, nq)
            trymove(pos, pos+W, 'U', bvisited, nq)
            trymove(pos, pos+1, 'L', bvisited, nq)
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

def main():
    limits, boards = read_problem()
    for i, b in enumerate(boards):
        debug("start solving", i)
        routes = solve_slide(b)
        print(i, repr(routes))

if __name__ == '__main__':
    #test()
    main()
