from __future__ import print_function

from cStringIO import StringIO
from collections import defaultdict

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
        print('LX={0}, RX={1}, UX={2}, DX={3}'.format(*LIMITS))

        num_problems = int(f.readline.strip())

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


def solve_slide(board):
    W = board.w
    H = board.h
    Z = W*H

    G = make_goal(board.state)
    print("Goal:", G)

    Q = deque()
    state = board.state
    pos = state.index(b'0')
    Q.append((pos, state, ''))
    visited = set()
    visited.add(bytes(board.state))
    old_v1 = set()

    BQ = deque()
    pos = G.index(b'0')
    BQ.append((pos, G, ''))
    bvisited = set()
    bvisited.add(G)

    goal_route = []

    # moving functions.
    # `pos`, `state` and `route` are enclosed.
    def trymove(from_, to_, d, visited, q):
        if not(0 <= to_ < Z) or state[to_] == '=':
            return
        newstate = bytearray(state)
        newstate[from_], newstate[to_] = state[to_], state[from_]
        newstate = bytes(newstate)
        if newstate == G:
            print("Goal!:", route+d)
            goal_route.append(route+d)
        if newstate in visited:
            return
        visited.add(newstate)
        q.append((to_, newstate, route+d))
        return newstate

    step = 0
    while True:
        nq = deque()
        old_v2 = old_v1
        old_v1 = set(visited)
        step += 1
        print("step:", step, "visited:", len(visited), "queue:", len(Q))
        while Q:
            pos, state, route = Q.popleft()
            if state in bvisited:
                print("match")
                break

            trymove(pos, pos-W, 'U', visited, nq)
            trymove(pos, pos+W, 'D', visited, nq)
            trymove(pos, pos-1, 'L', visited, nq)
            trymove(pos, pos+1, 'R', visited, nq)
        visited -= old_v2
        Q = nq
        if goal_route:
            break

        while BQ:
            break
            #pos, state, route = BQ.popleft()
            #if state in visited:
            #    print("match")
            #    break
            #if lc % 10000 == 0:
            #    print('back', len(route), len(BQ), len(bvisited)) 
            #trymove(pos, pos-W, 'D', bvisited, BQ)
            #trymove(pos, pos+W, 'U', bvisited, BQ)
            #trymove(pos, pos-1, 'R', bvisited, BQ)
            #trymove(pos, pos+1, 'L', bvisited, BQ)
    return goal_route

def test():
    #test_board = Board(3,3,b"168452=30")
    test_board = Board(6,6,b"71=45=28B0AID=CF9OJ===GHWVRSNZQP==UT")
    solve_slide(test_board)

if __name__ == '__main__':
    test()
