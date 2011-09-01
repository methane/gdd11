from __future__ import print_function

cdef extern from *:
    ctypedef char const_char "const char"

from libc.string cimport strchr
from cpython.bytes cimport PyBytes_FromString

from collections import defaultdict, deque
import sys

def debug(*args):
    print(*args, file=sys.stderr)

PLATES = b'123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0'


def make_goal(state):
    N = len(state)
    state = bytearray(state)
    for i in xrange(N):
        if state[i] != ord('='):
            state[i] = PLATES[i]
    state[-1] = '0'
    return bytes(state)

def join_route(route, state, remain):
    for p, s, r in remain:
        if state == s:
            return route + r[::-1]

def join_route_back(broute, state, remain):
    for p, s, r in remain:
        if state == s:
            return r + broute[::-1]

cdef int _abs(int x):
    if x < 0: return -x
    return x

cdef int dist(int w, int h, bytes from_, bytes to_):
    cdef int dist=0, i, c, index
    cdef char *f, *t
    f = from_
    t = to_
    for i in xrange(w*h):
        c = f[i]
        if c in b'=0':
            continue
        for index in xrange(w*h):
            if t[index] == c:
                break
        dist += _abs(i//w - index//w)
        dist += _abs(i%w - index%w)
    return dist


def solve_slide(board, int QMAX=200000):
    cdef int W, H, Z
    cdef int dist_limit, dist_limit_b
    cdef bytes state, S, G, route
    cdef int pos, i

    W = board.w
    H = board.h
    Z = W*H

    S = board.state
    G = make_goal(S)
    debug("Start:", S)
    debug("Goal: ", G)

    dist_limit_b = dist_limit = dist(W, H, G, S) + (W+H)*2 + 20

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

    def trymove(int W, int H, bytes state, int from_, int to_,
                 bytes d, set visited, q, set rvisited, remain,
                 joinfun, bytes goal, int limit):
        cdef char *pc

        if not(0 <= to_ < Z) or state[to_] == b'=':
            return

        newstate = PyBytes_FromString(state)
        pc = newstate
        pc[from_], pc[to_] = pc[to_], pc[from_]

        if newstate in visited:
            return
        if newstate in rvisited:
            ans = joinfun(route + d, newstate, remain)
            debug("Goal:", ans)
            goal_route.append(ans)
            return

        if dist(W, H, newstate, goal) < limit:
            visited.add(newstate)
            q.append((to_, newstate, route+d))

    for step in xrange(1,200):
        if not Q or not BQ:
            debug("No sattes. Give up")
            break
        if len(Q) > QMAX:
            hist = defaultdict(int)
            for pos,state,route in Q:
                hist[dist(W, H, state, G)] += 1
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
            if dist(W,H,state,G) > dist_limit:
                continue
            if state in bvisited:
                answer = join_route(route, state, BQ)
                goal_route.append(answer)
                debug("match:", answer)
                continue
            trymove(W,H,state, pos, pos-W, 'U', visited, nq, bvisited, BQ,
                    join_route, G, dist_limit)
            if pos%W:
                trymove(W,H,state, pos, pos-1, 'L', visited, nq, bvisited, BQ,
                        join_route, G, dist_limit)
            trymove(W,H,state, pos, pos+W, 'D', visited, nq, bvisited, BQ,
                    join_route, G, dist_limit)
            if (pos+1)%W:
                trymove(W,H,state, pos, pos+1, 'R', visited, nq, bvisited, BQ,
                        join_route, G, dist_limit)
        if goal_route: return goal_route
        visited -= old_v2
        old_v2 = old_v1
        Q = nq

        if len(BQ) > QMAX:
            hist = defaultdict(int)
            for pos,state,route in BQ:
                hist[dist(W,H,state,S)] += 1
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
            if dist(W,H,state,S) > dist_limit_b:
                continue
            if state in visited:
                answer = join_route_back(route, state, Q)
                goal_route.append(answer)
                debug("back match:", answer)
                continue
            trymove(W,H,state, pos, pos-W, 'D', bvisited, nq, visited, Q,
                    join_route_back, S, dist_limit_b)
            if pos%W:
                trymove(W,H,state, pos, pos-1, 'R', bvisited, nq, visited, Q,
                        join_route_back, S, dist_limit_b)
            trymove(W,H,state, pos, pos+W, 'U', bvisited, nq, visited, Q,
                    join_route_back, S, dist_limit_b)
            if (pos+1)%W:
                trymove(W,H,state, pos, pos+1, 'L', bvisited, nq, visited, Q,
                        join_route_back, S, dist_limit_b)
        if goal_route: return goal_route
        bvisited -= old_bv2
        old_bv2 = old_bv1
        BQ = nq

    return goal_route


def iterative_deeping(board):
    """Iterative deeping DFS. But use BFS for some initial steps."""
    cdef int QMAX, W, H, Z
    cdef bytes S,G
    cdef int pos

    QMAX = 200000 # Switch to IDDFS when number of states overs this.

    W = board.w
    H = board.h
    Z = W*H

    S = board.state
    G = make_goal(S)
    debug("Goal:", G)

    Q = deque()
    state = board.state
    pos = state.index(b'0')
    Q.append((pos, state, ''))
    visited = set()
    visited.add(bytes(state))
    old_v2 = old_v1 = set()

    goal_route = []

    def trymove(int from_, int to_, d, visited, q,
                distfun=None, limit=0):
        if not(0 <= to_ < Z) or state[to_] == '=':
            return
        newstate = bytearray(state)
        newstate[from_], newstate[to_] = state[to_], state[from_]
        newstate = bytes(newstate)

        if newstate in visited:
            return

        if newstate == G:
            goal_route.append(route+d)

        if distfun and limit and distfun(newstate) >= limit:
            return
        visited.add(newstate)
        q.append((to_, newstate, route+d))
        return newstate

    # BFS
    cdef int step = 0
    while len(Q) < QMAX and not goal_route:
        step += 1
        nq = deque()
        old_v1 = set(visited)
        debug("step:", step, "visited:", len(visited), "queue:", len(Q))
        while Q:
            pos, state, route = Q.popleft()
            trymove(pos, pos-W, 'U', visited, nq)
            if pos%W:
                trymove(pos, pos-1, 'L', visited, nq)
            trymove(pos, pos+W, 'D', visited, nq)
            if (pos+1)%W:
                trymove(pos, pos+1, 'R', visited, nq)
        visited -= old_v2
        old_v2 = old_v1
        Q = nq

    if goal_route:
        return goal_route
    del old_v1, old_v2

    debug("Start iterative deeping.")

    def move(state, pos, npos):
        n = bytearray(state)
        n[pos],n[npos] = n[npos],n[pos]
        return bytes(n)

    def slide_dfs(int pos, bytes state, int dlimit, bytes route,
                  int remain, answer):
        #print(print_board(W, H, state))
        if state == G:
            debug("Goal:", route)
            answer.append(route)
            return 0

        cdef int _dist = dist(W,H,state,G)
        if dlimit == 0:
            return min(_dist, remain)
        if dlimit < _dist:
            return min(_dist-dlimit, remain)

        nlimit = dlimit - 1

        if pos>W and route[-1] !='D':
            npos = pos-W
            if state[npos] != '=':
                ns = move(state, pos, npos)
                if ns not in visited:
                    remain = slide_dfs(npos, ns, nlimit, route+'U', remain, answer)
        if pos%W and route[-1] !='R':
            npos = pos-1
            if state[npos] != '=':
                ns = move(state, pos, npos)
                if ns not in visited:
                    remain = slide_dfs(npos, ns, nlimit, route+'L', remain, answer)
        if pos+W<Z and route[-1] !='U':
            npos = pos+W
            if state[npos] != '=':
                ns = move(state, pos, npos)
                if ns not in visited:
                    remain = slide_dfs(npos, ns, nlimit, route+'D', remain, answer)
        if (pos+1)%W and route[-1] !='L':
            npos = pos+1
            if state[npos] != '=':
                ns = move(state, pos, npos)
                if ns not in visited:
                    remain = slide_dfs(npos, ns, nlimit, route+'R', remain, answer)
        return remain

    cdef int depth_limit = min(dist(W,H,s[1],G) for s in Q)
    cdef int _dist = dist(W,H,S,G)
    cdef bytes s

    while not goal_route:
        debug("Iterative DFS: depth limit =", depth_limit)
        for pos,s,route in Q:
            _dist = slide_dfs(pos, s, depth_limit, route, _dist, goal_route)
        depth_limit += _dist

    return goal_route


