# coding: utf-8
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

cdef int dist(int w, int z, bytes from_, bytes to_):
    """
    w: width, z: width*height, from_: from state, to_: to state.
    """
    cdef int dist=0, i, c, index
    cdef char *f, *t
    f = from_
    t = to_
    for i in xrange(z):
        c = f[i]
        if c in b'=0':
            continue
        for index in xrange(z):
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

    dist_limit_b = dist_limit = dist(W, Z, G, S) + (W+H)*2 + 20

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

        if dist(W, Z, newstate, goal) < limit:
            visited.add(newstate)
            q.append((to_, newstate, route+d))

    for step in xrange(1,200):
        if not Q or not BQ:
            debug("No sattes. Give up")
            break
        if len(Q) > QMAX:
            hist = defaultdict(int)
            for pos,state,route in Q:
                hist[dist(W, Z, state, G)] += 1
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
            if dist(W, Z, state,G) > dist_limit:
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
                hist[dist(W, Z, state,S)] += 1
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
            if dist(W, Z, state,S) > dist_limit_b:
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


#### ID DFS

cdef bytes move(char* state, int pos, int npos):
    n = PyBytes_FromString(state)
    cdef char *p = n
    p[pos],p[npos] = n[npos],n[pos]
    return n

cdef int char_to_gpos(int c):
    cdef int c1='1', c9='9', ca='A', cz='Z'

    if c1 <= c <= c9:
        return c-c1;
    if ca <= c <= cz:
        return c-ca+9
    return -1

cdef int dist_diff(char *state, int pos, int npos, int W):
    u"""state の pos にある空白を npos に移動したときの、
    マンハッタン距離の変化を返す."""

    # 空白が pos->npos なので、タイルは npos->pos に移動する.
    cdef int c = state[npos]
    cdef int gpos = char_to_gpos(c) 

    cdef int prev_dist = _abs(gpos//W - npos//W) + _abs(gpos%W - npos%W)
    cdef int next_dist = _abs(gpos//W -  pos//W) + _abs(gpos%W -  pos%W)

    return next_dist - prev_dist


cdef int slide_dfs(int W, int Z, bytes G, int pos, bytes state, int _dist, int dlimit,
                   bytes route, int remain, list answer, set visited):
    cdef int nlimit = dlimit - 1
    cdef int ndist

    if state == G:
        debug("Goal:", route)
        answer.append(route)
        return 0

    if dlimit == 0:
        return min(_dist, remain)
    if dlimit < _dist:
        return min(_dist-dlimit, remain)


    if pos>W and route[-1] !='D':
        npos = pos-W
        if state[npos] != '=':
            ns = move(state, pos, npos)
            if ns not in visited:
                ndist = _dist + dist_diff(state, pos, npos, W)
                remain = slide_dfs(W,Z,G, npos, ns, ndist, nlimit, route+'U', remain, answer, visited)
    if pos%W and route[-1] !='R':
        npos = pos-1
        if state[npos] != '=':
            ns = move(state, pos, npos)
            if ns not in visited:
                ndist = _dist + dist_diff(state, pos, npos, W)
                remain = slide_dfs(W,Z,G, npos, ns, ndist, nlimit, route+'L', remain, answer, visited)
    if pos+W<Z and route[-1] !='U':
        npos = pos+W
        if state[npos] != '=':
            ns = move(state, pos, npos)
            if ns not in visited:
                ndist = _dist + dist_diff(state, pos, npos, W)
                remain = slide_dfs(W,Z,G, npos, ns, ndist, nlimit, route+'D', remain, answer, visited)
    if (pos+1)%W and route[-1] !='L':
        npos = pos+1
        if state[npos] != '=':
            ns = move(state, pos, npos)
            if ns not in visited:
                ndist = _dist + dist_diff(state, pos, npos, W)
                remain = slide_dfs(W,Z,G, npos, ns, ndist, nlimit, route+'R', remain, answer, visited)
    return remain

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

    cdef int depth_limit = min(dist(W, Z, s[1],G) for s in Q)
    cdef int _dist = dist(W,Z, S,G)
    cdef bytes s

    while not goal_route:
        debug("Iterative DFS: depth limit =", depth_limit)
        for pos,s,route in Q:
            _dist = slide_dfs(W, Z, G, pos, s, dist(W,Z,s,G), depth_limit, route, _dist, goal_route, visited)
        depth_limit += _dist

    return goal_route


