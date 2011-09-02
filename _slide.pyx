# coding: utf-8
from __future__ import print_function

cdef extern from *:
    ctypedef char const_char "const char"

from libc.string cimport strchr
from libc.stdlib cimport rand, srand
from cpython.bytes cimport PyBytes_FromString

import sys
import string
from time import time
from collections import defaultdict, deque


def debug(*args):
    print(*args, file=sys.stderr)

PLATES = b'123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0'


_T = string.maketrans(b'LRUD', b'RLDU')
def reverse_route(route):
    u"""ルートを逆にする。
    reverse して L<>R, D<>U の置換を行う."""
    route = route[::-1]
    return route.translate(_T)


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
    cdef int dist_limit, dist_limit_b
    cdef bytes state, S, G, route
    cdef int pos, i

    cdef int W = board.w
    cdef int H = board.h
    cdef int Z = W*H

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


cdef int slide_dfs(int W, int Z, G, int pos, bytes state, int dlimit,
                   bytes route, list answer):
    if state in G:
        route += G[state]
        debug("Goal:", route)
        answer.append(route)
        return -1

    if dlimit <= 0:
        return 0

    cdef int ndist
    cdef int nlimit = dlimit - 1

    if pos>W and route[-1] !=b'D':
        npos = pos-W
        if state[npos] != '=':
            ns = move(state, pos, npos)
            nlimit = slide_dfs(W,Z,G, npos, ns, nlimit, route+b'U', answer)

    if pos%W and route[-1] !=b'R':
        npos = pos-1
        if state[npos] != '=':
            ns = move(state, pos, npos)
            nlimit = slide_dfs(W,Z,G, npos, ns, nlimit, route+b'L', answer)

    if pos+W<Z and route[-1] !=b'U':
        npos = pos+W
        if state[npos] != '=':
            ns = move(state, pos, npos)
            nlimit = slide_dfs(W,Z,G, npos, ns, nlimit, route+b'D', answer)

    if (pos+1)%W and route[-1] !=b'L':
        npos = pos+1
        if state[npos] != '=':
            ns = move(state, pos, npos)
            nlimit = slide_dfs(W,Z,G, npos, ns, nlimit, route+b'R', answer)

    return nlimit+1


cdef trymove(bytes state, int from_, int to_, bytes route, bytes d, visited, q, goals, goal_route,
             int dist, int dist_limit, int W):
    if state[to_] == b'=':
        return

    cdef int newdist = dist_diff(state, from_, to_, W) + dist
    if newdist > dist_limit:
        return

    newstate = PyBytes_FromString(<char*>state)
    cdef char *p = <char*>newstate
    p[from_], p[to_] = state[to_], state[from_]

    if newstate in visited:
        return
    if newstate in goals:
        goal_route[newstate] = route+d
        return

    visited.add(newstate)
    q.append((to_, newstate, route+d, newdist))


cdef limited_bfs(int W, int H, bytes initial_state, int threshold, stop,
                 bint cutdown=0, int step_limit=99999, int dist=0):
    u""" `W` x `H` マスの `initial_state` を元に、状態数が
    `threshold` を超えるか、 `stop` に含まれる(``in stop``)状態に到達するまで
    幅優先探索を行う。

    `threshold` を超えた場合は2要素のタプルを返す。1要素目は深さで、2要素目は
    状態をキーに、そこまでのルートを値にした辞書である。

    `stop` に到達した場合は、 `threshold` を超えた場合と似ているが、1要素目が
    0 になり、2要素目は到達した状態だけを含む.
    """
    cdef bytes state = initial_state
    cdef int pos = state.index(b'0')
    cdef int Z = W*H
    cdef bytes G = make_goal(state)

    cdef list Q = [(pos, state, b'', dist)]

    visited = set([state])
    old_v2 = old_v1 = set()

    cdef int step = 0
    goal_routes = {}

    while Q and not goal_routes:
        step += 1

        if len(Q) > threshold:
            if cutdown:
                debug("step=", step, "cutting from", len(Q))
                shuffle(Q)
                del Q[threshold:]
            else:
                break

        nq = []
        old_v1 = set(visited)

        while Q:
            pos, state, route, dist = Q.pop()

            if pos >= W:
                trymove(state, pos, pos-W, route, b'U', visited, nq, stop, goal_routes,
                        dist, step_limit-step, W)
            if pos % W:
                trymove(state, pos, pos-1, route, b'L', visited, nq, stop, goal_routes,
                        dist, step_limit-step, W)
            if pos+W < Z:
                trymove(state, pos, pos+W, route, b'D', visited, nq, stop, goal_routes,
                        dist, step_limit-step, W)
            if (pos+1)%W:
                trymove(state, pos, pos+1, route, b'R', visited, nq, stop, goal_routes,
                        dist, step_limit-step, W)
        visited -= old_v2
        old_v2 = old_v1
        Q = nq

    if goal_routes:
        return 0, goal_routes

    routes = {}
    while Q:
        pos, state, route, dist = Q.pop()
        routes[state] = route
    return step, routes



cpdef shuffle(list x):
    """x, random=random.random -> shuffle list x in place; return None.

    Optional arg random is a 0-argument function returning a random
    float in [0.0, 1.0); by default, the standard random.random.
    """
    cdef int i,j
    for i in xrange(len(x)-1, -1, -1):
        # pick an element in x[:i+1] with which to exchange x[i]
        j = rand() % (i+1)
        x[i], x[j] = x[j], x[i]


def iterative_deeping(board, int QMAX=400000):
    """Iterative deeping DFS. But use BFS for some initial steps."""
    cdef int W, H, Z
    cdef int pos

    cdef bytes S = board.state
    cdef bytes G = make_goal(S)

    W = board.w
    H = board.h
    Z = W*H

    cdef int start_step, back_step
    results = []

    back_step, back_routes = limited_bfs(W, H, G, QMAX, (S,))
    for k in back_routes:
        back_routes[k] = reverse_route(back_routes[k])
    if back_step == 0:
        return back_routes.values()
    debug("Back step stopped at", back_step, "steps and", len(back_routes), "states.")

    start_step, fwd_routes = limited_bfs(W, H, S, QMAX, (G,))
    if start_step == 0:
        for k in fwd_routes:
            results.append(fwd_routes[k] + back_routes[k])
        return results
    debug("Forward step stopped at", start_step, "steps and", len(fwd_routes), "states.")


    cdef int depth_limit = min(dist(W, Z, s, G) for s in fwd_routes) - (back_step/2)
    cdef bytes s

    ks = fwd_routes.keys()
    shuffle(ks)

    while not results:
        debug("Iterative DFS: depth limit =", depth_limit)
        
        for s in ks:
            route = fwd_routes[s]
            pos = s.index(b'0')
            depth_limit = slide_dfs(W, Z, back_routes, pos, s, depth_limit, route, results)
        depth_limit += 2

    return results


def solve2(board, int QMAX=300000):
    u"""
    幅優先＋枝刈り。 Iterative Deeping をベースに再実装.
    """
    cdef bytes S = board.state
    cdef bytes G = make_goal(S)

    cdef int W = board.w
    cdef int H = board.h
    cdef int Z = W*H

    cdef int start_step, back_step
    results = []
    srand(int(time()))

    back_step, back_routes = limited_bfs(W, H, G, QMAX, (S,))
    for k in back_routes:
        back_routes[k] = reverse_route(back_routes[k])
    if back_step == 0:
        return back_routes.values()
    debug("Back step stopped at", back_step, "steps and", len(back_routes), "states.")

    cdef int step_limit = 50
    for step_limit in xrange(100, 500, 8):
        debug("Starting forward step with step_limit=", step_limit)
        start_step, fwd_routes = limited_bfs(W, H, S, QMAX, back_routes, 1, step_limit, dist(W, Z, S, G))
        if start_step == 0:
            for k in fwd_routes:
                results.append(fwd_routes[k] + back_routes[k])
            return results


#todo
"""
cdef dist_only_top():
    pass

cdef fill_top_line(int W, int H, bytes state):
    que = deque()
    que.add(('', state))
    cdef int pos = state.find(b'0')

    while True:
        route, state = que.popleft()

def heuristic(board):
    cdef int W = board.w
    cdef int H = board.h
    cdef bytes state = board.state

    cdef bytes prefix_route = ""

    while W+H > 9:
        prefix, state = fill_top_line(state)
        prefix_route += prefix
"""
