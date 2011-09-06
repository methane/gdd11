# coding: utf-8
from __future__ import print_function

cdef extern from *:
    ctypedef char const_char "const char"

from libc.string cimport strchr, strncmp, memset
from libc.stdlib cimport rand, srand, malloc, free
from cpython.bytes cimport PyBytes_FromString

import sys
import string
from time import time
from collections import defaultdict, deque

_base_time = time()

def debug(*args):
    t = time() - _base_time
    print(t, *args, file=sys.stderr)

PLATES = b'123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0'


_T = string.maketrans(b'LRUD', b'RLDU')
def reverse_route(route):
    u"""ルートを逆にする。
    reverse して L<>R, D<>U の置換を行う."""
    route = route[::-1]
    return route.translate(_T)


def make_goal(bytes state):
    N = len(state)
    s = bytearray(state)
    for i in xrange(N):
        if s[i] != ord('='):
            s[i] = PLATES[i]
    s[-1] = '0'
    return bytes(s)

def join_route(route, state, remain):
    for p, s, r in remain:
        if state == s:
            return route + r[::-1]

def join_route_back(broute, state, remain):
    for p, s, r in remain:
        if state == s:
            return r + broute[::-1]


cdef char _dist_table[64][64]

def init_dist_table(int w, int h, bytes s):
    cdef char *tbl
    cdef int pos, dist, npos

    for pos in xrange(w*h):
        tbl = _dist_table[pos]
        memset(tbl, -1, 64)
        tbl[pos] = 0
        q = deque()
        q.append(pos)

        while q:
            pos = q.popleft()
            dist = tbl[pos]+1

            if pos%w:
                npos = pos-1
                if tbl[npos] == -1 and s[npos] != b'=':
                    tbl[npos] = dist
                    q.append(npos)

            if (pos+1)%w:
                npos = pos+1
                if tbl[npos] == -1 and s[npos] != b'=':
                    tbl[npos] = dist
                    q.append(npos)

            if pos-w >= 0:
                npos = pos-w
                if tbl[npos] == -1 and s[npos] != b'=':
                    tbl[npos] = dist
                    q.append(npos)

            if pos+w < w*h:
                npos = pos+w
                if tbl[npos] == -1 and s[npos] != b'=':
                    tbl[npos] = dist
                    q.append(npos)

def get_table_dist(int pos, int to):
    return _dist_table[pos][to]

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
        dist += _dist_table[i][index]
    return dist



def solve_slide(int W, int H, bytes S, int QMAX=200000):
    cdef int dist_limit, dist_limit_b
    cdef bytes state, G, route
    cdef int pos, i

    cdef int Z = W*H

    G = make_goal(S)
    debug("Start:", S)
    debug("Goal: ", G)

    init_dist_table(W, H, S)   

    dist_limit_b = dist_limit = dist(W, Z, G, S) + (W*H)

    Q = deque()
    pos = S.index(b'0')
    Q.append((pos, S, ''))
    visited = set()
    visited.add(bytes(S))
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

        newstate = move(state, from_, to_)

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

    for step in xrange(1,250):
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
    ゴールまでの距離の変化を返す."""

    # 空白が pos->npos なので、タイルは npos->pos に移動する.
    cdef int c = state[npos]
    cdef int gpos = char_to_gpos(c) 

    cdef int prev_dist = _dist_table[gpos][npos]
    cdef int next_dist = _dist_table[gpos][pos]

    return next_dist - prev_dist


DEF VISITED_MAP_SIZE = 256*1024*1024
DEF VISITED_MAP_HASHSIZE = 32 * VISITED_MAP_SIZE

cdef unsigned int *_visited_map

cdef _init_visited_map():
    global _visited_map
    _visited_map = <unsigned int*>malloc(VISITED_MAP_SIZE*sizeof(int))

cdef _del_visited_map():
    global _visited_map
    free(<void*>_visited_map)
    _visited_map=NULL

cdef _reset_visited_map():
    memset(_visited_map, 0, VISITED_MAP_SIZE*sizeof(int))

cdef inline void _set_visited_map(Py_ssize_t pos):
    pos %= VISITED_MAP_HASHSIZE
    _visited_map[pos/32] |= (1 << (pos%32))

cdef inline bint _get_visited_map(Py_ssize_t pos):
    pos %= VISITED_MAP_HASHSIZE
    return _visited_map[pos/32] & (1 << (pos%32))

cdef inline int _count_bits(int bits):
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
    bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
    bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
    return (bits & 0x0000ffff) + (bits >>16 & 0x0000ffff);

cdef long _count_visited_map():
    cdef long bits = 0
    for i in xrange(VISITED_MAP_SIZE):
        bits += _count_bits(_visited_map[i])
    return bits

cdef bfs_trymove(bytes state, int from_, int to_, bytes route, bytes d, q, goals, goal_route,
                 int dist, int dist_limit, int W):
    if state[to_] == b'=':
        return

    cdef int newdist = dist_diff(state, from_, to_, W) + dist
    if newdist*3 > dist_limit:
        return

    newstate = PyBytes_FromString(<char*>state)
    cdef char *p = <char*>newstate
    p[from_], p[to_] = state[to_], state[from_]

    cdef Py_ssize_t hashval = hash(newstate)
    if _get_visited_map(hashval):
        if newstate in goals:
            goal_route[newstate] = (route+d).strip()
        return

    _set_visited_map(hashval)
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

    cdef list Q = [(pos, state, b' ', dist)]

    cdef int step = 0
    goal_routes = {}

    while Q and not goal_routes:
        step += 1

        if len(Q) > threshold:
            if cutdown:
                debug("step=", step, "cutting from", len(Q))
                shuffle(Q, threshold)
                del Q[threshold:]
            else:
                break

        nq = []

        while Q:
            pos, state, route, dist = Q.pop()

            if pos >= W and route[-1] != b'D':
                bfs_trymove(state, pos, pos-W, route, b'U', nq, stop, goal_routes,
                            dist, step_limit-step, W)
            if pos % W and route[-1] != b'R':
                bfs_trymove(state, pos, pos-1, route, b'L', nq, stop, goal_routes,
                            dist, step_limit-step, W)
            if pos+W < Z and route[-1] != b'U':
                bfs_trymove(state, pos, pos+W, route, b'D', nq, stop, goal_routes,
                            dist, step_limit-step, W)
            if (pos+1)%W and route[-1] != b'L':
                bfs_trymove(state, pos, pos+1, route, b'R', nq, stop, goal_routes,
                            dist, step_limit-step, W)
        Q = nq

    if goal_routes:
        return 0, goal_routes

    while Q:
        pos, state, route, dist = Q.pop()
        goal_routes[state] = route.strip()
    return step, goal_routes



cpdef shuffle(list x, int stop=-1):
    """x, random=random.random -> shuffle list x in place; return None.

    Optional arg random is a 0-argument function returning a random
    float in [0.0, 1.0); by default, the standard random.random.
    """
    cdef int i,j
    cdef int N = len(x)
    if stop == -1:
        stop = N
    for i in xrange(stop):
        j = rand() % N
        t = x[i]
        x[i] = x[j]
        x[j] = t


# iterative deeping用のDFS
cdef int slide_dfs(int W, int Z, G, int pos, bytes state,
                   int depth, int depth_limit,
                   bytes route, list answer, int dist):

    if depth_limit-depth < dist:
        return depth_limit

    cdef unsigned long hashval = hash(state)

    if _get_visited_map(hashval):
        # ビットマップが立っている場合は、ゴールか、すでに通ったルートか、
        # それに衝突した場合. 衝突は割り切る.
        if state in G:
            route += G[state]
            debug("Goal:", len(route))
            answer.append(route)
            return -2
        return depth_limit

    # ビットマップを埋めすぎると正当なルートも殺してしまうので、深さで制限する.
    if depth < 60:
        _set_visited_map(hashval)

    cdef char *ps = state
    cdef int npos
    cdef bytes ns

    if pos>=W and route[-1] !=b'D':
        npos = pos-W
        if state[npos] != '=':
            ns = move(state, pos, npos)
            depth_limit = slide_dfs(W,Z,G, npos, ns, depth+1, depth_limit, route+b'U',
                                    answer, dist+dist_diff(ps, pos, npos, W))

    if pos%W and route[-1] !=b'R':
        npos = pos-1
        if state[npos] != '=':
            ns = move(state, pos, npos)
            depth_limit = slide_dfs(W,Z,G, npos, ns, depth+1, depth_limit, route+b'L',
                                    answer, dist+dist_diff(ps, pos, npos, W))

    if pos+W<Z and route[-1] !=b'U':
        npos = pos+W
        if state[npos] != '=':
            ns = move(state, pos, npos)
            depth_limit = slide_dfs(W,Z,G, npos, ns, depth+1, depth_limit, route+b'D',
                                    answer, dist+dist_diff(ps, pos, npos, W))

    if (pos+1)%W and route[-1] !=b'L':
        npos = pos+1
        if state[npos] != '=':
            ns = move(state, pos, npos)
            depth_limit = slide_dfs(W,Z,G, npos, ns, depth+1, depth_limit, route+b'R',
                                    answer, dist+dist_diff(ps, pos, npos, W))

    return depth_limit


def iterative_deeping(int W, int H, bytes S, int QMAX=400000):
    """Iterative deeping DFS. But use BFS for some initial steps."""
    cdef int Z = W*H
    cdef bytes G = make_goal(S)
    _init_visited_map()

    init_dist_table(W, H, S)   

    cdef int start_step, back_step

    back_step, back_routes = limited_bfs(W, H, G, QMAX, (S,))
    for k in back_routes:
        back_routes[k] = reverse_route(back_routes[k])
    if back_step == 0:
        _del_visited_map()
        return back_routes.values()
    debug("Back step stopped at", back_step, "steps and", len(back_routes), "states.")


    cdef int pos = S.index(b'0')
    cdef int depth_limit = dist(W, Z, S, G)

    results = []
    while depth_limit < 300:
        debug("Iterative DFS: depth limit =", depth_limit)
        _reset_visited_map()
        # 辞書探索を減らすために、ゴールもビットマップに入れておく.
        for s in back_routes:
            _set_visited_map(hash(s))
        slide_dfs(W, Z, back_routes, pos, S, 0, depth_limit, b' ', results,
                  dist(W, Z, S, G) - max(dist(W, Z, s, G) for s in back_routes))
        debug("Visit BMP filled", _count_visited_map(), "/", VISITED_MAP_HASHSIZE)
        if results:
            break
        depth_limit += 16
    else:
        debug("Give up...")

    _del_visited_map()
    return [s.strip() for s in results]


def solve2(int W, int H, bytes S, int QMAX=400000, max_depth=200):
    u"""
    幅優先＋枝刈り。 Iterative Deeping をベースに再実装.
    """
    cdef bytes G = make_goal(S)
    cdef int Z = W*H

    if _visited_map == NULL:
        _init_visited_map()
    _reset_visited_map()

    init_dist_table(W, H, S)   

    cdef int start_step, back_step
    results = []
    srand(int(time()))

    back_step, back_routes = limited_bfs(W, H, G, QMAX*2, (S,))
    for k in back_routes:
        back_routes[k] = reverse_route(back_routes[k])
    if back_step == 0:
        return back_routes.values()
    debug("Back step stopped at", back_step, "steps and", len(back_routes), "states.")

    # Python のハッシュテーブルのコストを減らすために、ゴールへの到達判定も
    # ビットマップを使う。
    # 余計な部分まで枝刈りしないように、先端だけ埋めておく.
    _reset_visited_map()
    for k in back_routes:
        _set_visited_map(hash(k))

    debug("Starting forward step with step_limit=", max_depth)
    start_step, fwd_routes = limited_bfs(W, H, S, QMAX, back_routes, 1, max_depth, dist(W, Z, S, G))
    if start_step == 0:
        for k in fwd_routes:
            results.append(fwd_routes[k] + back_routes[k])
        return results
    return []

cdef enum DIRECT:
    NOT = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


cdef object brute_dfs(char *S, char *G, int pos, int limit, int fit, int W, int Z, int last=NOT):
    if strncmp(S, G, fit) == 0:
        return (S, b'')
    if limit <= 0:
        return None
    limit -= 1

    cdef int npos

    if pos >= W and last != DOWN:
        npos = pos - W
        if S[npos] != b'=':
            ns = move(S, pos, npos)
            r = brute_dfs(ns, G, npos, limit, fit, W, Z, UP)
            if r is not None: return (r[0], b'U'+r[1])

    if pos % W and last != RIGHT:
        npos = pos - 1
        if S[npos] != b'=':
            ns = move(S, pos, npos)
            r = brute_dfs(ns, G, npos, limit, fit, W, Z, LEFT)
            if r is not None: return (r[0], b'L'+r[1])

    if pos+W < Z and last != UP:
        npos = pos + W
        if S[npos] != b'=':
            ns = move(S, pos, npos)
            r = brute_dfs(ns, G, npos, limit, fit, W, Z, DOWN)
            if r is not None: return (r[0], b'D'+r[1])

    if (pos+1) % W and last != LEFT:
        npos = pos + 1
        if S[npos] != b'=':
            ns = move(S, pos, npos)
            r = brute_dfs(ns, G, npos, limit, fit, W, Z, RIGHT)
            if r is not None: return (r[0], b'R'+r[1])




def solve_brute_force(int W, int H, bytes S, int Q=0):
    u"""
    1つずつ着実にアルゴリズム
    """
    cdef bytes G = make_goal(S)

    cdef int Z = W*H
    cdef int i,j
 
    init_dist_table(W, H, S)   

    routes = []

    for i in xrange(Z):
        if S[:i] == G[:i]:
            continue
        debug("Fiting first", i, "chars")
        for j in xrange(60):
            debug("IDDFS step:", j)
            r = brute_dfs(S, G, S.find(b'0'), j, i, W, Z)
            if r is not None:
                S = r[0]
                routes.append(r[1])
                break
        else:
            return []
    return [b''.join(routes)]


cdef find_shortest_route(int W, int H, bytes state, int start, int goal):
    cdef int pos = start
    cdef int Z = W*H
    route = [pos]
    q = [route]

    while True:
        nq = []
        while q:
            route = q.pop()
            pos = route[-1]
            if pos == goal: return route

            if pos%W and state[pos-1] != b'=':
                nq.append(route + [pos-1])
            if (pos+1)%W and state[pos+1] != b'=':
                nq.append(route + [pos+1])
            if pos>=W and state[pos-W] != b'=':
                nq.append(route + [pos-W])
            if pos+W<Z and state[pos+W] != b'=':
                nq.append(route + [pos+W])
        q = nq


cdef pos_to_direct(int pos, int npos):
    if npos == pos+1:
        return b'R'
    elif npos == pos-1:
        return b'L'
    elif npos > pos: 
        return b'D'
    else:
        return b'U'

cdef move_one_panel_to_goal(int W, int H, bytes state, bytes panel):
    cdef int start = state.index(panel)
    cdef int goal = PLATES.index(panel)
    cdef int pos = state.index(b'0')
    cdef int panel_pos
    cdef int npos

    dummy = state[:start] + b'=' + state[start+1:]

    panel_route = find_shortest_route(W, H, state, start, goal)
    total_route = b""
    for i in xrange(len(panel_route)-1):
        route = find_shortest_route(W, H, dummy, pos, panel_route[i+1])
        for npos in route[1:]:
            dummy = move(dummy, pos, npos)
            state = move(state, pos, npos)
            total_route += pos_to_direct(pos, npos)
            pos = npos
        npos = panel_route[i]
        dummy = move(dummy, pos, npos)
        state = move(state, pos, npos)
        total_route += pos_to_direct(pos, npos)
    return state, total_route


def solve_brute_force2(int W, int H, bytes S, int Q=0):
    u"""
    もっと着実アルゴリズム
    """
    cdef bytes G = make_goal(S)
    cdef bytes state = S

    cdef int Z = W*H
    cdef int i,j

    init_dist_table(W, H, S)

    routes = []
    cdef int depth=20

    start_time = time()

    i=1
    while i < Z:
        if S[:i] == G[:i]:
            i += 1
            continue
        debug(time()-start_time, "Fiting first", i, "chars")
        for j in xrange(depth):
            r = brute_dfs(S, G, S.find(b'0'), j, i, W, Z)
            if r is not None:
                S = r[0]
                routes.append(r[1])
                break
            else:
                if time()-start_time > 300:
                    return []
        else:
            while i < Z-1:
                if S[i] != G[i]:
                    debug(time()-start_time, "Moving", G[i])
                    S,r = move_one_panel_to_goal(W, H, S, G[i])
                    routes.append(r)
                i += 1
            i = 1
            depth += 3
            continue
        i += 1
    return [b''.join(routes)]


def solve_combined(int W, int H, bytes S, int Q=300000):
    u"""複数の方式の組み合わせ."""

    cdef bytes G = make_goal(S)
    cdef bytes state = S
    cdef int Z = W*H

    init_dist_table(W, H, S)

    base_route = b''

    for i in xrange(Z-1):
        if S[i] != G[i]:
            S,r = move_one_panel_to_goal(W, H, S, G[i])
            base_route += r

    #routes = solve2(W, H, S)
    routes = iterative_deeping(W, H, S)
    routes = [base_route + r for r in routes]
    return routes

