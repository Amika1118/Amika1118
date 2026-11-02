#!/usr/bin/env python3
"""AI Arcade - tabular Q-learning agents that train on a GitHub contribution graph.

For every game the script trains an agent from scratch, snapshots its policy at
several points of training and renders one animated SVG that plays those stages
back to back (untrained -> learning -> trained). Pure standard library.

    python3 ai_arcade.py --user Amika1118 --out ai_dist     # needs GITHUB_TOKEN
    python3 ai_arcade.py --demo --out ai_dist               # synthetic graph, no network

Outputs ai-<game>.svg (light) and ai-<game>-dark.svg (dark) per game.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request
from collections import deque

# ---------------------------------------------------------------- layout --
CELL, PITCH = 10, 12          # cell size / pitch in px
MX, TOP, BOT = 10, 30, 30     # margins
DT = 0.08                     # seconds per animation frame
INTER, HOLD = 14, 12          # frames of "training..." pause / end-of-clip hold
MAX_PLAY = 170                # max frames of gameplay shown per clip

DARK = dict(empty="#161b22", lv=["#3b3380", "#5145A0", "#7566d9", "#a78bfa"],
            text="#c9d1d9", muted="#8b949e", ball="#f0f6fc", accent="#8B7CFF", line="#30363d")
LIGHT = dict(empty="#ebedf0", lv=["#d4cdfb", "#a99bf5", "#7c68e6", "#5145A0"],
             text="#24292f", muted="#57606a", ball="#24292f", accent="#5145A0", line="#d0d7de")


# ------------------------------------------------------------------ data --
GQL = ("query($login:String!){user(login:$login){contributionsCollection{contributionCalendar"
       "{weeks{contributionDays{contributionCount weekday}}}}}}")


def fetch_counts(login, token):
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": GQL, "variables": {"login": login}}).encode(),
        headers={"Authorization": "bearer " + token, "Content-Type": "application/json",
                 "User-Agent": "ai-arcade"})
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.load(r)
    if "errors" in data or not data.get("data", {}).get("user"):
        raise RuntimeError("GraphQL error: %s" % json.dumps(data)[:300])
    weeks = data["data"]["user"]["contributionsCollection"]["contributionCalendar"]["weeks"]
    grid = [[0] * len(weeks) for _ in range(7)]
    for c, w in enumerate(weeks):
        for d in w["contributionDays"]:
            grid[d["weekday"]][c] = d["contributionCount"]
    return grid


def demo_counts(seed=7, weeks=53):
    rng = random.Random(seed)
    return [[rng.choice([0, 0, 0, 1, 1, 2, 3, 5, 8]) for _ in range(weeks)] for _ in range(7)]


def to_levels(grid):
    nz = sorted(v for row in grid for v in row if v > 0)
    if not nz:
        return [[0] * len(grid[0]) for _ in grid]
    q1, q2, q3 = nz[len(nz) // 4], nz[len(nz) // 2], nz[(3 * len(nz)) // 4]
    return [[0 if v == 0 else 1 + (v > q1) + (v > q2) + (v > q3) for v in row] for row in grid]


# ------------------------------------------------------------- Q-learning --
class QAgent:
    def __init__(self, n, alpha, gamma):
        self.q, self.n, self.alpha, self.gamma = {}, n, alpha, gamma

    def qs(self, s):
        v = self.q.get(s)
        if v is None:
            v = self.q[s] = [0.0] * self.n
        return v

    def act(self, s, eps, rng):
        if rng.random() < eps:
            return rng.randrange(self.n)
        v = self.qs(s)
        m = max(v)
        best = [i for i, x in enumerate(v) if x == m]
        return best[0] if len(best) == 1 else rng.choice(best)

    def learn(self, s, a, r, s2, done):
        v = self.qs(s)
        target = r if done else r + self.gamma * max(self.qs(s2))
        v[a] += self.alpha * (target - v[a])

    def snapshot(self):
        return {k: list(v) for k, v in self.q.items()}


def make_policy(snap, n, rng, eps):
    def pol(s):
        if rng.random() < eps:
            return rng.randrange(n)
        v = snap.get(s)
        if v is None:
            return rng.randrange(n)
        m = max(v)
        best = [i for i, x in enumerate(v) if x == m]
        return best[0] if len(best) == 1 else rng.choice(best)
    return pol


def sign(x):
    return (x > 0) - (x < 0)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# ------------------------------------------------------------------ games --
class Game:
    """Interface: reset(rng) / state() / step(a)->(reward, done) / positions() / result()."""
    key = title = ""
    n_actions = 4
    alpha, gamma = 0.2, 0.95
    eps_start, eps_end = 1.0, 0.05
    episodes, cap_train = 2000, 250
    rows = 7
    score_word = "SCORE"

    def __init__(self, levels):
        self.lv = levels
        self.W = len(levels[0])
        self.food0 = [(r, c) for r in range(7) for c in range(self.W) if levels[r][c] > 0]
        self.removed = []

    def solid_cells(self):
        return [(r, c, self.lv[r][c]) for (r, c) in self.food0]

    def extra_static(self, th, anim_for=None):
        return ""

    def mover_svgs(self, th):
        return []

    def rank(self):
        """Higher is better; used to pick the best rollout and to draw the learning curve."""
        return self.score

    def result(self):
        return "%s %d" % (self.score_word, self.score)


class PacMan(Game):
    key, title = "pacman", "PAC-MAN"
    DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    episodes, cap_train = 2500, 220

    def reset(self, rng):
        self.rng = rng
        self.px, self.py = 0, 3
        self.food = set(self.food0)
        self.ghosts = [[self.W - 1, 1], [self.W - 1, 5]]
        self.score, self.dead = 0, False
        self.removed = []
        self._eat()

    def _eat(self):
        if (self.py, self.px) in self.food:
            self.food.discard((self.py, self.px))
            self.score += 1
            self.removed.append(("c_%d_%d" % (self.py, self.px), 0))
            return 1
        return 0

    def state(self):
        px, py = self.px, self.py
        if self.food:
            fr, fc = min(self.food, key=lambda f: abs(f[0] - py) + abs(f[1] - px))
            d = abs(fr - py) + abs(fc - px)
            fs = (sign(fc - px), sign(fr - py), 0 if d <= 1 else 1 if d <= 3 else 2)
        else:
            fs = (0, 0, 2)
        best, bd = None, 99
        for gx, gy in self.ghosts:
            rx, ry = gx - px, gy - py
            d = max(abs(rx), abs(ry))
            if d <= 3 and d < bd:
                best, bd = (rx, ry), d
        return fs + (best,)

    def _hit(self):
        return any(g[0] == self.px and g[1] == self.py for g in self.ghosts)

    def step(self, a):
        dx, dy = self.DIRS[a]
        nx, ny = self.px + dx, self.py + dy
        if 0 <= nx < self.W and 0 <= ny < 7:
            self.px, self.py = nx, ny
        reward = -0.02 + self._eat()
        if self._hit():
            self.dead = True
            return reward - 10, True
        rng = self.rng
        for g in self.ghosts:
            if rng.random() < 0.65:
                ddx, ddy = self.px - g[0], self.py - g[1]
                if rng.random() < 0.75:
                    if abs(ddx) > abs(ddy) or (abs(ddx) == abs(ddy) and rng.random() < 0.5):
                        g[0] += sign(ddx)
                    else:
                        g[1] += sign(ddy)
                else:
                    mx, my = self.DIRS[rng.randrange(4)]
                    if 0 <= g[0] + mx < self.W and 0 <= g[1] + my < 7:
                        g[0] += mx
                        g[1] += my
        if self._hit():
            self.dead = True
            return reward - 10, True
        if not self.food:
            return reward + 5, True
        return reward, False

    def positions(self):
        return [(self.px, self.py)] + [(g[0], g[1]) for g in self.ghosts]

    def result(self):
        return "%s %d%s" % (self.score_word, self.score, " (CAUGHT)" if self.dead else "")

    def mover_svgs(self, th):
        pac = ('<path d="M5 5 L10.02 2.1 A5.8 5.8 0 1 0 10.02 7.9 Z" fill="#FFD21E"/>')
        def ghost(col):
            return ('<path d="M0 10 V5 A5 5 0 0 1 10 5 V10 L8.3 8.3 L6.7 10 L5 8.3 L3.3 10 L1.7 8.3 Z" fill="%s"/>'
                    '<circle cx="3.5" cy="4.5" r="1.4" fill="#fff"/><circle cx="6.5" cy="4.5" r="1.4" fill="#fff"/>'
                    '<circle cx="3.9" cy="4.7" r=".7" fill="#1a1a2e"/><circle cx="6.9" cy="4.7" r=".7" fill="#1a1a2e"/>' % col)
        return [pac, ghost("#ff4d4d"), ghost("#ff9ecf")]


class Breakout(Game):
    key, title = "breakout", "BREAKOUT"
    n_actions = 3
    rows, PR = 16, 15
    episodes, cap_train = 4000, 400
    score_word = "BRICKS"
    alpha, gamma = 0.2, 0.9
    HALF, SPEED = 3, 2          # paddle half-width (cells) and speed (cells per step)
    eps_start, eps_end = 0.5, 0.02

    def reset(self, rng):
        self.rng = rng
        self.px = self.W // 2
        self.bx, self.by = self.px, self.PR - 1
        self.vx, self.vy = rng.choice([-1, 1]), -1
        self.bricks = set(self.food0)
        self.score, self.lost = 0, False
        self.removed = []

    def _target(self):
        """Where the ball will reach the paddle row (walls reflected, bricks ignored)."""
        if self.vy < 0:
            return self.bx
        x = self.bx + self.vx * (self.PR - self.by)
        period = 2 * (self.W - 1)
        x %= period
        return period - x if x > self.W - 1 else x

    def state(self):
        zone = 0 if self.by < 8 else 1 if self.by < 13 else 2
        return (clamp(self._target() - self.px, -6, 6), 1 if self.vy > 0 else 0, zone)

    def step(self, a):
        d_before = min(abs(self._target() - self.px), 10)
        self.px = clamp(self.px + (a - 1) * self.SPEED, self.HALF, self.W - 1 - self.HALF)
        reward = -0.01
        nx, ny = self.bx + self.vx, self.by + self.vy
        if nx < 0 or nx >= self.W:
            self.vx = -self.vx
            nx = self.bx + self.vx
        if ny < 0:
            self.vy = 1
            ny = self.by + 1
        if 0 <= ny < 7 and (ny, nx) in self.bricks:
            self.bricks.discard((ny, nx))
            self.removed.append(("c_%d_%d" % (ny, nx), 0))
            self.score += 1
            reward += 1
            self.vy = -self.vy
            ny = self.by
        done = False
        if ny >= self.PR:
            if abs(nx - self.px) <= self.HALF:
                self.vy, ny = -1, self.PR - 1
                reward += 0.5
                if nx < self.px:
                    self.vx = -1
                elif nx > self.px:
                    self.vx = 1
            else:
                reward -= 3
                done, self.lost = True, True
                ny = self.PR
        self.bx, self.by = nx, ny
        reward += 0.15 * (d_before - min(abs(self._target() - self.px), 10))   # shaping: close in on the landing spot
        if not self.bricks:
            return reward + 5, True
        return reward, done

    def positions(self):
        return [(self.px - self.HALF, self.PR), (self.bx, self.by)]

    def result(self):
        return "%s %d%s" % (self.score_word, self.score, " (BALL LOST)" if self.lost else "")

    def extra_static(self, th, anim_for=None):
        return '<rect x="0" y="%d" width="%d" height="1" fill="%s"/>' % (
            self.PR * PITCH + 11, self.W * PITCH - 2, th["line"])

    def mover_svgs(self, th):
        return ['<rect x="0" y="3" width="%d" height="6" rx="3" fill="%s"/>' % ((2 * self.HALF + 1) * PITCH - 2, th["accent"]),
                '<circle cx="5" cy="5" r="3.4" fill="%s"/>' % th["ball"]]


class Invaders(Game):
    key, title = "invaders", "SPACE INVADERS"
    n_actions = 3
    rows, SR, BSPEED = 16, 15, 3
    episodes, cap_train = 1800, 300
    score_word = "HITS"
    alpha, gamma = 0.2, 0.9

    def reset(self, rng):
        self.rng = rng
        self.sx, self.bullet, self.bombs = self.W // 2, None, []
        self.aliens = set(self.food0)
        self.cols = {}
        for r, c in self.aliens:
            self.cols[c] = self.cols.get(c, 0) + 1
        self.score, self.dead = 0, False
        self.removed = []

    def _nearest(self):
        return min(self.cols, key=lambda k: abs(k - self.sx)) if self.cols else self.sx

    def state(self):
        threat = None
        for bc, br in self.bombs:
            dx, dr = bc - self.sx, self.SR - br
            if abs(dx) <= 2 and 0 < dr <= 6 and (threat is None or dr < threat[1]):
                threat = (dx, dr)
        return (clamp(self._nearest() - self.sx, -8, 8), 1 if self.sx in self.cols else 0,
                1 if self.bullet else 0, threat)

    def step(self, a):
        reward = -0.02
        before = abs(self._nearest() - self.sx)
        if a == 0:
            self.sx = max(0, self.sx - 1)
        elif a == 1:
            self.sx = min(self.W - 1, self.sx + 1)
        elif self.bullet is None:
            self.bullet = [self.sx, self.SR]
        if self.bullet:
            bc = self.bullet[0]
            for _ in range(self.BSPEED):
                self.bullet[1] -= 1
                row = self.bullet[1]
                if row < 0:
                    self.bullet = None
                    reward -= 0.3
                    break
                if (row, bc) in self.aliens:
                    self.aliens.discard((row, bc))
                    self.cols[bc] -= 1
                    if not self.cols[bc]:
                        del self.cols[bc]
                    self.removed.append(("c_%d_%d" % (row, bc), 0))
                    self.score += 1
                    reward += 1
                    self.bullet = None
                    break
        # enemy bombs fall one row per step
        for b in self.bombs:
            b[1] += 1
        if any(b[1] == self.SR and b[0] == self.sx for b in self.bombs):
            self.dead = True
            return reward - 10, True
        self.bombs = [b for b in self.bombs if b[1] < self.SR]
        if len(self.bombs) < 3 and self.cols and self.rng.random() < 0.3:
            self.bombs.append([self.rng.choice(list(self.cols)), 7])
        if not self.aliens:
            return reward + 5, True
        if a != 2:
            reward += 0.05 * (before - abs(self._nearest() - self.sx))
        return reward, False

    def positions(self):
        b = (self.bullet[0], self.bullet[1]) if self.bullet else (-5, -5)
        bombs = [(x, y) for x, y in self.bombs] + [(-5, -5)] * 3
        return [(self.sx, self.SR), b, bombs[0], bombs[1], bombs[2]]

    def result(self):
        return "%s %d%s" % (self.score_word, self.score, " (SHIP DESTROYED)" if self.dead else "")

    def extra_static(self, th, anim_for=None):
        return '<rect x="0" y="%d" width="%d" height="1" fill="%s"/>' % (
            self.SR * PITCH + 11, self.W * PITCH - 2, th["line"])

    def mover_svgs(self, th):
        bomb = '<circle cx="5" cy="5" r="3.2" fill="#ff4d4d"/><circle cx="5" cy="5" r="1.2" fill="#ffd2d2"/>'
        return ['<polygon points="5,0 10,10 0,10" fill="%s"/>' % th["accent"],
                '<rect x="4" y="0" width="2" height="8" rx="1" fill="#FFD21E"/>', bomb, bomb, bomb]


class Maze(Game):
    key, title = "maze", "MAZE RUNNER"
    episodes, cap_train = 900, 300
    alpha, gamma = 0.3, 0.99
    eps_start, eps_end = 0.3, 0.02
    DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    score_word = "STEPS"

    def __init__(self, levels):
        Game.__init__(self, levels)
        self.start, self.goal = (3, 0), (3, self.W - 1)
        self.walls, self.thr, self.optimal = self._build()

    def _bfs(self, walls):
        dist = {self.start: 0}
        q = deque([self.start])
        while q:
            r, c = q.popleft()
            if (r, c) == self.goal:
                return dist[(r, c)]
            for dx, dy in self.DIRS:
                n = (r + dy, c + dx)
                if 0 <= n[0] < 7 and 0 <= n[1] < self.W and n not in walls and n not in dist:
                    dist[n] = dist[(r, c)] + 1
                    q.append(n)
        return None

    def _build(self):
        for thr in (1, 2, 3, 4, 5):
            walls = {(r, c) for (r, c) in self.food0 if self.lv[r][c] >= thr}
            walls.discard(self.start)
            walls.discard(self.goal)
            d = self._bfs(walls)
            if d is not None:
                return walls, thr, d
        return set(), 5, self._bfs(set())

    def solid_cells(self):
        return [(r, c, self.lv[r][c]) for (r, c) in sorted(self.walls)]

    def trail_ids(self):
        return ["t_%d_%d" % (r, c) for r in range(7) for c in range(self.W) if (r, c) not in self.walls]

    def reset(self, rng):
        self.rng = rng
        self.r, self.c = self.start
        self.steps, self.reached = 0, False
        self.seen = {self.start}
        self.removed = [("t_%d_%d" % self.start, 1)]
        self.score = 0

    def state(self):
        return (self.r, self.c)

    def step(self, a):
        dx, dy = self.DIRS[a]
        nr, nc = self.r + dy, self.c + dx
        reward = -0.05
        if 0 <= nr < 7 and 0 <= nc < self.W and (nr, nc) not in self.walls:
            self.r, self.c = nr, nc
        else:
            reward -= 0.1
        self.steps += 1
        self.score = self.steps
        if (self.r, self.c) not in self.seen:
            self.seen.add((self.r, self.c))
            self.removed.append(("t_%d_%d" % (self.r, self.c), 1))
        if (self.r, self.c) == self.goal:
            self.reached = True
            return reward + 10, True
        return reward, False

    def positions(self):
        return [(self.c, self.r)]

    def rank(self):
        return 1000 - self.steps if self.reached else 0

    def result(self):
        if self.reached:
            return "GOAL IN %d STEPS (BEST %d)" % (self.steps, self.optimal)
        return "LOST AFTER %d STEPS" % self.steps

    def extra_static(self, th, anim_for=None):
        gx, gy = self.goal[1] * PITCH + 5, self.goal[0] * PITCH + 5
        sx, sy = self.start[1] * PITCH + 5, self.start[0] * PITCH + 5
        out = ['<circle cx="%d" cy="%d" r="4.6" fill="#FFD21E"/>' % (gx, gy),
               '<circle cx="%d" cy="%d" r="4.6" fill="none" stroke="#34d399" stroke-width="1.4"/>' % (sx, sy)]
        for tid in self.trail_ids():
            _, r, c = tid.split("_")
            out.append('<circle cx="%d" cy="%d" r="2.2" fill="#34d399" opacity="0">%s</circle>' % (
                int(c) * PITCH + 5, int(r) * PITCH + 5, anim_for(tid) if anim_for else ""))
        return "".join(out)

    def mover_svgs(self, th):
        return ['<circle cx="5" cy="5" r="4.2" fill="#34d399"/><circle cx="5" cy="5" r="1.6" fill="%s"/>' % th["ball"]]


GAMES = [PacMan, Breakout, Invaders, Maze]


# --------------------------------------------------------------- training --
def train(env, rng):
    n = env.episodes
    ckpts = sorted({0, max(1, int(0.04 * n)), int(0.2 * n), n})
    agent = QAgent(env.n_actions, env.alpha, env.gamma)
    snaps, scores = {}, []
    for ep in range(n + 1):
        if ep in ckpts:
            snaps[ep] = agent.snapshot()
        if ep == n:
            break
        eps = env.eps_start + (env.eps_end - env.eps_start) * ep / max(1, n - 1)
        agent.alpha = env.alpha * (1 - 0.8 * ep / n)     # decaying learning rate keeps late training stable
        env.reset(rng)
        s = env.state()
        for _ in range(env.cap_train):
            a = agent.act(s, eps, rng)
            r, done = env.step(a)
            env.removed.clear()
            s2 = env.state()
            agent.learn(s, a, r, s2, done)
            s = s2
            if done:
                break
        scores.append(env.rank())
    return ckpts, snaps, scores


def rollout(env, snap, rng, eps):
    pol = make_policy(snap, env.n_actions, rng, eps)
    env.reset(rng)
    frames, events = [env.positions()], [(0, i, v) for i, v in env.removed]
    env.removed.clear()
    s = env.state()
    for _ in range(MAX_PLAY - 1):
        r, done = env.step(pol(s))
        frames.append(env.positions())
        events += [(len(frames) - 1, i, v) for i, v in env.removed]
        env.removed.clear()
        if done:
            break
        s = env.state()
    return dict(frames=frames, events=events, score=env.score, rank=env.rank(), result=env.result())


def build_clips(env, rng):
    t0 = time.time()
    ckpts, snaps, scores = train(env, rng)
    clips = []
    for j, e in enumerate(ckpts):
        eps, tries = (1.0, 1) if e == 0 else (0.03, 3)
        best = None
        for _ in range(tries):
            c = rollout(env, snaps[e], rng, eps)
            if best is None or c["rank"] > best["rank"] or (
                    c["rank"] == best["rank"] and len(c["frames"]) < len(best["frames"])):
                best = c
        best["episodes"], best["stage"] = e, j + 1
        best["best_of"] = tries
        clips.append(best)
    print("  %-9s trained %d episodes in %.1fs | stage scores: %s" % (
        env.key, env.episodes, time.time() - t0, [c["score"] for c in clips]))
    return clips, scores


# -------------------------------------------------------------- rendering --
def fmt(x):
    s = ("%.1f" % x).rstrip("0").rstrip(".")
    return s if s else "0"


def anim_vis(intervals, F):
    """opacity animation: visible during the given [start, end) frame intervals."""
    kt, vals = [], []
    first = intervals[0][0] == 0
    kt.append(0)
    vals.append(1 if first else 0)
    for s, e in intervals:
        if s != 0:
            kt.append(s)
            vals.append(1)
        if e < F:
            kt.append(e)
            vals.append(0)
    keys = ";".join("%.5f" % (k / F) for k in kt)
    return ('<animate attributeName="opacity" calcMode="discrete" values="%s" keyTimes="%s" '
            'dur="%.2fs" repeatCount="indefinite"/>' % (";".join(map(str, vals)), keys, F * DT))


def curve_points(scores, w=130, h=16, bins=40):
    if not scores:
        return ""
    size = max(1, len(scores) // bins)
    avg = [sum(scores[i:i + size]) / len(scores[i:i + size]) for i in range(0, len(scores), size)]
    hi = max(avg) or 1
    n = max(1, len(avg) - 1)
    return " ".join("%s,%s" % (fmt(i * w / n), fmt(h - (v / hi) * h)) for i, v in enumerate(avg))


def render(env, clips, scores, th, dark):
    W = env.W
    width = MX * 2 + W * PITCH - 2
    height = TOP + env.rows * PITCH + BOT
    # ---- global timeline
    timeline, starts, F = [], [], 0
    n_movers = len(clips[0]["frames"][0])
    tracks = [[] for _ in range(n_movers)]
    for c in clips:
        starts.append(F)
        seq = [c["frames"][0]] * INTER + c["frames"] + [c["frames"][-1]] * HOLD
        for fr in seq:
            for m in range(n_movers):
                tracks[m].append(fr[m])
        c["play_len"] = len(c["frames"]) + HOLD
        F += len(seq)
    kt_all = ";".join("%.5f" % (i / F) for i in range(F))
    dur = "%.2fs" % (F * DT)

    # ---- cell events: id -> [(frame, opacity)]
    ev = {}
    for c in clips:
        for _, i, _v in c["events"]:
            ev[i] = []
    for c, cs in zip(clips, starts):
        for i in ev:
            ev[i].append((cs, 0 if i.startswith("t_") else 1))
        for k, i, v in c["events"]:
            ev[i].append((cs + INTER + k, v))

    def anim_for(i):
        if i not in ev:
            return ""
        ks = sorted({f: v for f, v in ev[i]}.items())
        keys = ";".join("%.5f" % (f / F) for f, _ in ks)
        vals = ";".join(str(v) for _, v in ks)
        return ('<animate attributeName="opacity" calcMode="discrete" values="%s" keyTimes="%s" '
                'dur="%s" repeatCount="indefinite"/>' % (vals, keys, dur))

    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
           'font-family="Menlo,Consolas,\'DejaVu Sans Mono\',monospace" role="img" '
           'aria-label="AI agent learning to play %s on my GitHub contribution graph">' % (
               width, height, width, height, env.title.title())]
    out.append("<title>%s - AI agent learning on my contribution graph</title>" % env.title)
    out.append('<text x="%d" y="18" font-size="11" font-weight="bold" fill="%s" letter-spacing="1">'
               'AI ARCADE | %s</text>' % (MX, th["accent"], env.title))
    out.append('<text x="%d" y="18" font-size="9" fill="%s" text-anchor="end">'
               'Q-LEARNING | RETRAINED FROM SCRATCH ON EVERY RUN</text>' % (width - MX, th["muted"]))
    out.append('<g transform="translate(%d,%d)">' % (MX, TOP))
    # empty grid
    for r in range(7):
        for c in range(W):
            out.append('<rect x="%d" y="%d" width="%d" height="%d" rx="2" fill="%s"/>' % (
                c * PITCH, r * PITCH, CELL, CELL, th["empty"]))
    # solid (animated) cells
    for r, c, lvl in env.solid_cells():
        cid = "c_%d_%d" % (r, c)
        col = th["lv"][lvl - 1]
        base = '<rect id="%s" x="%d" y="%d" width="%d" height="%d" rx="2" fill="%s"' % (
            cid, c * PITCH, r * PITCH, CELL, CELL, col)
        a = anim_for(cid)
        out.append(base + ("/>" if not a else ">" + a + "</rect>"))
    out.append(env.extra_static(th, anim_for))
    # movers
    for m, svg in enumerate(env.mover_svgs(th)):
        pts = tracks[m]
        vals = ";".join("%s,%s" % (fmt(x * PITCH), fmt(y * PITCH)) for x, y in pts)
        out.append('<g transform="translate(%s,%s)">%s<animateTransform attributeName="transform" type="translate" '
                   'calcMode="discrete" values="%s" keyTimes="%s" dur="%s" repeatCount="indefinite"/></g>' % (
                       fmt(pts[0][0] * PITCH), fmt(pts[0][1] * PITCH), svg, vals, kt_all, dur))
    out.append("</g>")
    # status texts
    sy = height - 11
    for c, cs in zip(clips, starts):
        t_train = "TRAINING... %s EPISODES SO FAR" % format(c["episodes"], ",")
        if c["episodes"] == 0:
            t_train = "ATTEMPT 1 | UNTRAINED - PLAYING RANDOMLY"
        out.append('<text x="%d" y="%d" font-size="10" fill="%s" opacity="0">%s%s</text>' % (
            MX, sy, th["muted"], t_train, anim_vis([(cs, cs + INTER)], F)))
        t_play = "ATTEMPT %d/%d | %s EPISODES | %s" % (c["stage"], len(clips), format(c["episodes"], ","), c["result"])
        if c["best_of"] > 1:
            t_play += " | BEST OF %d" % c["best_of"]
        out.append('<text x="%d" y="%d" font-size="10" fill="%s" opacity="0">%s%s</text>' % (
            MX, sy, th["text"], t_play, anim_vis([(cs + INTER, cs + INTER + c["play_len"])], F)))
    # learning curve
    cw, ch = 130, 16
    cx, cy = width - MX - cw, height - 9 - ch
    out.append('<text x="%d" y="%d" font-size="8" fill="%s" text-anchor="end">LEARNING CURVE</text>' % (
        cx - 6, height - 11, th["muted"]))
    out.append('<polyline transform="translate(%d,%d)" points="%s" fill="none" stroke="%s" stroke-width="1.6" '
               'stroke-linejoin="round"/>' % (cx, cy, curve_points(scores, cw, ch), th["accent"]))
    out.append("</svg>")
    return "\n".join(out)


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=os.environ.get("GITHUB_REPOSITORY_OWNER", ""))
    ap.add_argument("--out", default="ai_dist")
    ap.add_argument("--demo", action="store_true", help="use a synthetic contribution graph")
    ap.add_argument("--seed", type=int, default=int(time.time() // 3600))
    ap.add_argument("--games", default=",".join(g.key for g in GAMES))
    args = ap.parse_args()

    if args.demo:
        counts = demo_counts()
    else:
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token or not args.user:
            sys.exit("Need --user and GITHUB_TOKEN (or use --demo)")
        counts = fetch_counts(args.user, token)
    levels = to_levels(counts)
    os.makedirs(args.out, exist_ok=True)
    wanted = set(args.games.split(","))
    print("seed=%d grid=7x%d active days=%d" % (args.seed, len(levels[0]), sum(1 for r in levels for v in r if v)))
    made = 0
    for gi, cls in enumerate(GAMES):
        if cls.key not in wanted:
            continue
        try:
            rng = random.Random(args.seed * 31 + gi)
            env = cls(levels)
            clips, scores = build_clips(env, rng)
            for th, dark, suffix in ((LIGHT, False, ""), (DARK, True, "-dark")):
                svg = render(env, clips, scores, th, dark)
                with open(os.path.join(args.out, "ai-%s%s.svg" % (cls.key, suffix)), "w", encoding="utf-8") as f:
                    f.write(svg)
            made += 1
        except Exception as exc:  # keep going: one broken game must not block the others
            print("  %s failed: %r" % (cls.key, exc), file=sys.stderr)
    if not made:
        sys.exit("No games were generated")


if __name__ == "__main__":
    main()
