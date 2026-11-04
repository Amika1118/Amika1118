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
    """Interface: reset(rng) / state() / step(a)->(reward, done) / positions() / result().

    positions() returns one (x, y[, visible]) tuple per animated sprite, in cell units.
    """
    key = title = ""
    n_actions = 4
    alpha, gamma = 0.2, 0.95
    eps_start, eps_end = 1.0, 0.05
    eval_eps = 0.03
    episodes, cap_train = 2000, 250
    rows = 7
    score_word = "SCORE"

    def __init__(self, levels):
        self.lv = levels
        self.W = len(levels[0])
        self.food0 = [(r, c) for r in range(7) for c in range(self.W) if levels[r][c] > 0]
        self.removed = []
        self.score = 0

    def solid_cells(self):
        return [(r, c, self.lv[r][c]) for (r, c) in self.food0]

    def extra_static(self, th, anim_for=None):
        return ""

    def mover_svgs(self, th):
        return []

    def progress(self):
        return self.score

    progress_ref = None      # fixed reference for the live bar; None = best attempt shown

    def rank(self):
        """Higher is better; used to pick the best rollout and to draw the learning curve."""
        return self.score

    def result(self):
        return "%s %d" % (self.score_word, self.score)


class PacMan(Game):
    """Pac-Man with power pellets: ghosts have different personalities and can be hunted."""
    key, title = "pacman", "PAC-MAN"
    DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    episodes, cap_train = 3000, 260
    alpha, gamma = 0.15, 0.93
    eps_start, eps_end = 0.8, 0.02
    FRIGHT = 20

    def __init__(self, levels):
        Game.__init__(self, levels)
        top = max((levels[r][c] for r, c in self.food0), default=0)
        cand = sorted((c, r) for (r, c) in self.food0 if levels[r][c] == top)
        self.power0, last = [], -99
        for c, r in cand:                       # up to 4 well-spaced power pellets
            if c - last >= 10 and c > 3 and len(self.power0) < 4:
                self.power0.append((r, c))
                last = c

    def reset(self, rng):
        self.rng = rng
        self.px, self.py, self.last = 0, 3, 3
        self.food = set(self.food0)
        self.power = set(self.power0)
        self.spawn = [(self.W - 1, 1), (self.W - 1, 5)]
        self.ghosts = [list(p) for p in self.spawn]
        self.fright = 0
        self.score, self.eaten, self.dead = 0, 0, False
        self.removed = []
        self._eat()

    def _eat(self):
        key = (self.py, self.px)
        if key not in self.food:
            return 0
        self.food.discard(key)
        self.eaten += 1
        self.score += 1
        r = 1
        self.removed.append(("c_%d_%d" % key, 0))
        if key in self.power:
            self.power.discard(key)
            self.removed.append(("p_%d_%d" % key, 0))
            self.fright = self.FRIGHT
            self.score += 1
            r += 2
        return r

    def _nearest(self, items):
        px, py = self.px, self.py
        return min(items, key=lambda f: abs(f[0] - py) + abs(f[1] - px)) if items else None

    def state(self):
        px, py = self.px, self.py
        fr = self.fright > 0
        dang = []
        for dx, dy in self.DIRS:
            nx, ny = px + dx, py + dy
            if not (0 <= nx < self.W and 0 <= ny < 7):
                nx, ny = px, py
            if fr:
                dang.append(0)
                continue
            dg = min(abs(nx - g[0]) + abs(ny - g[1]) for g in self.ghosts)
            dang.append(2 if dg <= 1 else 1 if dg <= 3 else 0)
        f = self._nearest(self.food)
        if f:
            d = abs(f[0] - py) + abs(f[1] - px)
            fs = (sign(f[1] - px), sign(f[0] - py), 0 if d <= 2 else 1)
        else:
            fs = (0, 0, 1)
        extra = 0
        if fr:      # hunting: direction of the nearest frightened ghost
            g = min(self.ghosts, key=lambda g: abs(g[0] - px) + abs(g[1] - py))
            if abs(g[0] - px) + abs(g[1] - py) <= 8:
                extra = (1, sign(g[0] - px), sign(g[1] - py))
        elif self.power and min(abs(g[0] - px) + abs(g[1] - py) for g in self.ghosts) <= 6:
            p = self._nearest(self.power)       # fleeing: is a power pellet close?
            if abs(p[0] - py) + abs(p[1] - px) <= 9:
                extra = (2, sign(p[1] - px), sign(p[0] - py))
        return tuple(dang) + fs + (extra,)

    def _collide(self):
        """Resolve pac vs ghosts. Returns reward delta, or None if pac was caught."""
        r = 0
        for i, g in enumerate(self.ghosts):
            if g[0] == self.px and g[1] == self.py:
                if self.fright > 0:
                    r += 3
                    self.score += 3
                    self.ghosts[i] = list(self.spawn[i])
                else:
                    self.dead = True
                    return None
        return r

    def _move_ghost(self, i, g):
        rng = self.rng
        px, py = self.px, self.py
        if self.fright > 0:
            if rng.random() > 0.45:
                return
            if rng.random() < 0.7:               # run away from pac
                ddx, ddy = g[0] - px, g[1] - py
            else:
                ddx, ddy = rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1])
        else:
            if rng.random() > (0.72 if i == 0 else 0.62):
                return
            if rng.random() < 0.2:               # a little noise
                ddx, ddy = rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1])
            else:
                tx, ty = px, py
                if i == 1:                       # flanker: aims 3 cells ahead of pac
                    ax, ay = self.DIRS[self.last]
                    tx, ty = clamp(px + 3 * ax, 0, self.W - 1), clamp(py + 3 * ay, 0, 6)
                ddx, ddy = tx - g[0], ty - g[1]
        if ddx == 0 and ddy == 0:
            return
        if abs(ddx) > abs(ddy) or (abs(ddx) == abs(ddy) and rng.random() < 0.5):
            g[0] = clamp(g[0] + sign(ddx), 0, self.W - 1)
        else:
            g[1] = clamp(g[1] + sign(ddy), 0, 6)

    def step(self, a):
        dx, dy = self.DIRS[a]
        nx, ny = self.px + dx, self.py + dy
        if 0 <= nx < self.W and 0 <= ny < 7:
            self.px, self.py = nx, ny
        self.last = a
        reward = -0.02 + self._eat()
        c = self._collide()
        if c is None:
            return reward - 10, True
        reward += c
        for i, g in enumerate(self.ghosts):
            self._move_ghost(i, g)
        c = self._collide()
        if c is None:
            return reward - 10, True
        reward += c
        if self.fright > 0:
            self.fright -= 1
        if not self.food:
            return reward + 5, True
        return reward, False

    def positions(self):
        fr = 1 if self.fright > 0 else 0
        alive = 0 if self.dead else 1
        gs = self.ghosts
        return [(self.px, self.py, alive),
                (gs[0][0], gs[0][1], 1 - fr), (gs[1][0], gs[1][1], 1 - fr),
                (gs[0][0], gs[0][1], fr), (gs[1][0], gs[1][1], fr)]

    def progress(self):
        return self.score

    def result(self):
        return "%s %d%s" % (self.score_word, self.score, " (CAUGHT)" if self.dead else "")

    def extra_static(self, th, anim_for=None):
        out = []
        for r, c in self.power0:
            pid = "p_%d_%d" % (r, c)
            out.append('<circle cx="%d" cy="%d" r="2.6" fill="%s">%s</circle>' % (
                c * PITCH + 5, r * PITCH + 5, th["ball"], anim_for(pid) if anim_for else ""))
        return "".join(out)

    def mover_svgs(self, th):
        pac = '<path d="M5 5 L10.02 2.1 A5.8 5.8 0 1 0 10.02 7.9 Z" fill="#FFD21E"/>'

        def ghost(col, eyes="#fff", pupil="#1a1a2e"):
            return ('<path d="M0 10 V5 A5 5 0 0 1 10 5 V10 L8.3 8.3 L6.7 10 L5 8.3 L3.3 10 L1.7 8.3 Z" fill="%s"/>'
                    '<circle cx="3.5" cy="4.5" r="1.4" fill="%s"/><circle cx="6.5" cy="4.5" r="1.4" fill="%s"/>'
                    '<circle cx="3.9" cy="4.7" r=".7" fill="%s"/><circle cx="6.9" cy="4.7" r=".7" fill="%s"/>'
                    % (col, eyes, eyes, pupil, pupil))
        return [pac, ghost("#ff4d4d"), ghost("#ff9ecf"), ghost("#3b6cff"), ghost("#3b6cff")]


class Snake(Game):
    """Classic snake that eats contribution cells (grows up to MAXLEN) and learns not to crash."""
    key, title = "snake", "SNAKE"
    n_actions = 3                       # 0 straight, 1 turn left, 2 turn right
    episodes, cap_train = 3000, 250
    alpha, gamma = 0.2, 0.9
    eps_start, eps_end = 0.6, 0.02
    score_word = "EATEN"
    MAXLEN = 16
    DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]      # up, right, down, left

    def reset(self, rng):
        self.rng = rng
        self.body = [(2, 3), (1, 3), (0, 3)]       # (x, y), head first
        self.dir = 1
        self.food = set(self.food0)
        self.score, self.dead = 0, False
        self.removed = []
        self._nr = self._nearest(self.body[0])

    def _nearest(self, head):
        if not self.food:
            return None
        hx, hy = head
        fr, fc = min(self.food, key=lambda f: abs(f[0] - hy) + abs(f[1] - hx))
        return abs(fr - hy) + abs(fc - hx), fr, fc

    def _blocked(self, x, y, occ):
        return x < 0 or x >= self.W or y < 0 or y >= 7 or (x, y) in occ

    def state(self):
        hx, hy = self.body[0]
        occ = set(self.body[:-1])
        d = self.dir

        def danger(rel):
            dx, dy = self.DIRS[(d + rel) % 4]
            return 1 if self._blocked(hx + dx, hy + dy, occ) else 0
        sx, sy = self.DIRS[d]
        ahead2 = 1 if self._blocked(hx + 2 * sx, hy + 2 * sy, occ) else 0
        if self._nr:
            dist, fr, fc = self._nr
            fs = (sign(fc - hx), sign(fr - hy), 0 if dist <= 3 else 1)
        else:
            fs = (0, 0, 1)
        return (danger(0), danger(3), danger(1), ahead2, d) + fs

    def step(self, a):
        self.dir = (self.dir + (0, 3, 1)[a]) % 4
        dx, dy = self.DIRS[self.dir]
        hx, hy = self.body[0]
        nx, ny = hx + dx, hy + dy
        eating = (ny, nx) in self.food
        grow = eating and len(self.body) < self.MAXLEN
        occ = set(self.body) if grow else set(self.body[:-1])
        if self._blocked(nx, ny, occ):
            self.dead = True
            return -10, True
        self.body.insert(0, (nx, ny))
        if not grow:
            self.body.pop()
        reward = -0.02
        old = self._nr[0] if self._nr else 0
        if eating:
            self.food.discard((ny, nx))
            self.removed.append(("c_%d_%d" % (ny, nx), 0))
            self.score += 1
            reward += 1
        self._nr = self._nearest((nx, ny))
        if not eating and self._nr:
            reward += 0.1 * (old - self._nr[0])       # shaping: head for food
        if not self.food:
            return reward + 5, True
        return reward, False

    def positions(self):
        pos = [(x, y) for x, y in self.body]
        return pos + [(-5, -5)] * (self.MAXLEN - len(pos))

    def result(self):
        return "%s %d%s" % (self.score_word, self.score, " (CRASHED)" if self.dead else "")

    def mover_svgs(self, th):
        out = ['<rect x="0" y="0" width="10" height="10" rx="3.5" fill="#34d399"/>'
               '<circle cx="3.2" cy="3.6" r="1.1" fill="#0d1117"/><circle cx="6.8" cy="3.6" r="1.1" fill="#0d1117"/>']
        for i in range(1, self.MAXLEN):
            op = max(0.35, 0.95 - i * 0.04)
            out.append('<rect x="0.8" y="0.8" width="8.4" height="8.4" rx="3" fill="#34d399" opacity="%.2f"/>' % op)
        return out


class Breakout(Game):
    key, title = "breakout", "BREAKOUT"
    n_actions = 3
    rows, PR = 16, 15
    episodes, cap_train = 6000, 400
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
    episodes, cap_train = 1600, 300
    eval_eps = 0.0
    alpha, gamma = 0.3, 0.99
    eps_start, eps_end = 0.3, 0.02
    DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    score_word = "STEPS"

    def __init__(self, levels):
        Game.__init__(self, levels)
        self.start, self.goal = (3, 0), (3, self.W - 1)
        self.progress_ref = self.W - 1
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

    progress_ref = None

    def heat(self, snap):
        """What the agent believes about each cell: normalised best Q-value (0..1)."""
        vals = {}
        for (r, c), q in snap.items():
            if (r, c) not in self.walls:
                vals[(r, c)] = max(q)
        if not vals:
            return {}
        lo, hi = min(vals.values()), max(vals.values())
        if hi - lo < 1e-9:
            return {k: 0.0 for k in vals}
        return {k: (v - lo) / (hi - lo) for k, v in vals.items()}

    def progress(self):
        return self.maxc

    def reset(self, rng):
        self.rng = rng
        self.maxc = 0
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
        self.maxc = max(self.maxc, self.c)
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


GAMES = [PacMan, Snake, Breakout, Invaders, Maze]


# --------------------------------------------------------------- training --
def evaluate(env, snap, rng, eps, m=6):
    """Mean rank of m greedy episodes, played exactly like the clips shown in the SVG."""
    pol = make_policy(snap, env.n_actions, rng, eps)
    total = 0
    for _ in range(m):
        env.reset(rng)
        s = env.state()
        for _ in range(MAX_PLAY - 1):
            _, done = env.step(pol(s))
            env.removed.clear()
            if done:
                break
            s = env.state()
        total += env.rank()
    return total / m


def train(env, rng, scale=1.0):
    """Q-learning with checkpoints. The final stage is the best checkpoint seen late in
    training (standard policy checkpointing), found by playing 6 greedy test games."""
    n = max(20, int(env.episodes * scale))
    early = sorted({0, max(1, int(0.04 * n)), int(0.2 * n)})
    check_from, every = int(0.3 * n), max(1, n // 16)
    agent = QAgent(env.n_actions, env.alpha, env.gamma)
    snaps, scores, best = {}, [], (-1e9, n)
    for ep in range(n + 1):
        if ep in early:
            snaps[ep] = agent.snapshot()
        if ep >= check_from and (ep % every == 0 or ep == n):
            snap = agent.snapshot()
            val = evaluate(env, snap, random.Random(777 + ep), env.eval_eps)
            if val >= best[0]:
                best = (val, ep)
                snaps["best"] = snap
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
    snaps[best[1]] = snaps.pop("best")
    return n, early + [best[1]], snaps, scores


def rollout(env, snap, rng, eps):
    pol = make_policy(snap, env.n_actions, rng, eps)
    env.reset(rng)
    frames, events = [env.positions()], [(0, i, v) for i, v in env.removed]
    prog = [env.progress()]
    env.removed.clear()
    s = env.state()
    for _ in range(MAX_PLAY - 1):
        r, done = env.step(pol(s))
        frames.append(env.positions())
        prog.append(env.progress())
        events += [(len(frames) - 1, i, v) for i, v in env.removed]
        env.removed.clear()
        if done:
            break
        s = env.state()
    return dict(frames=frames, events=events, prog=prog, score=env.score, rank=env.rank(),
                result=env.result())


def build_clips(env, rng, scale=1.0):
    t0 = time.time()
    n, ckpts, snaps, scores = train(env, rng, scale)
    clips = []
    for j, e in enumerate(ckpts):
        tries = 1 if e == 0 else (5 if j == len(ckpts) - 1 else 3)
        eps = 1.0 if e == 0 else env.eval_eps
        best = None
        for _ in range(tries):
            c = rollout(env, snaps[e], rng, eps)
            if best is None or c["rank"] > best["rank"] or (
                    c["rank"] == best["rank"] and len(c["frames"]) < len(best["frames"])):
                best = c
        best["episodes"], best["stage"], best["best_of"] = e, j + 1, tries
        if hasattr(env, "heat"):
            best["heat"] = env.heat(snaps[e])
        clips.append(best)
    print("  %-9s trained %d episodes in %.1fs | stage scores: %s" % (
        env.key, n, time.time() - t0, [c["score"] for c in clips]))
    return clips, scores, n


# -------------------------------------------------------------- rendering --
def fmt(x):
    s = ("%.1f" % x).rstrip("0").rstrip(".")
    return s if s else "0"


def compress(series):
    out, prev, first = [], None, True
    for i, v in enumerate(series):
        if first or v != prev:
            out.append((i, v))
            prev, first = v, False
    return out


def discrete_pts(attr, pts, F, dur, transform=False):
    """Discrete SMIL animation from (frame, value) change points; '' if nothing changes."""
    if len(pts) < 2:
        return ""
    vals = ";".join(str(v) for _, v in pts)
    keys = ";".join("%.5f" % (i / F) for i, _ in pts)
    tag, extra = ("animateTransform", ' type="translate"') if transform else ("animate", "")
    return ('<%s attributeName="%s"%s calcMode="discrete" values="%s" keyTimes="%s" dur="%s" '
            'repeatCount="indefinite"/>' % (tag, attr, extra, vals, keys, dur))


def anim_vis(intervals, F, dur):
    """opacity animation: visible during the given [start, end) frame intervals."""
    pts = [(0, 1 if intervals[0][0] == 0 else 0)]
    for s, e in intervals:
        if s != 0:
            pts.append((s, 1))
        if e < F:
            pts.append((e, 0))
    return discrete_pts("opacity", pts, F, dur)


def curve_series(scores, bins=40):
    if not scores:
        return [0.0]
    size = max(1, len(scores) // bins)
    return [sum(scores[i:i + size]) / len(scores[i:i + size]) for i in range(0, len(scores), size)]


def render(env, clips, scores, n_eps, th):
    W = env.W
    width = MX * 2 + W * PITCH - 2
    arena_h = env.rows * PITCH
    height = TOP + arena_h + BOT
    # ---- global timeline
    starts, F = [], 0
    nm = len(clips[0]["frames"][0])
    xy = [[] for _ in range(nm)]
    vis = [[] for _ in range(nm)]
    prog = []
    for c in clips:
        starts.append(F)
        seq = [c["frames"][0]] * INTER + c["frames"] + [c["frames"][-1]] * HOLD
        pseq = [0] * INTER + c["prog"] + [c["prog"][-1]] * HOLD
        for fr, p in zip(seq, pseq):
            for m in range(nm):
                t = fr[m]
                xy[m].append((round(t[0] * PITCH, 1), round(t[1] * PITCH, 1)))
                vis[m].append(t[2] if len(t) > 2 else 1)
            prog.append(p)
        c["play_len"] = len(c["frames"]) + HOLD
        F += len(seq)
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
        return discrete_pts("opacity", sorted({f: v for f, v in ev[i]}.items()), F, dur)

    o = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
         'font-family="Menlo,Consolas,\'DejaVu Sans Mono\',monospace" role="img" '
         'aria-label="AI agent learning to play %s on my GitHub contribution graph">' % (
             width, height, width, height, env.title.title())]
    o.append("<title>%s - AI agent learning on my contribution graph</title>" % env.title)
    o.append('<text x="%d" y="18" font-size="11" font-weight="bold" fill="%s" letter-spacing="1">'
             'AI ARCADE | %s</text>' % (MX, th["accent"], env.title))
    o.append('<text x="%d" y="18" font-size="9" fill="%s" text-anchor="end">'
             'Q-LEARNING | RETRAINED FROM SCRATCH ON EVERY RUN</text>' % (width - MX, th["muted"]))
    o.append('<g transform="translate(%d,%d)">' % (MX, TOP))
    for r in range(7):
        for c in range(W):
            o.append('<rect x="%d" y="%d" width="%d" height="%d" rx="2" fill="%s"/>' % (
                c * PITCH, r * PITCH, CELL, CELL, th["empty"]))
    # "brain view": what the agent believes about each cell (maze only)
    if clips[0].get("heat") is not None:
        cells = set()
        for c in clips:
            cells |= set(c["heat"])
        for (r, c) in sorted(cells):
            pts = [(cs, round(0.62 * cl["heat"].get((r, c), 0), 2)) for cl, cs in zip(clips, starts)]
            if max(v for _, v in pts) <= 0:
                continue
            o.append('<rect x="%d" y="%d" width="%d" height="%d" rx="2" fill="#34d399" opacity="%s">%s</rect>' % (
                c * PITCH, r * PITCH, CELL, CELL, pts[0][1], discrete_pts("opacity", pts, F, dur)))
    for r, c, lvl in env.solid_cells():
        cid = "c_%d_%d" % (r, c)
        base = '<rect id="%s" x="%d" y="%d" width="%d" height="%d" rx="2" fill="%s"' % (
            cid, c * PITCH, r * PITCH, CELL, CELL, th["lv"][lvl - 1])
        a = anim_for(cid)
        o.append(base + ("/>" if not a else ">" + a + "</rect>"))
    o.append(env.extra_static(th, anim_for))
    for m, svg in enumerate(env.mover_svgs(th)):
        a = discrete_pts("transform", [(i, "%s,%s" % (fmt(x), fmt(y))) for i, (x, y) in compress(xy[m])],
                         F, dur, transform=True)
        v = discrete_pts("opacity", compress(vis[m]), F, dur)
        o.append('<g transform="translate(%s,%s)" opacity="%s">%s%s%s</g>' % (
            fmt(xy[m][0][0]), fmt(xy[m][0][1]), vis[m][0], svg, a, v))
    o.append("</g>")
    # ---- live score bar
    bar_w = width - 2 * MX
    bar_y = TOP + arena_h + 1
    ref = env.progress_ref or max(max(c["prog"]) for c in clips) or 1
    widths = [round(min(1.0, p / ref) * bar_w, 1) for p in prog]
    o.append('<rect x="%d" y="%d" width="%d" height="3" rx="1.5" fill="%s"/>' % (MX, bar_y, bar_w, th["empty"]))
    o.append('<rect x="%d" y="%d" width="%s" height="3" rx="1.5" fill="%s">%s</rect>' % (
        MX, bar_y, fmt(widths[0]), th["accent"], discrete_pts("width", compress(widths), F, dur)))
    # ---- status texts
    sy = height - 11
    for c, cs in zip(clips, starts):
        if c["episodes"] == 0:
            t_train = "ATTEMPT 1 | UNTRAINED - PLAYING RANDOMLY"
        else:
            t_train = "TRAINING... %s EPISODES SO FAR" % format(c["episodes"], ",")
        o.append('<text x="%d" y="%d" font-size="10" fill="%s" opacity="0">%s%s</text>' % (
            MX, sy, th["muted"], t_train, anim_vis([(cs, cs + INTER)], F, dur)))
        t_play = "ATTEMPT %d/%d | %s EPISODES | %s" % (c["stage"], len(clips), format(c["episodes"], ","), c["result"])
        if c["best_of"] > 1:
            t_play += " | BEST OF %d" % c["best_of"]
        o.append('<text x="%d" y="%d" font-size="10" fill="%s" opacity="0">%s%s</text>' % (
            MX, sy, th["text"], t_play, anim_vis([(cs + INTER, cs + INTER + c["play_len"])], F, dur)))
    # ---- learning curve with a marker that follows the attempts
    cw, ch = 130, 16
    cx, cy = width - MX - cw, height - 9 - ch
    avg = curve_series(scores)
    hi = max(avg) or 1
    nb = max(1, len(avg) - 1)
    pts = " ".join("%s,%s" % (fmt(i * cw / nb), fmt(ch - (v / hi) * ch)) for i, v in enumerate(avg))
    mk = []
    for c in clips:
        idx = min(len(avg) - 1, int(round(c["episodes"] / max(1, n_eps) * nb)))
        mk.append((fmt(idx * cw / nb), fmt(ch - (avg[idx] / hi) * ch)))
    mx_pts = [(cs, m[0]) for cs, m in zip(starts, mk)]
    my_pts = [(cs, m[1]) for cs, m in zip(starts, mk)]
    o.append('<text x="%d" y="%d" font-size="8" fill="%s" text-anchor="end">LEARNING CURVE</text>' % (
        cx - 6, height - 11, th["muted"]))
    o.append('<g transform="translate(%d,%d)"><polyline points="%s" fill="none" stroke="%s" stroke-width="1.6" '
             'stroke-linejoin="round"/><circle cx="%s" cy="%s" r="2.8" fill="#FFD21E">%s%s</circle></g>' % (
                 cx, cy, pts, th["accent"], mk[0][0], mk[0][1],
                 discrete_pts("cx", mx_pts, F, dur), discrete_pts("cy", my_pts, F, dur)))
    o.append("</svg>")
    return "\n".join(o)


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=os.environ.get("GITHUB_REPOSITORY_OWNER", ""))
    ap.add_argument("--out", default="ai_dist")
    ap.add_argument("--demo", action="store_true", help="use a synthetic contribution graph")
    ap.add_argument("--seed", type=int, default=int(time.time() // 3600))
    ap.add_argument("--scale", type=float, default=1.0, help="multiply training episodes (testing)")
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
            clips, scores, n_eps = build_clips(env, rng, args.scale)
            for th, suffix in ((LIGHT, ""), (DARK, "-dark")):
                svg = render(env, clips, scores, n_eps, th)
                with open(os.path.join(args.out, "ai-%s%s.svg" % (cls.key, suffix)), "w", encoding="utf-8") as f:
                    f.write(svg)
            made += 1
        except Exception as exc:  # keep going: one broken game must not block the others
            import traceback
            traceback.print_exc()
            print("  %s failed: %r" % (cls.key, exc), file=sys.stderr)
    if not made:
        sys.exit("No games were generated")


if __name__ == "__main__":
    main()