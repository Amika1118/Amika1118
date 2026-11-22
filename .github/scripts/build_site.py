#!/usr/bin/env python3
"""Assemble the `output` branch: every generated SVG, a manifest and the viewer.

    python3 .github/scripts/build_site.py \
        --classic dist --ai ai_dist --viewer .github/viewer/index.html --out public

Published layout (flat, so README <img> URLs stay stable):

    ai-<game>.svg / ai-<game>-dark.svg      trained Q-learning agents
    <game>.svg    / <game>-dark.svg         classic renderers
    arcade.svg    / arcade-dark.svg         a random pick (70% AI) for simple embeds
    manifest.json                           what the viewer loads
    index.html                              the viewer

A missing source folder is not fatal (the AI step is allowed to fail); having no
games at all is.
"""
import argparse
import datetime
import json
import random
import shutil
import sys
from pathlib import Path

CLASSIC_TITLES = {
    "pacman": "PAC-MAN", "breakout": "BREAKOUT", "galaga": "GALAGA",
    "puzzle-bobble": "PUZZLE BOBBLE", "bomberman": "BOMBERMAN", "minesweeper": "MINESWEEPER",
}
AI_SHARE = 0.7


def copy_pair(src_light, src_dark, dst_dir, stem):
    shutil.copy(src_light, dst_dir / f"{stem}.svg")
    shutil.copy(src_dark, dst_dir / f"{stem}-dark.svg")


def collect_ai(src, dst):
    games = []
    meta_path = src / "ai-meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for dark in sorted(src.glob("ai-*-dark.svg")):
        stem = dark.name[: -len("-dark.svg")]
        light = src / f"{stem}.svg"
        if not light.exists():
            print(f"::warning::{light} missing, skipping {stem}")
            continue
        key = stem[len("ai-"):]
        info = meta.get(key, {})
        copy_pair(light, dark, dst, stem)
        games.append({
            "id": stem, "kind": "ai", "name": info.get("title", key.upper()),
            "light": f"{stem}.svg", "dark": f"{stem}-dark.svg",
            "episodes": info.get("episodes"), "stage_scores": info.get("stage_scores"),
            "score_word": info.get("score_word"),
        })
    return games


def collect_classic(src, dst, names):
    games = []
    for name in names:
        light = src / f"{name}-contribution-graph.svg"
        dark = src / f"{name}-contribution-graph-dark.svg"
        if not (light.exists() and dark.exists()):
            print(f"::warning::classic game '{name}' not found in {src}")
            continue
        copy_pair(light, dark, dst, name)
        games.append({
            "id": name, "kind": "classic", "name": CLASSIC_TITLES.get(name, name.upper()),
            "light": f"{name}.svg", "dark": f"{name}-dark.svg",
        })
    return games


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classic", type=Path, default=Path("dist"))
    ap.add_argument("--ai", type=Path, default=Path("ai_dist"))
    ap.add_argument("--classic-games", default=",".join(CLASSIC_TITLES))
    ap.add_argument("--viewer", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("public"))
    ap.add_argument("--seed", type=int, default=None, help="make the arcade.svg pick reproducible")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ai = collect_ai(args.ai, args.out) if args.ai.is_dir() else []
    classic = collect_classic(args.classic, args.out, args.classic_games.split(",")) if args.classic.is_dir() else []
    games = ai + classic
    if not games:
        sys.exit("::error::No games were generated: nothing to publish")

    rng = random.Random(args.seed)
    pool = ai if ai and (not classic or rng.random() < AI_SHARE) else classic
    pick = rng.choice(pool)
    print(f"Default arcade.svg -> {pick['id']} ({pick['kind']})")
    shutil.copy(args.out / pick["light"], args.out / "arcade.svg")
    shutil.copy(args.out / pick["dark"], args.out / "arcade-dark.svg")

    manifest = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "games": games,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy(args.viewer, args.out / "index.html")
    print(f"Published {len(ai)} AI + {len(classic)} classic games")


if __name__ == "__main__":
    main()
