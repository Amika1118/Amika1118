"""Tests for the AI arcade. Standard library only, no network:

    python3 -m unittest discover -s .github/scripts -v
"""
import json
import random
import sys
import tempfile
import unittest
import xml.dom.minidom
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import ai_arcade as A      # noqa: E402
import build_site          # noqa: E402

LEVELS = A.to_levels(A.demo_counts())


class LevelTests(unittest.TestCase):
    def test_empty_graph_is_all_zero(self):
        self.assertEqual(A.to_levels([[0] * 5 for _ in range(7)]), [[0] * 5 for _ in range(7)])

    def test_levels_are_bounded(self):
        flat = {v for row in LEVELS for v in row}
        self.assertTrue(flat <= {0, 1, 2, 3, 4})
        self.assertIn(0, flat)


class GameContractTests(unittest.TestCase):
    """The renderer assumes every game keeps a fixed number of sprites and terminates."""

    def test_every_game_terminates_with_stable_sprite_count(self):
        for cls in A.GAMES:
            with self.subTest(game=cls.key):
                env, rng = cls(LEVELS), random.Random(3)
                env.reset(rng)
                n_sprites = len(env.positions())
                self.assertEqual(n_sprites, len(env.mover_svgs(A.DARK)))
                for _ in range(A.MAX_PLAY):
                    _, done = env.step(rng.randrange(env.n_actions))
                    env.removed.clear()
                    self.assertEqual(len(env.positions()), n_sprites)
                    if done:
                        break


class DoodleRegressionTests(unittest.TestCase):
    """Doodle Jump used to score 0 at every stage: the agent could never bounce."""

    def test_a_random_agent_can_bounce(self):
        env, rng, best = A.Doodle(LEVELS), random.Random(1), 0
        for _ in range(200):
            env.reset(rng)
            for _ in range(60):
                if env.step(rng.randrange(3))[1]:
                    break
            best = max(best, env.score)
        self.assertGreater(best, 0)

    def test_platforms_stay_reachable(self):
        env, rng = A.Doodle(LEVELS), random.Random(5)
        env.reset(rng)
        for _ in range(200):
            env.step(1)
            rows = sorted(p[1] for p in env.platforms)
            self.assertTrue(all(b - a >= 2 for a, b in zip(rows, rows[1:])), rows)
            if env.dead:
                break

    def test_training_improves_the_agent(self):
        env = A.Doodle(LEVELS)
        clips, _, _ = A.build_clips(env, random.Random(11), scale=0.25)
        self.assertGreater(clips[-1]["score"], clips[0]["score"])


class RenderSmokeTests(unittest.TestCase):
    def test_every_game_renders_valid_svg(self):
        for cls in A.GAMES:
            with self.subTest(game=cls.key):
                env = cls(LEVELS)
                clips, scores, n = A.build_clips(env, random.Random(2), scale=0.05)
                for theme in (A.LIGHT, A.DARK):
                    xml.dom.minidom.parseString(A.render(env, clips, scores, n, theme))


class BuildSiteTests(unittest.TestCase):
    def _fake_svgs(self, folder, names):
        folder.mkdir()
        for n in names:
            (folder / n).write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")

    def _run(self, *argv):
        with mock.patch.object(sys, "argv", ["build_site.py", *argv]):
            build_site.main()

    def test_publishes_both_kinds_without_name_clashes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._fake_svgs(d / "ai", ["ai-pacman.svg", "ai-pacman-dark.svg"])
            self._fake_svgs(d / "classic", ["pacman-contribution-graph.svg",
                                            "pacman-contribution-graph-dark.svg"])
            (d / "viewer.html").write_text("<html></html>")
            self._run("--ai", str(d / "ai"), "--classic", str(d / "classic"), "--classic-games", "pacman",
                      "--viewer", str(d / "viewer.html"), "--out", str(d / "public"), "--seed", "1")
            out = d / "public"
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual({g["id"] for g in manifest["games"]}, {"ai-pacman", "pacman"})
            for name in ("ai-pacman.svg", "pacman-dark.svg", "arcade.svg", "arcade-dark.svg", "index.html"):
                self.assertTrue((out / name).exists(), name)

    def test_survives_missing_ai_folder(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._fake_svgs(d / "classic", ["pacman-contribution-graph.svg",
                                            "pacman-contribution-graph-dark.svg"])
            (d / "viewer.html").write_text("<html></html>")
            self._run("--ai", str(d / "nope"), "--classic", str(d / "classic"), "--classic-games", "pacman",
                      "--viewer", str(d / "viewer.html"), "--out", str(d / "public"))
            self.assertTrue((d / "public" / "arcade.svg").exists())

    def test_fails_when_nothing_to_publish(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "viewer.html").write_text("<html></html>")
            with self.assertRaises(SystemExit):
                self._run("--ai", str(d / "a"), "--classic", str(d / "c"),
                          "--viewer", str(d / "viewer.html"), "--out", str(d / "public"))


if __name__ == "__main__":
    unittest.main()
