"""--prelude-cache in the server: the id of "system" goes to the engine as --system-role-token (a prelude is then a
system turn only), and the prelude cache alone does not move a non-default effort to a trailing turn."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from serve.server import engine_extra_args, system_role_args


class PreludeArgs(unittest.TestCase):
    def test_the_system_role_token_goes_with_the_prelude_cache(self):
        class Tok:
            def __init__(self, ids): self.ids = ids
            def encode(self, text, parse_special=False): return self.ids
        with tempfile.TemporaryDirectory() as d:
            new, old = Path(d) / "strata", Path(d) / "strata-old"
            new.write_bytes(b"...--system-role-token..."); old.write_bytes(b"...")
            cfg = {"args": ["--prelude-cache", "/p"]}
            self.assertEqual(system_role_args(cfg, str(new), Tok([8948])), ["--system-role-token", "8948"])
            self.assertEqual(system_role_args({"args": []}, str(new), Tok([8948])), [])         # no prelude cache
            self.assertEqual(system_role_args(cfg, str(old), Tok([8948])), [])                  # an older engine
            self.assertEqual(system_role_args(cfg, str(new), Tok([1, 2])), [])                  # "system" split in two
            self.assertEqual(system_role_args({"args": ["--prelude-cache", "/p", "--system-role-token", "7"]},
                                              str(new), Tok([8948])), [])                       # set by hand: kept

    def test_the_prelude_cache_does_not_move_the_effort_to_the_end(self):
        class Tok:
            def encode(self, text, parse_special=False): return [8948]
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata"
            exe.write_bytes(b"...--system-role-token... --tail-role-token ID --serve: ...")
            prelude = {"args": ["--prelude-cache", "/p"]}
            self.assertEqual(engine_extra_args(prelude, str(exe), Tok()), (["--system-role-token", "8948"], False))
            self.assertEqual(engine_extra_args({}, str(exe), Tok()), ([], False))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(engine_extra_args(dict(prelude, effort_position="end"), str(exe), Tok()),
                                 (["--tail-role-token", "8948", "--system-role-token", "8948"], True))


if __name__ == "__main__":
    unittest.main()
