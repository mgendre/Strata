"""--conversation-cache-disk in the server: the log's RAM and disk restores told apart, the engine's status.json, the
disk's writes since boot, the usage samples and the Monitor's view of them."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from serve.server import (ConvCacheLog, ConvDiskStatus, StoreUses, conversation_cache_view, conversation_disk_dir,
                          engine_extra_args, long_read_speed, prelude_cache_dir, system_role_args)


class Engine:
    def __init__(self, args, cwd="/srv/strata"):
        self.spawn = ("/bin/strata", args, cwd, "strata.log", {}, False, None)


class ConvDiskServer(unittest.TestCase):
    RAM = "strata serve: conversation cache: restored 4000 tokens (live) in 30.0 ms; parked=2 bytes=209715200\n"
    DISK = ("strata serve: conversation cache: restored 90000 tokens from disk in 812.0 ms, read 1400 MiB (its return "
            "2); parked=1 bytes=104857600\n")
    STORED = ("strata serve: conversation cache: disk saved 90000 tokens (left RAM) in 650.0 ms: "
              "wrote 1400 MiB (22 of 22 blocks new), 0 MiB already on disk; disk=1 (1400 MiB) written_today=1400 MiB\n")

    def test_ram_and_disk_restores_are_told_apart(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.RAM + self.STORED + self.DISK + self.RAM, encoding="utf-8")
            st = ConvCacheLog().poll(str(log), 0)
            self.assertEqual((st["restores"], st["restores_disk"], st["tokens_ram"]), (2, 1, 8000))
            self.assertEqual((st["last_event"], st["last_source"]), ("restored", "ram"))
            log.write_text(self.DISK, encoding="utf-8")
            st = ConvCacheLog().poll(str(log), 0)
            self.assertEqual((st["restores"], st["restores_disk"], st["last_source"]), (0, 1, "disk"))
            back, written = [], []
            log.write_text(self.RAM + self.STORED + self.DISK + self.RAM, encoding="utf-8")
            reader = ConvCacheLog()
            reader.on_disk = lambda t, n: back.append(n)            # each restore from disk, once, for its time
            reader.on_disk_saved = lambda t, n: written.append(n)   # ... and each conversation written to it
            reader.poll(str(log), 0)
            reader.poll(str(log), 0)
            self.assertEqual((back, written), ([90000], [90000]))

    def test_prelude_restores_are_counted_apart(self):
        prelude = ("strata serve: prelude cache: restored a prelude of 8413 tokens in 187.0 ms, read 235 MiB (its use 1); "
                   "preludes=2\n")
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.RAM + prelude + self.DISK + prelude, encoding="utf-8")
            st = ConvCacheLog().poll(str(log), 0)
            self.assertEqual((st["restores"], st["restores_disk"], st["restores_prelude"]), (1, 1, 2))
            self.assertEqual((st["tokens_ram"], st["tokens_prelude"]), (4000, 16826))
            self.assertEqual((st["last_event"], st["last_tokens"], st["last_source"]), ("restored", 8413, "prelude"))
            seen, saved = [], []
            log.write_text(self.RAM + prelude + self.DISK + prelude +
                           "strata serve: prelude cache: saved a prelude of 9090 tokens in 216.0 ms: wrote 244 MiB; "
                           "preludes=6 (1496 MiB)\n", encoding="utf-8")
            reader = ConvCacheLog()
            reader.on_prelude = lambda t, n: seen.append(n)      # each prelude restore, once, for its time
            reader.on_prelude_saved = lambda t, n: saved.append(n)   # ... and each one written to the store
            reader.poll(str(log), 0)
            reader.poll(str(log), 0)
            self.assertEqual((seen, saved), ([8413, 8413], [9090]))

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

    def test_conversations_that_left_ram(self):
        lines = ("strata serve: conversation cache: left RAM (budget full): 59529 tokens, on disk 3100 behind\n"
                 "strata serve: conversation cache: left RAM (slots full): 9000 tokens, not on disk (below the first save step)\n"
                 "strata serve: conversation cache: left RAM (RAM low): 41000 tokens, on disk 900 behind\n")
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(lines, encoding="utf-8")
            st = ConvCacheLog().poll(str(log), 0)
            self.assertEqual((st["left_ram"], st["left_on_disk"], st["left_not_saved"]), (3, 2, 1))

    def test_the_folder_comes_from_the_engine_arguments(self):
        self.assertEqual(conversation_disk_dir(Engine(["--conversation-cache-disk", "/data/cc"])), "/data/cc")
        self.assertEqual(conversation_disk_dir(Engine(["--conversation-cache-disk", "cc"])), "/srv/strata/cc")
        self.assertIsNone(conversation_disk_dir(Engine(["--conversation-cache-mib", "8192"])))
        self.assertIsNone(conversation_disk_dir(Engine(["--conversation-cache-disk"])))   # no value
        self.assertIsNone(conversation_disk_dir(object()))

    def test_status_json_is_read_again_when_it_changes(self):
        with tempfile.TemporaryDirectory() as d:
            status = ConvDiskStatus()
            self.assertIsNone(status.read(d))                  # the engine has not written it yet
            p = Path(d) / "status.json"
            p.write_text(json.dumps({"disk": {"entries": 1}}), encoding="utf-8")
            self.assertEqual(status.read(d)["disk"]["entries"], 1)
            p.write_text(json.dumps({"disk": {"entries": 2}}), encoding="utf-8")
            os.utime(p, ns=(1, 2_000_000_000_000_000_000))
            self.assertEqual(status.read(d)["disk"]["entries"], 2)
            p.write_text("{cut sh", encoding="utf-8")        # a half-written file keeps the last good one
            os.utime(p, ns=(1, 3_000_000_000_000_000_000))
            self.assertEqual(status.read(d)["disk"]["entries"], 2)

    def test_the_saving_uses_long_reads_only(self):
        hist = [{"prompt_read": 30000, "prompt_ms": 15000.0}, {"prompt_read": 10000, "prompt_ms": 5000.0},
                {"prompt_read": 40, "prompt_ms": 355.0}, {"prompt_read": None, "prompt_ms": None}]
        self.assertEqual(long_read_speed(hist), 2000.0)
        self.assertIsNone(long_read_speed([{"prompt_read": 500, "prompt_ms": 900.0}]))   # nothing long yet
        self.assertIsNone(long_read_speed([]))

    def test_the_view_carries_each_store_apart(self):
        info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 3}   # no disk tier
        prelude = {"entries": [{"id": "p", "tokens": 8451, "returns": 2, "served_tokens": 16902, "last_used": 9}]}
        uses = {"uses": [[100, 8451]], "saves": [[40, 8451]]}
        view = conversation_cache_view(info, [], {}, ConvCacheLog().state, None, None, prelude, uses)
        self.assertNotIn("disk", view)
        self.assertEqual((view["prelude"]["uses"], view["prelude"]["saves"], view["prelude"]["stored"]),
                         ([[100, 8451]], [[40, 8451]], 1))
        self.assertNotIn("prelude", conversation_cache_view(info, [], {}, ConvCacheLog().state))   # no prelude cache
        disk = {"entries": [{"id": "a"}, {"id": "b"}]}
        view = conversation_cache_view(info, [], {}, ConvCacheLog().state, disk, {"uses": [[90, 30000]], "saves": []})
        self.assertEqual((view["disk"]["uses"], view["disk"]["stored"]), ([[90, 30000]], 2))   # the preludes' shape
        self.assertEqual(prelude_cache_dir(Engine(["--prelude-cache", "pre"])), "/srv/strata/pre")
        self.assertIsNone(prelude_cache_dir(Engine(["--conversation-cache-disk", "d"])))

    def test_store_uses_keep_the_last_day(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "uses.json")
            u = StoreUses()
            u.add(1000, 9090)
            u.add_saved(900, 9090)
            u.add(1000 + 23 * 3600, 10564)
            self.assertEqual(u.view(now=1000 + 23 * 3600), {"uses": [[1000, 9090], [83800, 10564]], "saves": [[900, 9090]]})
            self.assertEqual(u.view(now=1001 + 24 * 3600), {"uses": [[83800, 10564]], "saves": []})   # older than a day: gone
            u.add_saved(83900, 10564)
            u.save(path)
            again = StoreUses()
            again.load(path, now=83900)
            self.assertEqual(again.view(now=83900), {"uses": [[83800, 10564]], "saves": [[83900, 10564]]})
            os.remove(path)
            again.save(path)                                    # nothing new since the load: no write
            self.assertFalse(os.path.exists(path))
            Path(path).write_text(json.dumps([[83800, 10564]]), encoding="utf-8")   # the first version's list: restores
            first = StoreUses()
            first.load(path, now=83900)
            self.assertEqual(first.view(now=83900), {"uses": [[83800, 10564]], "saves": []})
            Path(path).write_text("[[1, 2", encoding="utf-8")   # damaged: ignored
            broken = StoreUses()
            broken.load(path)
            self.assertEqual(broken.view(), {"uses": [], "saves": []})

    def test_the_view(self):
        info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 3}
        totals = {"reused": 100, "prompt_tokens": 2100, "prompt_ms": 1000}
        parked = ConvCacheLog().state
        hist = [{"prompt_read": 20000, "prompt_ms": 10000.0}, {"prompt_read": 40, "prompt_ms": 350.0}]
        v = conversation_cache_view(info, hist, totals, parked, None, {"uses": [], "saves": []})
        self.assertEqual(v["disk"]["prompt_tok_s"], 2000.0)   # the long read only, not the short one
        self.assertNotIn("disk", conversation_cache_view(info, [], totals, parked))   # no disk tier


if __name__ == "__main__":
    unittest.main()
