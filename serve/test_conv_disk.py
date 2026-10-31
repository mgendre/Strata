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
                          engine_extra_args, long_read_speed, prelude_cache_dir, restore_rows, system_role_args)


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

    PROMPT = ("strata serve: prompt {} tokens = {} reused + {} read in {} ms (620.9 tok/s), 428 generated in 4351 ms "
              "(98.4 tok/s), drafts accepted 245 of 311, 2 checkpoints\n")
    PRELUDE = ("strata serve: prelude cache: restored a prelude of 8413 tokens in 187.0 ms, read 235 MiB (its use 3); "
               "preludes=2\n")

    def test_each_restore_gets_a_row_with_its_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.RAM + self.PROMPT.format(4500, 4000, 500, 700) + self.DISK +
                           self.PROMPT.format(91567, 90000, 1567, 2668) + self.PRELUDE, encoding="utf-8")
            reader, disk, prelude = ConvCacheLog(), [], []
            reader.on_row = {"disk": disk.append, "prelude": prelude.append}
            st = reader.poll(str(log), 0)
            ram = st["ram_rows"]
            self.assertEqual(len(ram), 1)
            self.assertEqual({k: ram[0][k] for k in ("tokens", "ms", "mib", "nth", "prompt", "read", "read_ms")},
                             {"tokens": 4000, "ms": 30.0, "mib": None, "nth": None, "prompt": 4500, "read": 500,
                              "read_ms": 700})
            self.assertEqual([(r["tokens"], r["ms"], r["mib"], r["nth"], r["prompt"], r["read"]) for r in disk],
                             [(90000, 812.0, 1400, 2, 91567, 1567)])
            self.assertEqual(prelude, [])                     # its prompt line has not come yet: in progress
            self.assertEqual({k: st["pending"][k] for k in ("source", "tokens", "ms", "mib", "nth", "read")},
                             {"source": "prelude", "tokens": 8413, "ms": 187.0, "mib": 235, "nth": 3, "read": None})
            with open(log, "a", encoding="utf-8") as f:
                f.write(self.PROMPT.format(9000, 8413, 587, 900))
            st = reader.poll(str(log), 0)
            self.assertIsNone(st["pending"])
            self.assertEqual([(r["tokens"], r["read"], r["read_ms"]) for r in prelude], [(8413, 587, 900)])
            with open(log, "a", encoding="utf-8") as f:   # a restore whose request stopped before its prompt line
                f.write(self.DISK + self.RAM + self.PROMPT.format(4100, 4000, 100, 90))
            st = reader.poll(str(log), 0)
            self.assertEqual([(r["tokens"], r["read"]) for r in disk], [(90000, 1567), (90000, None)])
            self.assertEqual([r["read"] for r in st["ram_rows"]], [500, 100])
            self.assertEqual(ConvCacheLog().poll(str(log), 0)["ram_rows"][-1]["read"], 100)   # read again from the start

    def test_the_tables_keep_the_latest_fifteen_newest_first(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text("".join(
                f"strata serve: conversation cache: restored {1000 + k} tokens (live) in 30.0 ms; parked=1 bytes=1\n" +
                self.PROMPT.format(1100 + k, 1000 + k, 100, 50) for k in range(20)), encoding="utf-8")
            st = ConvCacheLog().poll(str(log), 0)
            info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 3}
            view = conversation_cache_view(info, [], {}, st)
            self.assertEqual([r["tokens"] for r in view["recoveries"]], list(range(1019, 1004, -1)))
            self.assertNotIn("ram_rows", view)
            self.assertNotIn("pending", view)
        pending = {"source": "disk", "t": 5.0, "tokens": 7, "read": None}
        rows = restore_rows([{"t": 1.0, "tokens": 1}, {"t": 2.0, "tokens": 2}], pending, "disk")
        self.assertEqual([(r["tokens"], r.get("pending")) for r in rows], [(7, True), (2, None), (1, None)])
        self.assertNotIn("source", rows[0])
        self.assertEqual([r["tokens"] for r in restore_rows([{"t": 1.0, "tokens": 1}], pending, "prelude")], [1])
        self.assertEqual(restore_rows(None, None, "ram"), [])

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
        self.assertEqual(view["disk"]["recoveries"], [])        # no rows yet
        row = {"t": 95.0, "tokens": 30000, "ms": 400.0, "mib": 500, "nth": 1, "prompt": 31000, "read": 1000, "read_ms": 900}
        view = conversation_cache_view(info, [], {}, ConvCacheLog().state, disk, {"uses": [], "saves": [], "recent": [row]},
                                       prelude, {"uses": [], "saves": []})
        self.assertEqual((view["disk"]["recoveries"], view["prelude"]["recoveries"]), ([row], []))
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
            row = {"t": 83850.5, "tokens": 10564, "ms": 180.0, "mib": 266, "nth": 4, "prompt": 11000, "read": 436,
                   "read_ms": 600}
            again.add_row(row)
            again.save(path)
            rows = StoreUses()
            rows.load(path, now=83900)
            self.assertEqual(rows.rows(), [row])                # the table's rows come back after a restart
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            data["recent"] = [{"t": 1, "tokens": 2, "ms": "x", "mib": True}, {"tokens": 5}, "junk"]
            Path(path).write_text(json.dumps(data), encoding="utf-8")
            odd = StoreUses()
            odd.load(path, now=83900)                           # a row without its time or tokens is left out;
            self.assertEqual(odd.rows(), [{"t": 1.0, "tokens": 2, "ms": None, "mib": None, "nth": None, "prompt": None,
                                           "read": None, "read_ms": None}])   # an odd value is unknown
            for k in range(20):
                odd.add_row(dict(row, tokens=k))
            self.assertEqual([r["tokens"] for r in odd.rows()], list(range(5, 20)))   # 15 at most
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


class StopWait(unittest.TestCase):
    """A clean stop writes what only RAM holds to the disk level (the engine's QUIT path, at most 30 s), so close() gives
    QUIT ENGINE_DISK_STOP_WAIT_S more when the engine runs with --conversation-cache-disk, and the usual wait without."""

    class Proc:
        def __init__(self):
            self.waits, self.ended, self.stdin, self.stdout = [], False, io.StringIO(), io.StringIO()

        def poll(self):
            return 0 if self.ended else None

        def wait(self, timeout=None):
            self.waits.append(timeout)
            self.ended = True
            return 0

    def close_wait(self, args):
        from serve import server
        eng = object.__new__(server.StrataEngine)
        eng.proc, eng.spawn, eng.pump, eng.log = self.Proc(), ("/bin/strata", args, "/srv", "x.log", {}, False, None), None, None
        proc = eng.proc
        eng.close()
        return proc.waits[0]

    def test_quit_waits_longer_with_the_disk_level(self):
        from serve import server
        self.assertEqual(self.close_wait(["--conversation-cache-disk", "/data/cc"]),
                         server.ENGINE_QUIT_WAIT_S + server.ENGINE_DISK_STOP_WAIT_S)
        self.assertEqual(self.close_wait(["--prompt-cache", "12"]), server.ENGINE_QUIT_WAIT_S)
        self.assertLess(server.ENGINE_QUIT_WAIT_S + server.ENGINE_DISK_STOP_WAIT_S + server.ENGINE_TERM_WAIT_S, 60)
