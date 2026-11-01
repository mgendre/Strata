"""The Monitor's restore tables: each restore the engine's log tells (from RAM, from --conversation-cache-disk, from
--prelude-cache) becomes a row with the request's prompt line (the tokens read after the restored ones), the newest
first, 15 per table; the disk's and the preludes' tables only when the engine runs with them."""
import tempfile
import unittest
from pathlib import Path

from serve.server import ConvCacheLog, conversation_cache_view, engine_arg, long_read_speed, restore_rows

RAM = "strata serve: conversation cache: restored 4000 tokens (live) in 30.0 ms; parked=2 bytes=209715200\n"
DISK = "strata serve: conversation cache: disk: restored 90000 tokens in 812.0 ms, read 1400 MiB (score 2.0)\n"
PRELUDE = "strata serve: prelude cache: restored a prelude of 8413 tokens in 187.0 ms, read 235 MiB (score 3.0); preludes=2\n"
PROMPT = ("strata serve: prompt {} tokens = {} reused + {} read in {} ms (620.9 tok/s), 428 generated in 4351 ms "
          "(98.4 tok/s), drafts accepted 245 of 311, 2 checkpoints\n")


class Engine:
    def __init__(self, args):
        self.spawn = ("/bin/strata", args, "/srv", "strata.log", {}, False, None)


class RestoreTables(unittest.TestCase):
    def test_each_restore_gets_a_row_with_its_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(RAM + PROMPT.format(4500, 4000, 500, 700) + DISK + PROMPT.format(91567, 90000, 1567, 2668) +
                           PRELUDE, encoding="utf-8")
            reader = ConvCacheLog()
            st = reader.poll(str(log), 0)
            self.assertEqual((st["restores"], st["parked"]), (1, 2))   # a disk restore is not a RAM one
            ram, disk = st["rows"]["ram"], st["rows"]["disk"]
            self.assertEqual({k: ram[0][k] for k in ("tokens", "ms", "mib", "prompt", "read", "read_ms")},
                             {"tokens": 4000, "ms": 30.0, "mib": None, "prompt": 4500, "read": 500, "read_ms": 700})
            self.assertEqual([(r["tokens"], r["ms"], r["mib"], r["prompt"], r["read"]) for r in disk],
                             [(90000, 812.0, 1400, 91567, 1567)])
            self.assertEqual(st["rows"]["prelude"], [])           # its prompt line has not come yet: in progress
            self.assertEqual({k: st["pending"][k] for k in ("source", "tokens", "ms", "mib", "read")},
                             {"source": "prelude", "tokens": 8413, "ms": 187.0, "mib": 235, "read": None})
            with open(log, "a", encoding="utf-8") as f:
                f.write(PROMPT.format(9000, 8413, 587, 900))
            st = reader.poll(str(log), 0)
            self.assertIsNone(st["pending"])
            self.assertEqual([(r["tokens"], r["read"], r["read_ms"]) for r in st["rows"]["prelude"]], [(8413, 587, 900)])
            with open(log, "a", encoding="utf-8") as f:           # a restore whose request stopped before its prompt line
                f.write(DISK + RAM + PROMPT.format(4100, 4000, 100, 90))
            st = reader.poll(str(log), 0)
            self.assertEqual([(r["tokens"], r["read"]) for r in st["rows"]["disk"]], [(90000, 1567), (90000, None)])
            self.assertEqual([r["read"] for r in st["rows"]["ram"]], [500, 100])

    def test_the_tables_keep_the_latest_fifteen_newest_first(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text("".join(
                f"strata serve: conversation cache: restored {1000 + k} tokens (live) in 30.0 ms; parked=1 bytes=1\n" +
                PROMPT.format(1100 + k, 1000 + k, 100, 50) for k in range(20)), encoding="utf-8")
            st = ConvCacheLog().poll(str(log), 0)
            info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 3}
            view = conversation_cache_view(info, [], {}, st)
            self.assertEqual([r["tokens"] for r in view["recoveries"]], list(range(1019, 1004, -1)))
            self.assertNotIn("rows", view)
            self.assertNotIn("pending", view)
            self.assertNotIn("disk_recoveries", view)             # no --conversation-cache-disk: no table
            self.assertNotIn("prelude_recoveries", view)
            view = conversation_cache_view(info, [], {}, st, disk=True, prelude=True)
            self.assertEqual((view["disk_recoveries"], view["prelude_recoveries"]), ([], []))
        pending = {"source": "disk", "t": 5.0, "tokens": 7, "read": None}
        rows = restore_rows([{"t": 1.0, "tokens": 1}, {"t": 2.0, "tokens": 2}], pending, "disk")
        self.assertEqual([(r["tokens"], r.get("pending")) for r in rows], [(7, True), (2, None), (1, None)])
        self.assertNotIn("source", rows[0])
        self.assertEqual([r["tokens"] for r in restore_rows([{"t": 1.0, "tokens": 1}], pending, "prelude")], [1])
        self.assertEqual(restore_rows(None, None, "ram"), [])

    def test_the_saving_uses_long_reads_only(self):
        self.assertIsNone(long_read_speed([]))
        self.assertIsNone(long_read_speed([{"prompt_read": 4000, "prompt_ms": 2000.0}]))   # below 8,192 tokens
        hist = [{"prompt_read": 16384, "prompt_ms": 8000.0}, {"prompt_read": 8192, "prompt_ms": 4000.0},
                {"prompt_read": 100, "prompt_ms": 5000.0}]
        self.assertAlmostEqual(long_read_speed(hist), 2048.0)

    def test_the_engine_arguments(self):
        e = Engine(["--prelude-cache", "/p", "--conversation-cache-mib", "8192"])
        self.assertEqual(engine_arg(e, "--prelude-cache"), "/p")
        self.assertIsNone(engine_arg(e, "--conversation-cache-disk"))
        self.assertIsNone(engine_arg(Engine(["--prelude-cache"]), "--prelude-cache"))   # no value
        self.assertIsNone(engine_arg(object(), "--prelude-cache"))


if __name__ == "__main__":
    unittest.main()
