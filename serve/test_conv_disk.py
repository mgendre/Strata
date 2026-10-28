"""--conversation-cache-disk in the server: the log's RAM and disk restores told apart, the engine's status.json, the
disk's writes since boot, the usage samples and the Monitor's view of them."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from serve.server import (CacheHistory, ConvCacheLog, ConvDiskStatus, conversation_cache_view, conversation_disk_dir,
                          disk_written_since_boot)


class Engine:
    def __init__(self, args, cwd="/srv/strata"):
        self.spawn = ("/bin/strata", args, cwd, "strata.log", {}, False, None)


class ConvDiskServer(unittest.TestCase):
    RAM = "strata serve: conversation cache: restored 4000 tokens (live) in 30.0 ms; parked=2 bytes=209715200\n"
    DISK = ("strata serve: conversation cache: restored 90000 tokens from disk in 812.0 ms, read 1400 MiB (its return "
            "2); parked=1 bytes=104857600\n")
    STORED = ("strata serve: conversation cache: disk saved 90000 tokens (after the reply, a step reached) in 650.0 ms: "
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

    @unittest.skipUnless(os.path.exists("/sys/dev/block"), "Linux only")
    def test_the_disk_under_the_folder(self):
        with tempfile.TemporaryDirectory() as d:
            dev = disk_written_since_boot(d)
            if dev is not None:                                # (a tmpfs has no block device)
                self.assertTrue(dev["device"] and dev["written"] >= 0)
        self.assertIsNone(disk_written_since_boot(None))

    def test_history_keeps_one_day(self):
        h = CacheHistory()
        for i in range(CacheHistory.KEEP + 5):
            h.add(i, 2 * i, 3 * i, 4 * i, 5 * i, now=1000 + 60 * i)
        v = h.view()
        self.assertEqual(len(v), CacheHistory.KEEP)
        k = CacheHistory.KEEP + 4
        self.assertEqual(v[-1], [1000 + 60 * k, k, 2 * k, 3 * k, 4 * k, 5 * k])

    def test_the_view(self):
        info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 3, "conversation_cache_disk": 1}
        totals = {"reused": 100, "prompt_tokens": 2100, "prompt_ms": 1000}
        parked = ConvCacheLog().state
        disk = {"disk": {"entries": 1, "bytes": 5, "budget": 10}, "entries": []}
        v = conversation_cache_view(info, [], totals, parked, disk, {"device": "nvme1n1", "written": 7}, [[1, 2, 3, 4, 5, 6]])
        self.assertEqual(v["disk"]["prompt_tok_s"], 2000.0)
        self.assertEqual((v["disk"]["device"]["device"], v["disk"]["history"]), ("nvme1n1", [[1, 2, 3, 4, 5, 6]]))
        self.assertNotIn("disk", conversation_cache_view(info, [], totals, parked))
        off = dict(info, conversation_cache_disk=0)              # the engine said its disk levels are off
        self.assertNotIn("disk", conversation_cache_view(off, [], totals, parked, disk))


if __name__ == "__main__":
    unittest.main()
