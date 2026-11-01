"""--conversation-cache-disk in the server: a clean stop writes what only RAM holds to the disk level (the engine's QUIT
path, at most 30 s), so close() gives QUIT ENGINE_DISK_STOP_WAIT_S more when the engine runs with it."""
import io
import unittest

from serve import server
from serve.server import conversation_disk_dir


class Engine:
    def __init__(self, args, cwd="/srv/strata"):
        self.spawn = ("/bin/strata", args, cwd, "strata.log", {}, False, None)


class StopWait(unittest.TestCase):
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
        eng = object.__new__(server.StrataEngine)
        eng.proc, eng.spawn, eng.pump, eng.log = self.Proc(), ("/bin/strata", args, "/srv", "x.log", {}, False, None), None, None
        proc = eng.proc
        eng.close()
        return proc.waits[0]

    def test_quit_waits_longer_with_the_disk_level(self):
        self.assertEqual(self.close_wait(["--conversation-cache-disk", "/data/cc"]),
                         server.ENGINE_QUIT_WAIT_S + server.ENGINE_DISK_STOP_WAIT_S)
        self.assertEqual(self.close_wait(["--prompt-cache", "12"]), server.ENGINE_QUIT_WAIT_S)
        self.assertLess(server.ENGINE_QUIT_WAIT_S + server.ENGINE_DISK_STOP_WAIT_S + server.ENGINE_TERM_WAIT_S, 60)

    def test_the_folder_comes_from_the_engine_arguments(self):
        self.assertEqual(conversation_disk_dir(Engine(["--conversation-cache-disk", "/data/cc"])), "/data/cc")
        self.assertEqual(conversation_disk_dir(Engine(["--conversation-cache-disk", "cc"])), "/srv/strata/cc")
        self.assertIsNone(conversation_disk_dir(Engine(["--prompt-cache", "12"])))
        self.assertIsNone(conversation_disk_dir(Engine(["--conversation-cache-disk"])))   # no value: not a folder
        self.assertIsNone(conversation_disk_dir(object()))


if __name__ == "__main__":
    unittest.main()
