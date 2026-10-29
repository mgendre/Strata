"""The "prompt_heads" diagnostic: one text file per distinct system turn, an index that counts them, nothing for a
prompt without a system turn, and the cap on the files."""
import json
import tempfile
import unittest
from pathlib import Path

from serve.server import PromptHeads

T, SYS, USER, END = PromptHeads.TURN_ID, 7, 8, 9


class Tok:
    def encode(self, text, parse_special=False):
        return [SYS] if text == "system" else [ord(c) for c in text]

    def decode(self, ids):
        return " ".join(map(str, ids))


def prompt(system, user):
    return [T, SYS, *system, END, T, USER, *user, END, T]


class Heads(unittest.TestCase):
    def test_distinct_system_turns_are_kept_once_and_counted(self):
        with tempfile.TemporaryDirectory() as d:
            h = PromptHeads(d, Tok())
            a = h.note(prompt([1, 2, 3], [4]))
            self.assertEqual(h.note(prompt([1, 2, 3], [5, 6])), a)       # another chat, the same system turn
            b = h.note(prompt([1, 2, 4], [4]))                          # one token changed: another one
            self.assertNotEqual(a, b)
            index = json.loads((Path(d) / "index.json").read_text())
            self.assertEqual((index[a]["count"], index[a]["tokens"], index[b]["count"]), (2, 6, 1))
            self.assertEqual((Path(d) / index[a]["file"]).read_text(), f"{T} {SYS} 1 2 3 {END}")
            self.assertEqual(PromptHeads(d, Tok()).index.keys(), index.keys())   # kept across a restart

    def test_a_prompt_without_a_system_turn_is_not_counted(self):
        with tempfile.TemporaryDirectory() as d:
            h = PromptHeads(d, Tok())
            self.assertIsNone(h.note([T, USER, 1, 2, END, T]))
            self.assertIsNone(h.note([T, SYS, 1, 2]))                   # no turn after it
            self.assertEqual(h.index, {})

    def test_past_the_cap_the_index_still_counts(self):
        with tempfile.TemporaryDirectory() as d:
            h = PromptHeads(d, Tok(), cap=1)
            a, b = h.note(prompt([1], [2])), h.note(prompt([3], [2]))
            self.assertTrue(h.index[a]["file"] and h.index[b]["file"] is None)
            self.assertEqual(len(list(Path(d).glob("*.txt"))), 1)


if __name__ == "__main__":
    unittest.main()
