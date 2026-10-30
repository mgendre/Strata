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


class Divergences(unittest.TestCase):
    def setUp(self):
        self.saved = (PromptHeads.MIN_TOKENS, PromptHeads.SLACK, PromptHeads.WINDOW)
        PromptHeads.MIN_TOKENS, PromptHeads.SLACK, PromptHeads.WINDOW = 10, 4, 4   # small prompts for the test

    def tearDown(self):
        PromptHeads.MIN_TOKENS, PromptHeads.SLACK, PromptHeads.WINDOW = self.saved

    def test_a_history_rewritten_early_is_written_down(self):
        with tempfile.TemporaryDirectory() as d:
            h = PromptHeads(d, Tok())
            first = list(range(100, 140))
            h.note(prompt([1, 2, 3], first))                          # (the system turn: T SYS 1 2 3 END, 6 ids)
            h.note(prompt([1, 2, 3], first) + [7, 7])                 # the same conversation going on
            h.note(prompt([1, 2, 3], [500] * 40))                     # another chat of the same agent
            self.assertFalse((Path(d) / "divergences").exists())
            self.assertFalse((Path(d) / "tail-rewrites").exists())
            changed = first[:5] + [999] + first[6:]                  # the client rewrote id 5 of the first message
            h.note(prompt([1, 2, 3], changed + [7, 7]))
            rows = [json.loads(l) for l in (Path(d) / "divergences" / "index.jsonl").read_text().splitlines()]
            self.assertEqual([(r["common"], r["head"], r["recent_tokens"]) for r in rows], [(13, 6, 52)])
            text = (Path(d) / "divergences" / rows[0]["file"]).read_text()
            self.assertIn("parted from a recent one of 52 at id 13", text)
            self.assertIn(" 104 999 106 ", text)                      # this prompt's text around the change
            self.assertIn(" 104 105 106 ", text)                      # the recent prompt's

    def test_a_change_at_the_very_start_of_the_first_message_is_written_down_too(self):
        with tempfile.TemporaryDirectory() as d:
            h = PromptHeads(d, Tok())
            history = list(range(100, 160))
            h.note(prompt([1, 2, 3], history))
            h.note(prompt([1, 2, 3], [999] + history[1:] + [7, 7]))   # id 0 of the first message rewritten
            rows = [json.loads(l) for l in (Path(d) / "divergences" / "index.jsonl").read_text().splitlines()]
            self.assertEqual([r["common"] for r in rows], [8])         # (the system turn 6 + <|im_start|> user)

    def test_a_rewrite_that_shrinks_the_prompt_is_written_down(self):
        with tempfile.TemporaryDirectory() as d:
            h = PromptHeads(d, Tok())
            PromptHeads.MIN_TOKENS = 40
            tool, later = list(range(100, 140)), list(range(200, 230))   # an old tool output, then the later turns
            h.note(prompt([1, 2, 3], tool + later))                   # 80 ids: kept to compare with
            h.note(prompt([1, 2, 3], [5] + later))                    # the tool output cleared: 41 ids
            rows = [json.loads(l) for l in (Path(d) / "divergences" / "index.jsonl").read_text().splitlines()]
            self.assertEqual([(r["tokens"], r["recent_tokens"], r["common"]) for r in rows], [(41, 80, 8)])

    def test_a_re_rendered_end_of_the_previous_prompt_is_written_down(self):
        with tempfile.TemporaryDirectory() as d:
            h = PromptHeads(d, Tok())
            history = list(range(100, 140))
            h.note(prompt([1, 2, 3], history + [50, 51]))             # a tool output, then a reminder (50 51)
            h.note(prompt([1, 2, 3], history + [60, 7, 7]))           # the reminder left out, the reply, a new turn
            self.assertFalse((Path(d) / "divergences").exists())      # (not an early rewrite)
            rows = [json.loads(l) for l in (Path(d) / "tail-rewrites" / "index.jsonl").read_text().splitlines()]
            self.assertEqual([(r["common"], r["back"], r["recent_tokens"]) for r in rows], [(48, 4, 52)])
            text = (Path(d) / "tail-rewrites" / rows[0]["file"]).read_text()
            self.assertIn("up to id 48, 4 ids before its end", text)
            self.assertIn(f"===== the previous prompt, from the change to its end =====\n50 51 {END} {T}\n", text)
            self.assertIn(f"===== this prompt, from the change =====\n60 7 7 {END} {T}\n", text)

    def test_past_the_tail_cap_the_index_still_counts(self):
        with tempfile.TemporaryDirectory() as d:
            h = PromptHeads(d, Tok())
            PromptHeads.TAIL_CAP, saved = 1, PromptHeads.TAIL_CAP
            try:
                history = list(range(100, 140))
                for k in range(3):                                    # each turn re-renders the previous end
                    h.note(prompt([1, 2, 3], history + [50 + k]))
                    history += [60 + k, 7]
            finally:
                PromptHeads.TAIL_CAP = saved
            rows = [json.loads(l) for l in (Path(d) / "tail-rewrites" / "index.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(list((Path(d) / "tail-rewrites").glob("*.txt"))), 1)

    def test_a_new_subagent_on_the_same_task_is_not_a_rewrite(self):
        with tempfile.TemporaryDirectory() as d:
            h = PromptHeads(d, Tok())
            task = list(range(100, 120))                              # "You are a subagent ... Implement <plan>"
            h.note(prompt([1, 2, 3], task + list(range(300, 360))))   # the first subagent's long session
            h.note(prompt([1, 2, 3], task + list(range(600, 620))))   # a new one: the same task, another report
            self.assertFalse((Path(d) / "divergences").exists())

    def test_kept_later(self):
        import array
        b = lambda xs: array.array("i", xs).tobytes()
        old = b(list(range(0, 200)))
        self.assertTrue(PromptHeads.kept_later(old, b([5] + list(range(1, 200)) + [9]), 0, width=8))
        self.assertFalse(PromptHeads.kept_later(old, b([5] * 200), 0, width=8))

    def test_common_prefix(self):
        import array
        a = array.array("i", [1, 2, 3, 4, 5]).tobytes()
        b = array.array("i", [1, 2, 3, 9]).tobytes()
        self.assertEqual(PromptHeads.common_prefix(a, b), 3)
        self.assertEqual(PromptHeads.common_prefix(a, a), 5)
        self.assertEqual(PromptHeads.common_prefix(a, b""), 0)


if __name__ == "__main__":
    unittest.main()
