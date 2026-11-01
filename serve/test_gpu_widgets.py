"""The Monitor's GPU widgets (one per card, under the tiles): each card's own readings and its link's top speed from the
sampler, a short history per card, the label of what each card does, and the widgets on the page."""
import time
import unittest
from pathlib import Path
from unittest import mock

from serve import telemetry
from serve.server import gpu_card_roles


class Reader:
    """A card as gpu_reader gives it: NVML's readings of an RTX 3080 Ti on Gen4 x16 or an RTX 3060 on Gen4 x4, idle at Gen1."""
    def __init__(self, index):
        self.index = index

    def ok(self):
        return True

    def name(self):
        return {0: "NVIDIA GeForce RTX 3060", 1: "NVIDIA GeForce RTX 3080 Ti"}[self.index]

    def read(self):
        if self.index == 1:
            return {"util": 80, "mem_used": 11 << 30, "mem_total": 12 << 30, "temp": 61, "power": 287.5,
                    "power_limit": 320.0, "pcie_gen": 4, "pcie_gen_max": 4, "pcie_width": 16, "pcie_rx_mb": 9000.0,
                    "pcie_tx_mb": 300.0}
        return {"util": 40, "mem_used": 10 << 30, "mem_total": 12 << 30, "temp": 48, "power": 95.0, "power_limit": 170.0,
                "pcie_gen": 1, "pcie_gen_max": 4, "pcie_width": 4, "pcie_rx_mb": 1000.0, "pcie_tx_mb": 2000.0}


class GpuCards(unittest.TestCase):
    def test_link_speed(self):
        self.assertAlmostEqual(telemetry.pcie_link_mb(4, 16), 16e9 * 128 / 130 / 8 * 16 / 2**20)   # ~30,040 MB/s
        self.assertAlmostEqual(telemetry.pcie_link_mb(4, 4) * 4, telemetry.pcie_link_mb(4, 16))
        self.assertAlmostEqual(telemetry.pcie_link_mb(1, 1), 250e6 / 2**20)
        for gen, width in ((None, 16), (4, None), (4, 0), (9, 16), ("4", 16)):
            self.assertIsNone(telemetry.pcie_link_mb(gen, width))

    def test_card_uses_the_top_generation(self):
        idle = telemetry.gpu_card(0, "RTX 3060", Reader(0).read())   # an idle link at Gen1 still carries Gen4's
        self.assertEqual((idle["index"], idle["name"], idle["pcie_gen"]), (0, "RTX 3060", 1))
        self.assertAlmostEqual(idle["pcie_link_mb"], telemetry.pcie_link_mb(4, 4))
        none = telemetry.gpu_card(2, None, {})                         # a Mac's: nothing to read, nothing breaks
        self.assertIsNone(none["pcie_link_mb"])
        self.assertIsNone(none["util"])

    def test_sampler_keeps_each_card_and_its_history(self):
        with mock.patch.object(telemetry, "gpu_reader", lambda index=0, amd=False: Reader(index)):
            t = telemetry.Telemetry(gpu_index=1, gpu_indices=[1, 0])
        try:
            s = t.sample()
            self.assertEqual(s["gpu_mem_used"], 21 << 30)   # the tiles keep their totals
            cards = s["gpu_cards"]
            self.assertEqual([c["index"] for c in cards], [1, 0])
            self.assertEqual([c["name"] for c in cards], ["NVIDIA GeForce RTX 3080 Ti", "NVIDIA GeForce RTX 3060"])
            self.assertEqual((cards[0]["util"], cards[0]["pcie_rx_mb"], cards[1]["pcie_tx_mb"]), (80, 9000.0, 2000.0))
            self.assertAlmostEqual(cards[1]["pcie_link_mb"], telemetry.pcie_link_mb(4, 4))
            deadline = time.monotonic() + 5.0
            while "gpu0_pcie_tx_mb" not in t.snapshot()["history"] and time.monotonic() < deadline:
                time.sleep(0.05)
            hist = t.snapshot()["history"]
            self.assertEqual((hist["gpu1_util"][-1], hist["gpu0_pcie_tx_mb"][-1], hist["gpu1_pcie_rx_mb"][-1]),
                             (80, 2000.0, 9000.0))
            self.assertNotIn("gpu1_mem_used", hist)
        finally:
            t.close()

    def test_one_card(self):
        with mock.patch.object(telemetry, "gpu_reader", lambda index=0, amd=False: Reader(index)):
            t = telemetry.Telemetry(gpu_index=1)
        try:
            s = t.sample()
            self.assertEqual([c["index"] for c in s["gpu_cards"]], [1])
            self.assertNotIn("gpus", s)
        finally:
            t.close()


class Roles(unittest.TestCase):
    PEER = {"gpu": [1, 0], "args": ["--peer-device", "1", "--max-context", "131072"]}

    def test_peer_device(self):
        # CUDA_VISIBLE_DEVICES=1,0 in bus order: the engine's card 1 is nvidia-smi's 0
        self.assertEqual(gpu_card_roles(self.PEER, [1, 0], 1, [1, 0]), {"1": "model", "0": "peer experts"})

    def test_peer_device_without_a_known_order(self):
        self.assertEqual(gpu_card_roles(self.PEER, [], 1, [1, 0]), {"1": "model"})
        bad = {"gpu": [1, 0], "args": ["--peer-device", "x"]}
        self.assertEqual(gpu_card_roles(bad, [1, 0], 1, [1, 0]), {"1": "model"})
        far = {"gpu": [1, 0], "args": ["--peer-device", "5"]}
        self.assertEqual(gpu_card_roles(far, [1, 0], 1, [1, 0]), {"1": "model"})

    def test_layer_split_and_one_card(self):
        self.assertEqual(gpu_card_roles({"gpu": [0, 1], "args": []}, [0, 1], 0, [0, 1]),
                         {"0": "layer split", "1": "layer split"})
        self.assertEqual(gpu_card_roles({"gpu": 0, "args": []}, [0], 0, [0]), {"0": "model"})


class Page(unittest.TestCase):
    def test_the_widgets_are_on_the_page(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="gpu-cards"', html)
        self.assertLess(html.index('id="metrics"'), html.index('id="gpu-cards"'))
        self.assertLess(html.index('id="gpu-cards"'), html.index('id="cc-card"'))
        self.assertIn('$("gpu-cards")', js)
        self.assertIn("renderGpuCards(hw.gpu_cards, h, st.gpu_roles)", js)


if __name__ == "__main__":
    unittest.main()
