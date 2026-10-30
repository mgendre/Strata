# Strata Experimental

> A fork of **[Strata](https://github.com/Niko1221/Strata)**, created by **[Niko1221](https://github.com/Niko1221)**.
> Strata itself (its engine and apps) is their work; this fork only adds the changes listed below.
> The [original README](#the-original-strata-readme) follows, unchanged.

The branch `strata-experimental` is Strata
plus the changes below; `main` follows upstream. Every new option is off by default: without them, Strata behaves as
upstream does, apart from the few fixes marked "always".

**The goal.** Agent tools (OpenCode, Claude Code, subagents) switch between several long conversations and start many
new chats with the same system prompt. Strata keeps a few conversations in RAM; one that leaves RAM, or a new chat of
an agent whose prelude is gone, reads its whole prompt again: seconds to tens of seconds. This fork keeps those on an
SSD without writing during normal work, makes parked conversations smaller in RAM, and speeds up a second GPU on a
slow link.

## What changes compared with Strata

| Change | Option |
| --- | --- |
| **Conversation cache on disk.** A long conversation that leaves RAM is written to an SSD and comes back from there instead of being read again. Nothing is written while it is in RAM. | `--conversation-cache-disk DIR`; its size: `--conversation-cache-disk-mib N`, `--conversation-cache-disk-min-tokens N` |
| **Prelude cache.** Each agent's system prompt and tools are saved once; a new chat of that agent starts after them, also after a restart. | `--prelude-cache DIR`; its size: `--prelude-cache-mib N`, `--prompt-cache-root N` |
| **Smaller parked conversations.** A conversation parked in RAM keeps only its last checkpoints and its first one. | `--conversation-cache-park-checkpoints N` |
| **Second GPU refill by SM loads.** With `--peer-device`, the experts copied into the second GPU no longer hold back that card's work. | `--peer-refill-sm N` |
| **Image encoder on the CPU, started on demand.** `"elastic": true` now works for a CPU encoder, also with `--peer-device`. | `"elastic": true` (config, `vision` section) |
| **No expert on both GPUs.** The main GPU no longer takes an expert that is being copied into the second GPU. | always, with `--peer-device` |
| **Less waiting for a second GPU on a slow PCIe link.** At every layer of every token, the PC sends the second GPU its share of the work and waits for the results. The package sent is now 3.5 times smaller (converted by the CPU before it leaves), and the results come back while the card computes instead of in a separate copy afterwards. The same results, bit for bit; decode about 4% faster with the reference PC's RTX 3060 on PCIe 4.0 x4. | `--peer-slow-link` (with `--peer-device`) |
| **Second GPU line.** One log line per request: where the experts ran (main GPU, second GPU, PCIe, CPU) and how long the CPU waited for the second GPU. | always, with `--peer-device` |
| **Prompt reads skip the experts no token uses** (taken from upstream PR [#4283](https://github.com/Niko1221/Strata/pull/4283), by [@bitaccesscomau](https://github.com/bitaccesscomau); see below). A prompt chunk of 1,024 tokens or more no longer copies over PCIe the experts no token of the chunk routes to. The same bits. | `STRATA_PREFILL_ROUTED_STREAM=1` (env) |
| **Monitor.** A widget per GPU under the tiles: its role (model, peer experts for the second GPU, layer split), load, VRAM, temperature, power, and PCIe traffic each way as a share of its link (Gen4 x16 about 30 GB/s each way, x4 about 7.5). The Conversation cache card has a small graph per store on disk (conversations, preludes): the prompt tokens it gave back each hour of the last day, and a line of what it wrote. A Disk write card sits next to Disk read, with what the disks took over the last 24 hours. | always |

## Taken from upstream pull requests

Open pull requests of [Niko1221/Strata](https://github.com/Niko1221/Strata/pulls) that this fork carries before they
are merged upstream. Each keeps its author's commit; credit goes to them.

| Upstream PR | Author | What it does | Here |
| --- | --- | --- | --- |
| [#4283](https://github.com/Niko1221/Strata/pull/4283) | [@bitaccesscomau](https://github.com/bitaccesscomau) | `STRATA_PREFILL_ROUTED_STREAM=1` (opt-in): a streamed prompt chunk publishes each layer's routing, and the copies not issued yet of the experts no token routes to are dropped. Routed experts go to the same ring slots, so the results are the same bits. | On in the reference config. Measured on agent follow-ups (after about 25K reused tokens): reads of 1-2K new tokens +16%, 2-4K +4%, 4-8K +6%. One conflict with upstream #4005 resolved: the switch stays off with `STRATA_PREFILL_COPY_EVENT_EVERY` > 1 or `STRATA_PREFILL_COPY_WAIT_DEDUPE`. |

## The settings

Numbers from the reference PC: Linux, Ryzen 7 5800X3D, 64 GB DDR4-3600, RTX 3080 Ti + RTX 3060 behind the X570
chipset (PCIe 4.0 x4), a PCIe 4.0 NVMe SSD, Qwen3.8-Flash-Next IQ3\_S. Other PCs will see other numbers.

**`--conversation-cache-disk DIR`** (off). The folder of the conversation cache on disk; put it on an SSD.
- Sizes, set with it (each one below): `--conversation-cache-disk-mib N`, the space it may take (32 GiB by default);
  `--conversation-cache-disk-min-tokens N`, the shortest conversation it writes (12,288 tokens); and
  `--conversation-cache-disk-day-gib N`, the daily writes past which the log warns (200 GiB).
- When: a conversation of `--conversation-cache-disk-min-tokens` or more is written when it leaves RAM (the slots or
  the budget are full, RAM is low) or when RAM cannot take it. Only new 4K-token blocks are written; a copy the disk
  already holds (all but its last block) or an older copy of a conversation still in RAM is not written.
- Gain: the conversation comes back in 0.2-0.7 s (14K-84K tokens); a 13.6K-token one read its prompt in 0.96 s
  instead of 6.8 s.
- Cost: the request that pushed it out waits for the write (0.26 s, 310 MiB for 13.6K tokens), and disk space.
- Use it when agents switch between more long conversations than the RAM cache holds (`--conversation-cache-slots`),
  or when RAM is short. With a single long conversation it writes nothing.
- Values: a folder on an SSD (an NVMe one is best; a hard disk is too slow to beat reading the prompt again). Not set:
  off. Reference PC: a folder on its PCIe 4.0 NVMe SSD.

**`--conversation-cache-disk-mib N`** (32768). The disk budget. A conversation takes about 16-18 KB a token (2.5 GB at
150K tokens). When it is full, the copy that came back least often leaves first, then the oldest.
- Values: in MiB, the disk space you give it. 16384 (16 GiB) holds about six 150K-token conversations or twenty-five
  40K ones; 32768 (the default) twice that. Reference PC: 65536 (64 GiB).

**`--conversation-cache-disk-min-tokens N`** (12288). Shorter conversations are not written: reading them again costs
a few seconds at most.
- Values: in tokens. Lower writes more, shorter conversations (below about 4096 it saves nothing: the disk works in
  4,096-token blocks); higher writes only the long ones. On the reference PC 12,288 tokens read again in about 6 s.
  Reference PC: 12288.

**`--conversation-cache-disk-day-gib N`** (200). A warning in the log past N GiB written in a day; nothing stops.
- Values: in GiB a day. Pick what your SSD's endurance allows: its TBW divided by the days you want it to last (a
  600 TBW drive over 5 years: about 330). The reference PC wrote 13-18 GiB a day with agents at work.

**`--prelude-cache DIR`** (off). The folder of the prelude cache; put it on an SSD. The prelude is an agent's system
turn (system prompt and tools). It is saved once, right after the first reply that read it; a new chat that starts
with it, and that RAM cannot resume further, starts after it.
- Sizes, set with it: `--prelude-cache-mib N`, the space it may take (5 GiB by default); `--prompt-cache-root N`, the
  shortest prelude it saves (2,048 tokens, an upstream option that also sets the root checkpoint); and
  `--conversation-cache-disk-day-gib N`, the daily writes past which the log warns (200 GiB, the same value as the disk tier's, counted per store).
- Gain: an 8,413-token prelude came back in 0.19 s; the request read its prompt in 0.56 s instead of 7.4 s. RAM
  already does this while one of the agent's conversations is parked; the prelude cache also covers the time after
  they left RAM and after a restart.
- Cost: one write per distinct system prompt (235 MiB for 8K tokens, 0.2 s). An agent whose system prompt changes
  every session (a date, a folder) gets a new prelude each time; the least used leaves first when the budget is full.
- The server passes the id of `system` to the engine (`--system-role-token`) by itself: nothing to set.
- Values: `--prelude-cache` a folder on an SSD (not set: off). `--prelude-cache-mib` in MiB: a prelude takes about
  28 KB a token (245 MiB for 9,116 tokens), so 5120 (the default) holds about 20 preludes of 8-9K tokens; when it is
  full, the least used leaves first. `--prompt-cache-root` in tokens: system prompts shorter than that are cheap to
  read again. Reference PC: 10240 and 2048.

**`--conversation-cache-park-checkpoints N`** (0: all). A conversation parked in RAM keeps its last N checkpoints and
its first one (the end of the system prompt), instead of all of them; each is about 118 MB on Qwen3.8-Flash-Next.
- Gain: more conversations fit in `--conversation-cache-mib` (9 checkpoints: 807 MB with N = 3 instead of 1.4 GB).
- Cost: a request that goes back to an older point of a parked conversation reads more again. 3 is a good start.
- Values: 0 (the default) keeps every checkpoint; 1 or more keeps that many of the last ones, plus the first. Each one
  is about 118 MB. 2-3 for agents, which mostly continue from the end; more if you often edit or regenerate an older
  message. Reference PC: 3.

**`--peer-refill-sm N`** (0: the copy engine). With `--peer-device`, the adaptive tier copies experts into the second
GPU with N blocks of that GPU's own loads. On the copy engine, the small copies that start each layer's work on that
card waited behind the whole round (0.11 ms alone, 23 ms during a round of 72 copies).
- Gain: decode 91.7 -> 97.0 tok/s; with `--adapt-decay 0.92 --peer-adapt-swaps 144` on top, 98.9 tok/s.
- Cost: none measured (a layer's work there took 0.12 ms instead of 0.11 ms during a round). 4 fill a PCIe 4.0 x4 link.
- Use it with a second GPU on a chipset or x4 slot; measured only there.
- Values: 0 (the default) keeps the copy engine; N is the number of blocks of the second GPU's cores that do the
  copies. 4 reach the full speed of a PCIe 4.0 x4 link (6.0 GB/s), so more would only take cores from the experts. A
  wider link (x8, x16) may need more blocks; not measured. Reference PC: 4.

**`--peer-slow-link`** (off). With `--peer-device`: less waiting for a second GPU on a slow PCIe link. In plain
words: at every layer of every token (48 layers per token), the PC sends the second GPU the data for its share of the
work, the card computes, and it sends the results back; the CPU waits for them before the layer can finish. Two things
changed: the data is converted to the card's compact format by the CPU before it is sent (3.5 times fewer bytes on
the link), and the results are written straight into the PC's RAM as they are computed (no separate copy at the end).
The details: a second GPU computes its share of every layer's experts, and the CPU waits for its results before the
layer can finish. When that
card sits in a chipset or x4 slot (the reference PC's RTX 3060: PCIe 4.0 x4, through the X570 chipset), the link is
narrow, so the input going in and the rows coming back are a real part of that wait, at every layer of every token.
Its share used to be copied in as floats, quantized there, computed into its memory, then scattered to RAM over its
link after the experts (x4: ~27 us a layer). Now its down kernel writes each entry's rows straight into RAM as one
128-byte store while the other blocks still compute, and the host quantizes the input to q8_1 the way the device does
(AVX2 + F16C; else the device's), so 3.5x fewer bytes go in.
- When it helps: a second GPU on a narrow or shared link (x4, x1, a chipset slot). On an x8 or x16 slot, or NVLink,
  the transfers are shorter and so is the gain; not measured.
- Gain: per layer 164-173 -> ~136 us, the CPU's wait for the second GPU 82-84 -> ~47 us; decode windows 17.2 -> 16.5
  ms (about 4% at short context, about 3% at long context).
- The same bits: `STRATA_PEER_CHECK=1` also runs the old path and compares every row (880,771 rows, 0 different).
- `STRATA_PEER_TIMING=1` logs, per request, where a layer's call on the second GPU goes (copies, quantization,
  experts, the CPU's wait).
- Values: a switch, no number. Add `--peer-slow-link` to turn it on; without it the second GPU works as upstream
  (floats in, rows copied back after its experts). Turn it on for a second GPU on a narrow or shared link; on a fast
  one it is harmless, but untested. It is the only setting: both
  halves (the smaller input, the results straight to RAM) come with it, and both give the same results as without it,
  so there is nothing to choose between them. Reference PC: on. The direct rows apply to two down-row
  formats at Qwen3.8-Flash-Next's expert width (IQ4_NL among them, the reference pack's); other formats keep the old
  path by themselves. `STRATA_PEER_CHECK=1` and `STRATA_PEER_TIMING=1` are diagnostics: off in normal use.

**`"elastic": true`** and **`"elastic_idle_s"`** (120) in the config's `vision` section. The image encoder starts with
the first picture and stops after that many seconds without one. Gain: its RAM (about 1 GB on the CPU) is free between
pictures. Cost: the first picture after a stop takes 2.0 s instead of 0.2 s.
- Values: `"elastic"` true or false (false: the encoder stays loaded); `"elastic_idle_s"` in seconds. Shorter frees the
  RAM sooner but restarts the encoder more often: pictures that come in bursts want 60-120 s. Reference PC: 60.

**`STRATA_PREFILL_ROUTED_STREAM=1`** (env, off) and **`STRATA_PREFILL_ROUTED_STREAM_DEPTH`** (16). From upstream PR
[#4283](https://github.com/Niko1221/Strata/pull/4283) (see above). A prompt chunk of 1,024 tokens or more no longer copies
over PCIe the experts no token of the chunk routes to; the results are the same bits.
- Gain: agent follow-ups (after about 25K reused tokens) read 1-2K new tokens 16% faster, 2-4K 4%, 4-8K 6%.
- Cost: none measured on decode.
- Values: `STRATA_PREFILL_ROUTED_STREAM` 1 (on) or 0 / not set (off). `..._DEPTH`: the copies kept queued ahead of the
  PCIe link, 1 to 256; the default, 16, was the only value measured (here and upstream). Off by itself with a prompt
  share on a second GPU (`--peer-prefill-rows` other than 0), and with upstream's `STRATA_PREFILL_COPY_EVENT_EVERY` > 1
  or `STRATA_PREFILL_COPY_WAIT_DEDUPE`. Reference PC: on, depth 16.

All of them on the reference PC (in the run config's `args`, and the `vision` section):

```json
"--conversation-cache-disk", "/path/to/conversation-cache",
"--conversation-cache-disk-mib", "65536",
"--conversation-cache-disk-min-tokens", "12288",
"--conversation-cache-park-checkpoints", "3",
"--prelude-cache", "/path/to/prelude-cache",
"--prelude-cache-mib", "10240",
"--peer-refill-sm", "4",
"--peer-slow-link"
```

```json
"vision": {"elastic": true, "elastic_idle_s": 60}
```

```json
"env": {"STRATA_PREFILL_ROUTED_STREAM": "1"}
```

## Settings that empty the disk stores

Both stores are kept across restarts and rebuilds of the same engine version with the same model and the settings that
change what the saved bytes mean. Changing one of these makes the next start drop every copy (they fill again as they
are used): the engine version, the model, `--kv`, `--max-context`, `--kv-resident`, the draft layer's window, rope
scaling, control vectors, `--vision`, and the experimental `--native-*` and `--no-*` switches. A layer split
(`--layer-split`) keeps both stores off.

## Upstream settings tuned with these changes (reference PC)

These options come from upstream Strata; each was measured on the reference PC with the fork:

| Setting | What it changed |
| --- | --- |
| `--kv-resident 20480` | The K/V in pinned RAM with 20,480 cells per layer in VRAM: the RTX 3080 Ti holds about 600 more experts, decode about +2.5%. Costs 1.81 GiB of pinned RAM at 150K tokens of context. |
| `--vram-reserve-mib 444` | Since 0.2.8, `--expert-cache auto` leaves the reserve plus 256 MiB for what loads after the cache; 444 keeps the 0.2.7 cache (2,511 experts instead of 2,385, about 1.5% of decode). Fine on Linux, where a nearly full card does not page to system RAM; keep the default on Windows. |
| `STRATA_PARK_ASYNC=1`, `STRATA_CKPT_ASYNC=1`, `STRATA_CKPT_GATHER=1`, `STRATA_CKPT_POOL=1`, `STRATA_CKPT_POOL_READY=4` (env) | Parks and checkpoints off the request's path: two agents taking turns, a turn 1,542 -> 1,179 ms, a park 236 -> 7 ms. Costs 4 x 118 MB of RAM held from start. |
| `--adapt-decay 0.92 --peer-adapt-swaps 144` | With `--peer-refill-sm`, the second GPU follows the text more closely (about +2%). Without it, slower. |
| `--conversation-cache-min-tokens 2048` | With parks this cheap, 2K-6K-token conversations are parked too (a next turn 0.37 s instead of 2.4 s). Not 0: short side requests would take the slots. |
| `--peer-reserve-mib 280 --peer-prefill-rows 0` | The second GPU (RTX 3060, headless) keeps 280 MiB free instead of 600 and holds 166 more experts (5,856 instead of 5,690). `--peer-prefill-rows 0` keeps the prompt path off that card (its share did not fit anyway) and skips a cuBLAS handle there. 128 MiB made the engine fail at start (`cublasCreate`); keep 600 if a desktop runs on that card. |
| `--max-context 153600` | 150K tokens of context (the model was trained on 262K). Raise the agent's own context limit too. |

Not measured yet: AMD (HIP), Intel (SYCL), Mac (Metal) and Windows.

---

# The original Strata README

<sub>From [Niko1221/Strata](https://github.com/Niko1221/Strata), by Niko1221, kept as it is.</sub>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#install"><b>Install</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#help">Help</a> · <a href="#support-strata">Support</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Download Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

## Models

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) for coding. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

In the **desktop app** and the **Terminal**. What does not fit in memory is read from your SSD, so it needs a lot of free
disk space.

| Set (IST-DAS) | Download | Free disk needed |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | about 245 GB |
| 2.00 bpw | 257 GB | about 265 GB |
| 2.25 bpw | 274 GB | about 280 GB |

Runs on NVIDIA RTX 30, 40 and 50 series (Windows and Linux), AMD Radeon AI PRO R9700 (Linux), AMD Ryzen AI Max — Strix Halo
(Windows), and Apple Silicon Macs with 64 GB or more, text only ([Mac](docs/DEEPSEEK_MAC.md)). Setup downloads the files and the DeepSeek engine, then starts the model.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Measured with the 2.00 bpw set on Strata 0.2.5 (Strix Halo: 0.2.7). Output is the speed of the answer, prompt is how fast Strata reads your input.

| Hardware | Output | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4 to 6 tokens/s | about 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9 to 11 tokens/s | about 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21 to 25 tokens/s | 320 to 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 to 77 tokens/s | 400 to 450 tokens/s (8K) |
| Ryzen AI Max+ 395 (Strix Halo), 128 GB unified | 19 to 21 tokens/s (code), 11 (chat) | 87 tokens/s (4K), 125 tokens/s (8K) |

[Memory, disk and details](docs/DEEPSEEK41.md) · [Release notes](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.8-windows-x64-setup.exe` · `Strata-0.2.8-linux-x86_64.AppImage` · `Strata-0.2.8-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.

<details open>
<summary><b>Install commands</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |
| **NVIDIA DGX Spark · Linux** | [DGX Spark setup](docs/DGX_SPARK_LINUX.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A coffee helps fund development and testing on more hardware.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

