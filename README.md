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

## Strata Experimental (this fork)

This is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata). The branch `strata-experimental` is Strata
plus the changes below; `main` follows upstream. Every new option is off by default: without them, Strata behaves as
upstream does, apart from the few fixes marked "always".

**The goal.** Agent tools (OpenCode, Claude Code, subagents) switch between several long conversations and start many
new chats with the same system prompt. Strata keeps a few conversations in RAM; one that leaves RAM, or a new chat of
an agent whose prelude is gone, reads its whole prompt again: seconds to tens of seconds. This fork keeps those on an
SSD without writing during normal work, makes parked conversations smaller in RAM, and speeds up a second GPU on a
slow link.

### What changes compared with Strata

| Change | Option |
| --- | --- |
| **Conversation cache on disk.** A long conversation that leaves RAM is written to an SSD and comes back from there instead of being read again. Nothing is written while it is in RAM. | `--conversation-cache-disk DIR` |
| **Prelude cache.** Each agent's system prompt and tools are saved once; a new chat of that agent starts after them, also after a restart. | `--prelude-cache DIR` |
| **Smaller parked conversations.** A conversation parked in RAM keeps only its last checkpoints and its first one. | `--conversation-cache-park-checkpoints N` |
| **Second GPU refill by SM loads.** With `--peer-device`, the experts copied into the second GPU no longer hold back that card's work. | `--peer-refill-sm N` |
| **Image encoder on the CPU, started on demand.** `"elastic": true` now works for a CPU encoder, also with `--peer-device`. | `"elastic": true` (config, `vision` section) |
| **No expert on both GPUs.** The main GPU no longer takes an expert that is being copied into the second GPU. | always, with `--peer-device` |
| **Second GPU rows straight to RAM, input quantized on the host.** The second GPU writes its expert rows into RAM as whole 128-byte lines while it computes, and gets its input already quantized (3.5x fewer bytes). The same bits; decode windows about 4% shorter. | always, with `--peer-device` (`STRATA_PEER_DOWN_HOST=0` / `STRATA_PEER_HOST_Q8=0` turn either off) |
| **Second GPU line.** One log line per request: where the experts ran (main GPU, second GPU, PCIe, CPU) and how long the CPU waited for the second GPU. | always, with `--peer-device` |
| **Monitor.** The Conversation cache card has a small graph per store on disk (conversations, preludes): the prompt tokens it gave back each hour of the last day, and a line of what it wrote. A Disk write card sits next to Disk read. | always |

### The settings

Numbers from the reference PC: Linux, Ryzen 7 5800X3D, 64 GB DDR4-3600, RTX 3080 Ti + RTX 3060 behind the X570
chipset (PCIe 4.0 x4), a PCIe 4.0 NVMe SSD, Qwen3.8-Flash-Next IQ3\_S. Other PCs will see other numbers.

**`--conversation-cache-disk DIR`** (off). The folder of the conversation cache on disk; put it on an SSD.
- When: a conversation of `--conversation-cache-disk-min-tokens` or more is written when it leaves RAM (the slots or
  the budget are full, RAM is low) or when RAM cannot take it. Only new 4K-token blocks are written; a copy the disk
  already holds (all but its last block) or an older copy of a conversation still in RAM is not written.
- Gain: the conversation comes back in 0.2-0.7 s (14K-84K tokens); a 13.6K-token one read its prompt in 0.96 s
  instead of 6.8 s.
- Cost: the request that pushed it out waits for the write (0.26 s, 310 MiB for 13.6K tokens), and disk space.
- Use it when agents switch between more long conversations than the RAM cache holds (`--conversation-cache-slots`),
  or when RAM is short. With a single long conversation it writes nothing.

**`--conversation-cache-disk-mib N`** (32768). The disk budget. A conversation takes about 16-18 KB a token (2.5 GB at
150K tokens). When it is full, the copy that came back least often leaves first, then the oldest.

**`--conversation-cache-disk-min-tokens N`** (12288). Shorter conversations are not written: reading them again costs
a few seconds at most.

**`--conversation-cache-disk-day-gib N`** (200). A warning in the log past N GiB written in a day; nothing stops.

**`--prelude-cache DIR`** (off) and **`--prelude-cache-mib N`** (5120). The prelude is an agent's system turn (system
prompt and tools, at least `--prompt-cache-root` tokens, 2,048 by default). It is saved once, right after the first
reply that read it; a new chat that starts with it, and that RAM cannot resume further, starts after it.
- Gain: an 8,413-token prelude came back in 0.19 s; the request read its prompt in 0.56 s instead of 7.4 s. RAM
  already does this while one of the agent's conversations is parked; the prelude cache also covers the time after
  they left RAM and after a restart.
- Cost: one write per distinct system prompt (235 MiB for 8K tokens, 0.2 s). An agent whose system prompt changes
  every session (a date, a folder) gets a new prelude each time; the least used leaves first when the budget is full.
- The server passes the id of `system` to the engine (`--system-role-token`) by itself: nothing to set.

**`--conversation-cache-park-checkpoints N`** (0: all). A conversation parked in RAM keeps its last N checkpoints and
its first one (the end of the system prompt), instead of all of them; each is about 118 MB on Qwen3.8-Flash-Next.
- Gain: more conversations fit in `--conversation-cache-mib` (9 checkpoints: 807 MB with N = 3 instead of 1.4 GB).
- Cost: a request that goes back to an older point of a parked conversation reads more again. 3 is a good start.

**`--peer-refill-sm N`** (0: the copy engine). With `--peer-device`, the adaptive tier copies experts into the second
GPU with N blocks of that GPU's own loads. On the copy engine, the small copies that start each layer's work on that
card waited behind the whole round (0.11 ms alone, 23 ms during a round of 72 copies).
- Gain: decode 91.7 -> 97.0 tok/s; with `--adapt-decay 0.92 --peer-adapt-swaps 144` on top, 98.9 tok/s.
- Cost: none measured (a layer's work there took 0.12 ms instead of 0.11 ms during a round). 4 fill a PCIe 4.0 x4 link.
- Use it with a second GPU on a chipset or x4 slot; measured only there.

**Always on with `--peer-device`: the second GPU's rows and input.** Its share of a layer's experts used to be copied in
as floats, quantized there, computed into its memory, then scattered to RAM over its link (x4: ~27 us a layer, after the
experts). Now its down kernel writes each entry's rows straight into RAM as one 128-byte store while the other blocks
compute, and the host quantizes the input to q8_1 the way the device does (AVX2 + F16C; else the device's).
- Gain: per layer 164-173 -> ~136 us, the CPU's wait for the second GPU 82-84 -> ~47 us; decode windows 17.2 -> 16.5
  ms (about 4% at short context, about 3% at long context).
- The same bits: `STRATA_PEER_CHECK=1` also runs the old path and compares every row (880,771 rows, 0 different).
- `STRATA_PEER_TIMING=1` logs, per request, where a layer's call on the second GPU goes (copies, quantization,
  experts, the CPU's wait).

**`"elastic": true`** and **`"elastic_idle_s"`** (120) in the config's `vision` section. The image encoder starts with
the first picture and stops after that many seconds without one. Gain: its RAM (about 1 GB on the CPU) is free between
pictures. Cost: the first picture after a stop takes 2.0 s instead of 0.2 s.

All of them on the reference PC (in the run config's `args`, and the `vision` section):

```json
"--conversation-cache-disk", "/path/to/conversation-cache",
"--conversation-cache-disk-mib", "65536",
"--conversation-cache-disk-min-tokens", "12288",
"--conversation-cache-park-checkpoints", "3",
"--prelude-cache", "/path/to/prelude-cache",
"--prelude-cache-mib", "10240",
"--peer-refill-sm", "4"
```

```json
"vision": {"elastic": true, "elastic_idle_s": 60}
```

### Settings that empty the disk stores

Both stores are kept across restarts and rebuilds of the same engine version with the same model and the settings that
change what the saved bytes mean. Changing one of these makes the next start drop every copy (they fill again as they
are used): the engine version, the model, `--kv`, `--max-context`, `--kv-resident`, the draft layer's window, rope
scaling, control vectors, `--vision`, and the experimental `--native-*` and `--no-*` switches. A layer split
(`--layer-split`) keeps both stores off.

### Upstream settings tuned with these changes (reference PC)

These options come from upstream Strata; each was measured on the reference PC with the fork:

| Setting | What it changed |
| --- | --- |
| `--kv-resident 20480` | The K/V in pinned RAM with 20,480 cells per layer in VRAM: the RTX 3080 Ti holds about 600 more experts, decode about +2.5%. Costs 1.81 GiB of pinned RAM at 150K tokens of context. |
| `--vram-reserve-mib 444` | Since 0.2.8, `--expert-cache auto` leaves the reserve plus 256 MiB for what loads after the cache; 444 keeps the 0.2.7 cache (2,511 experts instead of 2,385, about 1.5% of decode). Fine on Linux, where a nearly full card does not page to system RAM; keep the default on Windows. |
| `STRATA_PARK_ASYNC=1`, `STRATA_CKPT_ASYNC=1`, `STRATA_CKPT_GATHER=1`, `STRATA_CKPT_POOL=1`, `STRATA_CKPT_POOL_READY=4` (env) | Parks and checkpoints off the request's path: two agents taking turns, a turn 1,542 -> 1,179 ms, a park 236 -> 7 ms. Costs 4 x 118 MB of RAM held from start. |
| `--adapt-decay 0.92 --peer-adapt-swaps 144` | With `--peer-refill-sm`, the second GPU follows the text more closely (about +2%). Without it, slower. |
| `--conversation-cache-min-tokens 2048` | With parks this cheap, 2K-6K-token conversations are parked too (a next turn 0.37 s instead of 2.4 s). Not 0: short side requests would take the slots. |
| `--max-context 153600` | 150K tokens of context (the model was trained on 262K). Raise the agent's own context limit too. |

Not measured yet: AMD (HIP), Intel (SYCL), Mac (Metal) and Windows.

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

