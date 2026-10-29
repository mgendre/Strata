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
plus the changes below; `main` follows upstream. Every new option is off by default. A few fixes are always on (marked
"always" below); apart from those, Strata without the new options behaves as upstream does.

**The goal.** Agent tools (OpenCode, Claude Code, subagents) switch between several long conversations. Strata keeps
a few of them in RAM. When RAM is full, the oldest one is dropped, and its next turn reads the whole prompt again: tens
of seconds for a long conversation. This fork keeps those conversations on disk, makes each one smaller in RAM, and
shows what both do in the Monitor, while keeping writes to the SSD low.

### What is new

| Change | Options (default) |
| --- | --- |
| **Conversation cache on disk.** Long conversations are saved on an SSD and come back from there instead of being read again. | `--conversation-cache-disk DIR` (off), `--conversation-cache-disk-mib` (32768), `--conversation-cache-disk-first-tokens` (32768), `--conversation-cache-disk-step-tokens` (16384), `--conversation-cache-disk-topup-tokens` (0: off), `--conversation-cache-disk-day-gib` (200: a warning only) |
| **Prelude cache.** The prelude of each agent (its system prompt and tools, up to the first turn) is saved once in a folder of its own. A new chat of that agent starts after it, also after a restart, instead of reading it again. | `--prelude-cache DIR` (off), `--prelude-cache-mib` (5120) |
| **Spill.** A conversation below the first save step is written to disk when RAM lets it go, so only the ones RAM loses are saved. | `--conversation-cache-disk-spill-tokens N` (0: off) |
| **Smaller parked conversations in RAM.** A parked conversation keeps its last N checkpoints and its first one (the end of the system prompt) instead of all of them. Each checkpoint is about 118 MB on Qwen3.8-Flash-Next. | `--conversation-cache-park-checkpoints N` (0: all) |
| **Clearer log.** One line for each save to disk, each restore from disk, and each conversation that leaves RAM, with whether it is on disk. | always, with the disk cache |
| **Second GPU line.** With `--peer-device`, one line per request says where the experts ran: the main GPU's cache, the second GPU, over PCIe, or on the CPU. | always, with `--peer-device` |
| **Second GPU refill by SM loads.** With `--peer-device`, the adaptive tier copies experts into the second GPU with the GPU's own loads instead of its copy engine. On the copy engine, each layer's work on that card waited until the whole round of copies had landed. | `--peer-refill-sm N` (0: the copy engine) |
| **No expert on both GPUs.** The main GPU's adaptive tier no longer takes an expert that is still being copied into the second GPU, which left both cards holding it. | always, with `--peer-device` |
| **Image encoder on the CPU, started on demand.** The vision section's `"elastic": true` now works for an encoder on the CPU, also with `--peer-device` (it no longer needs `"vram_elastic"`). | `"elastic": true`, `"elastic_idle_s"` (120) in the config's `vision` section |
| **Prompt heads (a diagnostic).** Each distinct system turn a request opens with (an agent's system prompt and tools) is written to a folder once, with an index of how often and when it came, so a diff shows what an agent changes between sessions and why its next chat cannot start after the stored prelude. | `"prompt_heads": DIR` in the config (off) |
| **Monitor.** The Conversation cache card shows the RAM and disk bars, where restored conversations came from, the prompt tokens given back per hour (RAM and disk), the disk writes per hour and today, and one row per conversation on disk. A Disk write card sits next to Disk read. | always |

### The options, and what turning them on does

Measured on the reference PC (Linux, Ryzen 7 5800X3D, 64 GB DDR4-3600, RTX 3080 Ti + RTX 3060 behind the X570
chipset, a PCIe 4.0 NVMe SSD, Qwen3.8-Flash-Next IQ3\_S). Other PCs will see other numbers.

**`--conversation-cache-disk DIR`** (off). Saves long conversations on an SSD, so they come back from it when RAM has
let them go or after a restart.
- Gain: a returning conversation is restored in 0.2-0.26 s; its request took 0.7-1.2 s instead of about 7 s to read
  a 14K-20K-token prompt again.
- Cost: SSD space up to `--conversation-cache-disk-mib`, and writes: about 0.3 s after a reply that reaches a save
  step (only new 4K-token blocks are written: a conversation that grew by 6K tokens wrote 232 MiB). A request that
  arrives during a save waits for it.
- Use it when agents switch between long conversations, or when the engine restarts often.

**`--conversation-cache-disk-mib N`** (32768). The disk budget. More keeps more conversations (on disk a conversation takes about 16-18 KB a
token: about 2.5 GB at 150K tokens); it costs only disk space.

**`--conversation-cache-disk-first-tokens N`** (32768) and **`--conversation-cache-disk-step-tokens N`** (16384).
A conversation is saved once it reaches N tokens, then again every step further. Lower values save more and fresher
copies and write more. A 13-day replay of the reference PC's log: lowering the first step to 12K gave back 0.03
minutes of prompt reading per GB written, less than the spill below (0.05).

**`--conversation-cache-disk-topup-tokens N`** (0: off). When RAM pushes out, or cannot hold, a conversation N or
more tokens past its copy, that gap is saved too. More writes, fresher copies.

**`--conversation-cache-disk-day-gib N`** (200). Only a warning in the Monitor past N GiB written in a day; nothing
stops.

**`--conversation-cache-disk-spill-tokens N`** (0: off). A conversation of N tokens or more that RAM lets go, or
cannot take, with no copy on disk is saved then, so the mid-size ones below the first step are not lost.
- Gain: a 13.6K-token conversation came back from disk in 0.21 s and read its prompt in 0.96 s instead of about 6.8 s.
- Cost: the request that pushed it out waits for the write (0.26 s, 310 MiB for 13.6K tokens).
- A copy that leaves RAM while a newer copy of the same conversation is still kept (the live one, or another parked
  copy holding all of it but its last few thousand tokens) is not spilled: agent clients that re-render the last
  turn make the engine borrow from the old copy and park a new one, and only the newest is worth saving.

**`--conversation-cache-park-checkpoints N`** (0: all). A conversation parked in RAM keeps its last N checkpoints and
its first one instead of all of them (118 MB each on Qwen3.8-Flash-Next).
- Gain: more conversations fit in the RAM budget (9 checkpoints took 807 MB with N = 3 instead of about 1.4 GB).
- Cost: a request that branches off an older point of the conversation reads more again.

**`--prelude-cache DIR`** (off) and **`--prelude-cache-mib N`** (5120). Saves each agent's prelude (its system turn:
system prompt and tools, at least `--prompt-cache-root` tokens) once, in a store of its own.
- Gain: a new chat of that agent starts after it, also after a restart: an 8,413-token prelude came back in 0.19 s
  and the request read its prompt in 0.56 s instead of 7.4 s.
- Cost: one write per distinct system prompt (235 MiB for 8K tokens, 0.2 s after the first reply) and disk space up
  to N MiB. An agent that changes its system prompt every session (a date, a folder) gets a new prelude each time;
  `"prompt_heads"` below shows whether it does.
- The server passes the id of `system` as `--system-role-token` by itself: a chat without a system prompt has no
  prelude. Nothing to set.

**`--peer-refill-sm N`** (0: the copy engine). With `--peer-device`, the adaptive tier copies experts into the second
GPU with N blocks of that GPU's own loads instead of its copy engine.
- Gain: decode 91.7 -> 97.0 tok/s on the reference PC; the CPU waited 1.5 ms a decode window for the second GPU
  instead of 3.3 ms. On the copy engine, each layer's work on that card waited behind the whole round of copies.
- Cost: during a round, N blocks of the second GPU copy (4 fill a PCIe 4.0 x4 link at 6 GB/s); a layer's work there
  took 0.12 ms instead of 0.11 ms meanwhile. Experts outside registered memory keep the copy engine.
- Use it with a second GPU on a slow link (a chipset or x4 slot). Measured only there.

**`"prompt_heads": DIR`** in the config (off). A diagnostic: each distinct system turn a request opens with is
written to DIR once as text, with `index.json` counting how often and when it came. A diff of two files shows what an
agent changes between sessions. Cost: system prompts stored as plain text on the local disk (at most 200 files).

**`"elastic": true`** in the config's `vision` section, with `"elastic_idle_s"` (120). The image encoder starts with
the first picture and stops after that many seconds without one.
- Gain: the CPU encoder's RAM (about 1 GB) is free while no picture comes.
- Cost: the first picture after a stop waits for the encoder (2.0 s in all, the encoder started in 1.1 s); the next
  ones as before (0.2 s).

**Always on.** With `--peer-device`, the main GPU's tier no longer takes an expert still being copied into the second
GPU (both cards held it before). The log lines, the second GPU's line and the Monitor cards need nothing either.

### Settings that empty the disk caches

The copies on disk are kept across restarts of the same engine version (and disk format) with the same model files and
the same settings that change what the saved bytes mean. Changing one of these makes the next start drop every copy
of both caches (they fill again as the conversations are read): the model, `--kv`, `--max-context`, `--kv-resident`,
the draft layer's window, rope scaling, control vectors, `--vision`, and the experimental `--native-*` and `--no-*`
switches. A rebuild of the same version keeps them.

### Upstream settings tuned with these changes (reference PC)

These options come from upstream Strata; the reference PC runs them with the fork, and each was measured there:

| Setting | What it changed |
| --- | --- |
| `--kv-resident 20480` | The K/V lives in pinned RAM with 20,480 cells per layer in VRAM; the RTX 3080 Ti's expert cache went from 1,932 to 2,550 experts. Decode about +2.5%; 97-99.6% of the K/V reads still hit VRAM. Costs 1.33 GiB of pinned RAM at 112K tokens of context, 1.81 GiB at 150K. Empties the disk caches once. |
| `STRATA_PARK_ASYNC=1`, `STRATA_CKPT_ASYNC=1`, `STRATA_CKPT_GATHER=1`, `STRATA_CKPT_POOL=1`, `STRATA_CKPT_POOL_READY=4` (env) | Parking a conversation and saving checkpoints move off the request's path. Two agents taking turns on ~10K-token conversations: a turn 1,542 -> 1,179 ms, a park 236 -> 7 ms. Costs 4 x 118 MB of RAM held from start. |
| `--adapt-decay 0.92 --peer-adapt-swaps 144` | With `--peer-refill-sm`, the second GPU follows the text more closely: decode 97.0 -> 98.9 tok/s (about 2%, close to the noise between runs). Without the refill change the same settings made decode slower (79 tok/s). |
| `--conversation-cache-min-tokens 2048` | With parks this cheap, conversations of 2K-6K tokens are parked too: the next turn of a 2.5K-token one took 0.37 s instead of 2.4 s. Not 0: the RAM cache has 4 slots, kept in least-recently-used order, and short side requests (session titles) would push agents' conversations out. |
| `--max-context 153600` | 150K tokens of context (the model was trained on 262K). With `--kv-resident` it costs pinned RAM (+0.48 GiB) and 37 experts of the main GPU's cache. Empties the disk caches once. Agents compact by their own configured limit: raise it there too. |
| `"vision": {"elastic": true, "elastic_idle_s": 60}` | The CPU image encoder's 1 GB of RAM free between pictures. |

### How the disk cache works

- A conversation is saved right after a reply, once it reaches 32K tokens, and again every 16K tokens more. The answer
  is already sent: nobody waits for the save.
- Only what is new is written. The conversation is stored in blocks of 4K tokens named by their content, so a block
  already on disk is never written again. Each save adds the new blocks and one snapshot of the model's state.
- When RAM drops a conversation, nothing is written: its copy is on disk already.
- With `--conversation-cache-disk-spill-tokens N`, a conversation of N tokens or more that RAM drops with no copy on
  disk is written then. The request that pushed it out waits for the write (0.26 s for 13.6K tokens).
- When the conversation comes back, its copy goes from the disk to the GPU, and only the tokens since the last save are
  read again.
- The copies are kept across restarts and rebuilds of the same engine version, with the same model and settings.
  Another version, model or setting that changes the saved data starts with an empty cache.
- When the cache is full, the conversation that came back least often leaves first, then the oldest.
- To spare the SSD, files are written in large pieces and never rewritten, and nothing is flushed only for safety: a
  file cut short by a crash fails its check and is deleted.

To turn it on, add to the run config's `args` (the folder should be on an SSD):

```json
"--conversation-cache-disk", "/path/to/conversation-cache",
"--conversation-cache-disk-mib", "32768",
"--conversation-cache-park-checkpoints", "3"
```

### Measured

On one Linux PC with a PCIe 4 NVMe SSD, Qwen3.8-Flash-Next IQ3\_S, test conversations of 14K to 20K tokens:

- A save after a reply took about 0.3 s.
- A restore from disk took 0.2 to 0.26 s. The request then took 0.7 to 1.2 s, instead of about 7 s to read the prompt
  again.
- A conversation that grew by 6K tokens wrote 232 MiB and reused 178 MiB that was already on disk.
- After a restore, every conversation still knew a code given at its very start. Five answers matched a fresh read
  token for token.
- A parked conversation with 9 checkpoints took 807 MB with `--conversation-cache-park-checkpoints 3`, instead of about
  1.4 GB.

### The prelude cache

Agents send the same system prompt and tools at the start of every new chat: 7K to 10K tokens for the agents measured
here. Strata already keeps that part in RAM while one of the agent's conversations is parked there. The prelude cache
keeps it on disk too, so a new chat still starts after it once those conversations have left RAM or the engine has
restarted.

- The prelude ends at the first turn boundary after the system prompt, when it is at least `--prompt-cache-root`
  tokens long (2,048 by default). It is saved right after the reply of the first chat that reads it, once. A chat that
  does not start with a system prompt has no prelude (the server passes the id of `system` to the engine).
- A new chat that starts with a saved prelude, and that RAM cannot resume further, comes back from it. The Monitor counts
  these restores and their tokens next to the RAM and disk ones.
- The folder keeps `--prelude-cache-mib` in all (5 GiB by default, about 20 preludes of 8K tokens). The prelude used
  least leaves first. Like the disk cache, it is kept across restarts of the same engine version, model and settings.

```json
"--prelude-cache", "/path/to/prelude-cache",
"--prelude-cache-mib", "5120"
```

Measured on the same PC with an 8,413-token system prompt: the save took 0.2 s (235 MiB). After a restart, the
restore took 0.19 s and the request read its prompt in 0.56 s instead of 7.4 s. The answer (a code planted in the
system prompt) was the same text.

A conversation of 13.6K tokens spilled when RAM let it go wrote 310 MiB in 0.26 s; it came back from disk in 0.21 s and
read its prompt in 0.96 s instead of about 6.8 s.

### The second GPU's refill

With `--peer-device`, the adaptive tier moves the experts the text uses into the second GPU while Strata writes an
answer: up to `--peer-adapt-swaps` copies of about 2 MiB per round. On the copy engine, the small copies that start each
layer's work on that card wait behind the whole round. Measured on an RTX 3060 behind the X570 chipset (PCIe 4.0 x4),
one layer's work there took 0.11 ms alone and 23 ms during a round of 72 copies. With `--peer-refill-sm 4`, four
blocks of the GPU copy the experts at the same 6 GB/s, and the layer's work took 0.12 ms.

```json
"--peer-refill-sm", "4"
```

Decode on the same PC (RTX 3080 Ti + RTX 3060, four answers of 400 tokens, three rounds): 91.7 tok/s before, 97.0
tok/s with `--peer-refill-sm 4`. The CPU waited 1.5 ms per decode window for the second GPU instead of 3.3 ms. Once
these copies are cheap, the second GPU can follow the text more closely: `--adapt-decay 0.92 --peer-adapt-swaps 144`
gave 98.9 tok/s (three runs; runs of the same settings differ by about 2%). Before the change, the same two settings
made decode slower (79 tok/s), because each copy into the second GPU cost about 0.3 ms of every window. The copies move
the same bytes, so the answers keep their quality.

Not measured yet: days of real use, AMD (HIP), Intel (SYCL), Mac (Metal) and Windows. With a layer split across two
GPUs (`--layer-split`) the disk cache and the prelude cache stay off.

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

<sub>Files: `Strata-0.2.7-windows-x64-setup.exe` · `Strata-0.2.7-linux-x86_64.AppImage` · `Strata-0.2.7-macos-arm64.dmg`</sub>

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

