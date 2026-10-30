# Strata Experimental

> A fork of **[Strata](https://github.com/Niko1221/Strata)**, created by **[Niko1221](https://github.com/Niko1221)**.
> Strata itself (its engine and apps) is their work; this fork only adds the changes below.
> The [original README](#the-original-strata-readme) follows, unchanged.

Made for agent tools (OpenCode, Claude Code, subagents) that switch between long conversations, and for a second GPU on
a slow PCIe link. The branch `strata-experimental` is Strata plus these changes; `main` follows upstream. Every new
option is off by default: without them, Strata behaves as upstream.

Reference PC for the numbers: Linux, Ryzen 7 5800X3D, 64 GB DDR4-3600, RTX 3080 Ti + RTX 3060 (PCIe 4.0 x4 through
the chipset), NVMe SSD, Qwen3.8-Flash-Next IQ3\_S.

## What it adds

| Change | Option | Gain on the reference PC |
| --- | --- | --- |
| **Conversations on disk.** A long conversation that leaves RAM is written to an SSD and comes back from there. Nothing is written while it is in RAM; a clean stop writes what RAM holds. | `--conversation-cache-disk DIR` | Back in 0.2-0.7 s instead of being read again (13.6K tokens: 0.96 s instead of 6.8 s) |
| **Prelude cache.** Each agent's system prompt and tools are saved once; its new chats start after them, also after a restart. | `--prelude-cache DIR` | 0.56 s instead of 7.4 s for an 8.4K-token prelude |
| **Smaller parked conversations.** A parked conversation keeps only its last checkpoints. | `--conversation-cache-park-checkpoints N` | More conversations fit in RAM |
| **Second GPU: copies by its own compute units.** Experts copied into the second GPU no longer hold back its work. | `--peer-refill-compute-units N` | Decode +6% |
| **Second GPU: less waiting on a slow link.** The data sent to it is 3.5 times smaller, and its results come back while it computes. Same results. | `--peer-slow-link-optimizations` | Decode +4% |
| **Prompt reads skip unused experts** (upstream [#4283](https://github.com/Niko1221/Strata/pull/4283), by [@bitaccesscomau](https://github.com/bitaccesscomau)). Same results. | `STRATA_PREFILL_ROUTED_STREAM=1` (env) | Follow-up reads of 1-2K tokens +16% |
| **Image encoder on the CPU, on demand.** | `"elastic": true` (config, `vision`) | About 1 GB of RAM free between pictures |
| **Always on:** no expert on both GPUs; a log line per request on the second GPU; the Monitor's GPU widgets, store graphs (with the time they saved) and Disk write card. | — | — |

## The settings

| Option | Default | Values | Reference PC |
| --- | --- | --- | --- |
| `--conversation-cache-disk DIR` | off | A folder on an SSD (NVMe best) | a folder on its NVMe |
| `--conversation-cache-disk-mib N` | 32768 | Disk space in MiB. About 17 KB a token: 2.5 GB for a 150K-token conversation | 65536 |
| `--conversation-cache-disk-min-tokens N` | 12288 | The shortest conversation written. Below 4096 it saves nothing | 12288 |
| `--conversation-cache-disk-day-gib N` | 200 | Daily writes before a warning in the log (nothing stops). Your SSD's TBW divided by the days it should last | 200 |
| `--prelude-cache DIR` | off | A folder on an SSD | a folder on its NVMe |
| `--prelude-cache-mib N` | 5120 | Space in MiB. About 28 KB a token: 245 MiB for a 9K-token prelude | 10240 |
| `--prompt-cache-root N` (upstream) | 2048 | The shortest prelude saved, in tokens | 2048 |
| `--conversation-cache-park-checkpoints N` | 0 (all) | Checkpoints kept per parked conversation, about 118 MB each. 2-3 for agents | 3 |
| `--peer-refill-compute-units N` | 0 (copy engine) | How many of the second GPU's compute units (NVIDIA's SMs; an RTX 3060 has 28) do the copies. 2 and 4 measured the same on a PCIe 4.0 x4 link; more only take units from the experts. A wider link may need more (not measured) | 4 |
| `--peer-slow-link-optimizations` | off | A switch, for a second GPU in an x4 or chipset slot | on |
| `STRATA_PREFILL_ROUTED_STREAM` (env) | off | 1 = on | 1 |
| `"elastic"`, `"elastic_idle_s"` (`vision`) | false, 120 | The encoder starts with the first picture and stops after N seconds without one | true, 60 |

The reference PC's run config:

```json
"--conversation-cache-disk", "/path/to/conversation-cache",
"--conversation-cache-disk-mib", "65536",
"--conversation-cache-disk-min-tokens", "12288",
"--conversation-cache-park-checkpoints", "3",
"--prelude-cache", "/path/to/prelude-cache",
"--prelude-cache-mib", "10240",
"--peer-refill-compute-units", "4",
"--peer-slow-link-optimizations"
```

```json
"vision": {"elastic": true, "elastic_idle_s": 60},
"env": {"STRATA_PREFILL_ROUTED_STREAM": "1"}
```

Good to know:
- **Disk stores dropped:** changing the engine version, the model, `--kv`, `--max-context`, `--kv-resident`, `--vision`,
  rope scaling or control vectors empties both stores at the next start. A layer split keeps them off.
- **systemd:** set `KillMode=mixed` in the unit, so a stop reaches the server only and it can write the conversations
  before the engine ends.

## Taken from upstream pull requests

Open pull requests of [Niko1221/Strata](https://github.com/Niko1221/Strata/pulls) carried here before they are merged
upstream. Each keeps its author's commit; credit goes to them.

| Upstream PR | Author | What | Measured here |
| --- | --- | --- | --- |
| [#4283](https://github.com/Niko1221/Strata/pull/4283) | [@bitaccesscomau](https://github.com/bitaccesscomau) | `STRATA_PREFILL_ROUTED_STREAM=1`: a prompt read no longer copies the experts no token uses | Follow-up reads: 1-2K tokens +16%, 2-8K +4-6% |

## Upstream settings tuned for the reference PC

| Setting | Why |
| --- | --- |
| `--kv-resident 20480` | About 600 more experts on the RTX 3080 Ti, decode +2.5% (1.8 GiB of pinned RAM) |
| `--vram-reserve-mib 444` | About 130 more experts on the 3080 Ti, decode +1.5% (Linux only) |
| `STRATA_PARK_ASYNC=1` and `STRATA_CKPT_ASYNC=1`, `_GATHER=1`, `_POOL=1`, `_POOL_READY=4` (env) | An agent switch 1.5 s to 1.2 s |
| `--adapt-decay 0.92 --peer-adapt-swaps 144` | Decode +2%, with `--peer-refill-compute-units` |
| `--conversation-cache-min-tokens 2048` | Conversations of 2K-6K tokens are kept in RAM too |
| `--peer-reserve-mib 280 --peer-prefill-rows 0` | 166 more experts on the headless RTX 3060 (128 fails at start; keep 600 if a desktop uses that card) |
| `--max-context 153600` | 150K tokens of context |

Not measured: AMD (HIP), Intel (SYCL), Mac (Metal) and Windows.

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

