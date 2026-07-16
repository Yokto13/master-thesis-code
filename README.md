
# Architectures for World Models in Reinforcement Learning

This repository contains the source code accompanying my master's thesis.
The main contribution of this thesis is a new reconstruction-free architecture built on top of [Dreamer 3](https://www.nature.com/articles/s41586-025-08744-2) and [LeJEPA](https://arxiv.org/abs/2511.08544).
Furthermore, I conducted multiple experiments using the Dreamer 3 architecture; source code of these experiments is also part of the codebase.

See the thesis (link to be added soon) for details.

## Get Started

The models are relatively large; therefore, training them on a GPU is highly encouraged.

This repository uses `uv` for dependency management. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

Now, run:

```bash
WANDB_MODE="disabled" uv run train_dreamer.py --config configs/carracing.gin

```

This will start training on the CarRacing environment. It takes approximately 5,000 to 10,000 steps before the agent learns to drive successfully.

For a simple reconstruction-free run, I recommend training Pong from configs/atari100k_dreaming_jepa on the reconstruction-free branch. The full training takes several hours on a GPU, but the agent improves relatively quickly; hence, some improvements should be visible after a few dozen minutes.

**Note on Wandb**

By default, all experiments log to Wandb.
Unless you plan to run large training sessions, it is best to disable logging by setting `WANDB_MODE="disabled"`.

## Results

### Atari100k

| Game | Dreamer 3 | MuDreamer | Our Reimpl. of Dreamer 3 | Dreaming JEPA |
| :--- | ---: | ---: | ---: | ---: |
| Alien | 1278 | 951 | 923 | **1453** |
| Amidar | 120 | 153 | 199 | 216 |
| Assault | 741 | 891 | 741 | 629 |
| Asterix | 1020 | 1411 | 962 | 955 |
| Bank Heist | 422 | 156 | 364 | **1069** |
| Battle Zone | 20800 | 12080 | 15120 | 13240 |
| Boxing | 87 | 96 | **98** | 76 |
| Breakout | 11 | 34 | 15 | 91 |
| Chopper Command | 2440 | 808 | 1714 | 1552 |
| Crazy Climber | 80060 | **96128** | 81462 | 73776 |
| Demon Attack | 454 | 553 | 189 | 302 |
| Freeway | 0 | 5 | 4 | 0 |
| Frostbite | **3914** | 1652 | 2222 | 1754 |
| Gopher | **2252** | 1500 | 2073 | 1444 |
| Hero | **13324** | 8272 | 9963 | 12094 |
| Jamesbond | 490 | 409 | 205 | 208 |
| Kangaroo | 2840 | 4380 | 6504 | **7748** |
| Krull | 8604 | 9644 | **10313** | 9088 |
| Kung Fu Master | 25560 | **26832** | 22792 | 23422 |
| Ms Pacman | 1400 | 2311 | 1837 | **2976** |
| Pong | -5 | 18 | 14 | **21** |
| Private Eye | **3238** | 1042 | 1815 | 139 |
| Qbert | 3700 | 4061 | 4321 | 2489 |
| Road Runner | 18440 | 8460 | 12690 | 11688 |
| Seaquest | 964 | 428 | 842 | 966 |
| Up N Down | 49456 | 26494 | 30498 | **68669** |
| HNS mean | 128 | 126 | 128 | 143 |
| HNS med | 49 | 34 | 47 | 48 |
| HNS IQM | 52 | n/a | 50 | 48 |
| Better than human | 7/26 | 11/26 | 7/26 | 10/26 |

*The results of Dreaming JEPA in comparison with the public Dreamer 3 implementation, MuDreamer (different reconstruction-free approach), and our reimplementation.*

Dreaming JEPA corresponds to the **reconstruction-free** branch.
Our Dreamer 3 reimplementation to the **main** branch.

### Crafter

| | Score | Return |
| :--- | :---: | :---: |
| Rainbow | 4.3 ± 0.2 | 6.0 ± 1.3 |
| PPO | 4.6 ± 0.3 | 4.2 ± 1.2 |
| Dreamer 3 | 14.5 ± 1.6 | 11.7 ± 1.9 |
| DreamerPro | 4.7 ± 0.5 | -- |
| MuDreamer | 7.3 ± 2.6 | 5.6 ± 1.6 |
| Dreamer-CDP | 16.2 ± 2.1 | 9.8 ± 0.4 |
| Dreaming JEPA | 5.6 ± 0.3 | 5.7 ± 0.3 |

*Results of our reconstruction-free architecture Dreaming JEPA compared to other popular approaches used on Crafter.*

Dreaming JEPA corresponds to the **reconstruction-free** branch.


## Structure of the Repository

The repository contains two branches:

* `main`: Contains the reimplementation of the Dreamer 3 algorithm and supports running the weight decay and replay buffer experiments.
* `reconstruction-free`: Provides the implementation of the reconstruction-free agent.

I conducted many of the experiments over several months, during which the codebase was constantly changing.
Consequently, it is difficult to guarantee that minor changes will not alter the results of older experiments.
Instead of providing a single implementation that handles everything, I provide snapshots of the `src` directory from the exact moments the older experiments were conducted.
These snapshots are in the `experiment_snapshots` directory.
This should facilitate reproducibility.