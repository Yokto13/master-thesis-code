
# Architectures for World Models in Reinforcement Learning

This repository contains the source code accompanying my master's thesis.
The main contribution of this thesis is a new reconstruction-free architecture built on top of [Dreamer 3](https://www.nature.com/articles/s41586-025-08744-2) and [LeJEPA](https://arxiv.org/abs/2511.08544).
Furthermore, I conducted multiple experiments using the Dreamer 3 architecture; source code of these experiments is also part of the codebase.

See the thesis (link to be added soon) for details.

## Get Started

The models are relatively large; therefore, training them on a GPU is highly encouraged.

This repository uses `uv` for dependency management. Install it with:

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

```

Now, run:

```bash
WANDB_MODE="disabled" uv run train_dreamer.py --config configs/carracing.gin

```

This will start training on the CarRacing environment. It takes approximately 5,000 to 10,000 steps before the agent learns to drive successfully.

**Note on Wandb**

By default, all experiments log to Wandb.
Unless you plan to run large training sessions, it is best to disable logging by setting `WANDB_MODE="disabled"`.

## Structure of the Repository

The repository contains two branches:

* `main`: Contains the reimplementation of the Dreamer 3 algorithm and supports running some of the experiments from the thesis.
* `reconstruction-free`: Provides the implementation of the reconstruction-free agent.

I conducted these experiments over several months, during which the codebase was constantly changing.
Consequently, it is difficult to guarantee that minor changes will not alter the results of older experiments.
Instead of providing a single implementation that handles everything, I have provided snapshots of the `src` directory from the exact moments the experiments were conducted.
These snapshots are in `experiments_snapshots` directory.
This should facilitate reproducibility.