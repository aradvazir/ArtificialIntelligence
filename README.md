# Artificial Intelligence — Course Projects

Three projects from an **Artificial Intelligence** course, spanning the classic AI syllabus from local search to reinforcement learning to deep learning:

1. **Local search / metaheuristics** — Hill Climbing, Simulated Annealing and a Genetic Algorithm, all attacking the same NP-hard **Cutting Stock Problem**, so the three can be compared head-to-head on identical inputs.
2. **Reinforcement learning** — tabular **Q-learning** teaching a Pac-Man agent to clear a maze, with a hand-designed state abstraction that shrinks the state space from 2⁴³ to 3⁸.
3. **Deep learning** — a **U-Net** (and a stacked two-stage U-Net) built from scratch in Keras for medical image segmentation of gastrointestinal polyps on **Kvasir-SEG**.

The metaheuristics comparison is the highlight: three algorithms, one problem, one cost function, four benchmark instances with hard targets to beat — and a clear winner.

---

## Table of Contents

- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Data you need to supply](#data-you-need-to-supply)
- [Project 1 — Cutting Stock: Hill Climbing vs. Simulated Annealing vs. Genetic Algorithm](#project-1--cutting-stock-hill-climbing-vs-simulated-annealing-vs-genetic-algorithm)
- [Project 2 — Pac-Man with Tabular Q-Learning](#project-2--pac-man-with-tabular-q-learning-pacmanipynb)
- [Project 3 — Polyp Segmentation with U-Net on Kvasir-SEG](#project-3--polyp-segmentation-with-u-net-on-kvasir-seg-unet-kavirseg)
- [Results at a glance](#results-at-a-glance)
- [Known issues, quirks and caveats](#known-issues-quirks-and-caveats)
- [Ideas for improvement](#ideas-for-improvement)
- [Author](#author)

---

## Repository structure

```
ArtificialIntelligence/
│
├── HillClimbing-SA-Genetic-stock/
│   ├── HillClimbing-stock.ipynb     # Steepest-ascent hill climbing over cut orderings
│   ├── SA-stock.ipynb               # Simulated annealing with geometric cooling
│   ├── Genetic-stock.ipynb          # GA: tournament selection, order crossover, swap mutation
│   └── AI_HW2.pdf                   # Assignment brief (defines the target roll counts)
│
├── Pacman.ipynb                     # Tabular Q-learning + pygame visualization
│
└── UNet-KavirSeg/
    ├── UNetSegmentation.ipynb       # U-Net and stacked U-Net in TensorFlow/Keras
    └── Report.pdf                   # Written report
```

---

## Setup

Python 3.8+ with Jupyter.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install jupyter numpy matplotlib                   # cutting stock
pip install pygame                                     # Pac-Man visualization
pip install tensorflow pillow pandas scikit-learn      # U-Net
```

Note that `Pacman.ipynb` opens a **real pygame window** during `Test(...)`, so it needs a desktop session — it won't render in a headless environment or in Colab.

---

## Data you need to supply

| Notebook | Expects | What it is |
|---|---|---|
| The three `*-stock.ipynb` notebooks | `input1.stock` … `input4.stock` | Course-supplied problem instances. Format: the stock (roll) length is the 3rd token of line 1; lines 2–3 are skipped; line 4 is a comma-separated list of requested piece lengths, with a trailing separator. |
| `UNetSegmentation.ipynb` | `./data/Kvasir-SEG/` containing `images/`, `masks/` and `kavsir_bboxes.json` | **Kvasir-SEG** — 1,000 endoscopy images of gastrointestinal polyps with pixel-level ground-truth masks and bounding boxes |
| `Pacman.ipynb` | *(nothing)* | Both mazes are hard-coded in the notebook |

---

## Project 1 — Cutting Stock: Hill Climbing vs. Simulated Annealing vs. Genetic Algorithm

### The problem

Given an unlimited supply of rolls of fixed length *L* and a list of requested piece lengths, cut all the requests using **as few rolls as possible**. It's NP-hard. (Classic toy example: 10 m rolls, requests of 3, 5 and 7 m → the optimum is 2 rolls: `3+5` on one, `7` on the other.)

### The shared encoding

All three notebooks make the same modelling choice, and it's the key design decision of the project:

> **A solution is a *permutation* of the requests. The cost of a permutation is the number of rolls consumed by cutting the requests in that order, greedily (next-fit): keep adding to the current roll until the next piece doesn't fit, then start a new roll.**

This is elegant — it converts a messy bin-packing problem into a clean **permutation search space**, exactly the shape that all three metaheuristics know how to explore. The `calculate_cost` function is identical across the three notebooks, so the comparison is genuinely apples-to-apples.

### The three algorithms

**Hill Climbing** (`HillClimbing-stock.ipynb`)
A **steepest-ascent** variant: `neighbour()` generates `neib=500` random pairwise swaps, evaluates all of them, and returns the best. The outer loop runs `iteration=500` rounds and moves whenever a strictly better neighbour is found. Purely greedy — it never accepts a worse solution, so it gets stuck in the first local optimum with no escape route.

**Simulated Annealing** (`SA-stock.ipynb`)
Same swap neighbourhood, but a **single** random swap per step, accepted by the Metropolis criterion:

$$P(\text{accept}) = \exp\!\left(\frac{\text{cost} - \text{cost}_{\text{new}}}{T}\right)$$

Improvements are always accepted; worsening moves are accepted with probability decaying in the size of the damage and rising with temperature. Geometric cooling `T ← 0.99·T` from `T=1`, with `TL=1000` moves per temperature level, until `T < 0.1`. That works out to roughly **460 temperature levels × 1000 moves ≈ 460,000 evaluations**, with a `best_order` tracked separately so a lucky early solution is never lost.

**Genetic Algorithm** (`Genetic-stock.ipynb`)
A full GA on permutation chromosomes:

| Component | Implementation |
|---|---|
| Representation | Permutation of request **indices** (not values — this differs from HC/SA) |
| Selection | **Tournament**, size 4 |
| Crossover | **Order Crossover (OX)**, 95% rate — copies a random slice from one parent, fills the rest from the other parent in cyclic order, skipping duplicates so the child stays a valid permutation |
| Mutation | Swap of two random positions, 50% rate |
| Replacement | Elitist: the worst `replacement_rate × n_pop` (50%) of the population are replaced by the best children |
| Stopping | Terminates when the best cost hasn't improved for `iteration` generations, or reaches 0 |
| Restarts | The whole thing runs `s=5` independent times; the best of the 5 is reported |

Run with `n_pop=50`, `iteration=1000`, `replacement_rate=0.5`, `mutation=0.5`.

### Results

The assignment (`AI_HW2.pdf`) sets a **target roll count for each instance**. Here is every algorithm against every target:

| Instance | Target | Hill Climbing | **Simulated Annealing** | Genetic Algorithm |
|---|---:|---:|---:|---:|
| `input1` | < 56 | 52 ✅ | **51 ✅** | 54 ✅ |
| `input2` | < 80 | 84 ❌ | **78 ✅** | 83 ❌ |
| `input3` | < 115 | 99 ✅ | **96 ✅** | 105 ✅ |
| `input4` | < 235 | 225 ✅ | **218 ✅** | 243 ❌ |

**Simulated Annealing wins outright** — it is the only one of the three to beat all four targets, and it produces the best solution on every single instance.

The ranking is a textbook illustration of *why* the algorithms behave the way they do:

- **Hill Climbing** does well on the easy instances but **fails `input2`** (84 vs. the 80 target). No mechanism to accept a worse move means no escape from a local optimum, and on a rugged permutation landscape that's fatal.
- **SA is the same neighbourhood as HC** — one swap — plus the single ingredient HC lacks: a principled way to move *downhill* early on. That one addition is worth 6 rolls on `input2` and 7 on `input4`.
- **The GA is the weakest**, and it's the interesting result. Two plausible reasons: its swap mutation is a *very* weak explorer for permutations (a single transposition barely perturbs a 500-element chromosome), and 50% elitist replacement with a small population (50) drives premature convergence — the 5 restarts on `input3` return 105, 106, 107, 108, 108, a tight cluster suggesting the population collapses long before it finds the good regions. A GA's advantage is *recombination*, but for cutting stock the building blocks that OX preserves (contiguous slices of the ordering) aren't obviously the right building blocks.

---

## Project 2 — Pac-Man with Tabular Q-Learning (`Pacman.ipynb`)

Teach an agent to clear a maze of dots using tabular Q-learning, then visualize it with pygame.

### The state-space problem, and the fix

The maze has 43 dots, 2 empty cells and 17 walls. Naively, a state is "which dots are still on the board" — that's **2⁴³ ≈ 8.8 trillion states**, and a Q-table of that size is not happening.

The notebook's answer is a **local perception window**: the state is the **3×3 grid centered on the agent**. Each of the 8 surrounding cells is one of `E` (empty), `W` (wall), `D` (dot), so:

$$3^8 = 6{,}561 \text{ states} \times 4 \text{ actions} = \text{a } 6561 \times 4 \text{ Q-table}$$

`hash_function` maps a 3×3 window to an index by reading the 8 neighbours as a **base-3 number** (`E`→0, `W`→1, `D`→2) — a neat, collision-free hash.

The notebook explicitly argues *against* a bigger window (5×5, 7×7…) on the grounds that it would overfit to this specific maze and stop generalizing — and it then **tests that claim** by evaluating every trained Q-table on a *second, unseen maze*. That's a real generalization experiment, not just a training score.

### MDP definition

| | |
|---|---|
| **Actions** | 4 — right, left, up, down |
| **Reward** | Dot: **+3** · Empty: **−1** (discourages aimless wandering and loops) · Wall: **−2** (and the move is not executed) |
| **Goal** | All dots eaten |
| **Update** | $Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha\big(R + \gamma \max_{a'} Q(s',a')\big)$ |

Exploration uses an **annealed ε-greedy schedule**: the greedy probability starts at 0.05 and is multiplied by 1.0001 each episode until it reaches 0.9 — roughly **29,000 episodes** of gradually shifting from near-random exploration to near-pure exploitation. At test time the agent is fully greedy, with one safeguard: if it goes 20 steps without eating a dot (i.e. it's stuck in a loop, which a purely local policy is prone to), it takes a random non-wall action to break out.

### Results — a 3×3 grid search over α and γ

Each cell is `total reward (steps taken)`. Training is always on maze 1; maze 2 is never seen during training.

| α \ γ | γ = 0.25 | γ = 0.5 | γ = 1.0 |
|---|---|---|---|
| **α = 0.25** | 52 (120) · *alt:* −275 (365) | −206 (378) · *alt:* **−48 (141)** | −476 (463) · *alt:* −468 (433) |
| **α = 0.5** | 21 (151) · *alt:* −119 (208) | **120 (52)** · *alt:* −355 (431) | −627 (532) · *alt:* −715 (654) |
| **α = 0.75** | 66 (106) · *alt:* −110 (214) | 30 (142) · *alt:* −108 (201) | −579 (584) · *alt:* −397 (392) |

*(first number = training maze, 43 dots; `alt:` = the unseen alternative maze, 31 dots)*

**The best run — α = 0.5, γ = 0.5 — is essentially optimal.** It clears all 43 dots in **52 steps for a reward of 120**. The theoretical ceiling is 129 (43 dots × +3 with zero wasted moves); this agent wastes only 9 moves. On a maze that requires backtracking through corridors, that's about as good as it gets.

**Two findings stand out:**

1. **γ = 1.0 is catastrophic across the board** — every single configuration with no discounting produces a large negative reward. With γ=1 the Bellman backup has no contraction, so Q-values in a cyclic maze (where you can walk in circles forever) diverge rather than converge. This is the theory of discounting showing up empirically, and the notebook's conclusion ("increasing γ always tends to a worse learning for the agent") is exactly right.
2. **The policy does not transfer.** Every configuration scores worse on the unseen maze, and most go strongly negative. This is the honest cost of the 3×3 abstraction: the agent is effectively a **reflex agent with 1-cell vision and no memory**. It cannot represent "the remaining dots are over *there*", so once the local neighbourhood is all empty it has no signal at all — hence the loops, and hence the 20-step random-escape hatch. The state abstraction that made the problem tractable is also the thing that caps its performance. That trade-off is the real lesson of the project.

---

## Project 3 — Polyp Segmentation with U-Net on Kvasir-SEG (`UNet-KavirSeg/`)

Binary semantic segmentation: given an endoscopy image, produce a pixel mask of the gastrointestinal polyp. **Kvasir-SEG**: 1,000 images + ground-truth masks + bounding boxes.

### Preprocessing

Images resized to **128×128** and scaled to [0, 1]; masks converted to greyscale (`L`) and resized identically. 70/30 train/test split, with a further 15% of train held out for validation → **595 train / 105 val / 300 test**.

### Model 1 — U-Net (7.9M parameters)

A textbook U-Net, hand-built from three reusable blocks (`down_block`, `bottleneck`, `up_block`):

```
Input 128×128×3
  ↓ down 8 → 16 → 32 → 64 → 128 → 256     (each: 2× Conv3×3 + MaxPool 2×2)
  ↓ bottleneck 512
  ↑ up 256 → 128 → 64 → 32 → 16 → 8       (each: UpSample + concat skip + 2× Conv3×3)
Conv 1×1, sigmoid → 128×128×1 mask
```

The **skip connections** are what make it a U-Net: each decoder stage concatenates the matching encoder feature map, so fine spatial detail lost to pooling is restored at the output. Trained with Adam + binary crossentropy, 30 epochs, batch 32.

| | |
|---|---:|
| Test loss (BCE) | 0.261 |
| **Test pixel accuracy** | **0.890** |

### Model 2 — Stacked U-Net (2.4M parameters)

The idea, and it *is* a good one: run a first (shallower) U-Net, then **concatenate its predicted mask back onto the original RGB image** (3 channels + 1 mask channel = 4) and feed that into a **second U-Net**. The first network's output acts as an attention map, telling the second where to look.

| | |
|---|---:|
| Test loss (BCE) | 0.364 |
| **Test pixel accuracy** | **0.845** |

**It performs worse than the single U-Net**, and the training log shows why: over the final epochs the loss goes *up* (0.3778 → 0.3787 → 0.3803 → 0.3819) while accuracy sits frozen at 0.8338. That's a network that has stopped learning. The cause is visible in the code — every decoder block of the second U-Net is built with `activation="sigmoid"`:

```python
up2_1 = up_block(bn, conv2_5, 128, activation="sigmoid")
up2_2 = up_block(up2_1, conv2_4, 64, activation="sigmoid")
up2_3 = up_block(up2_2, conv2_3, 32, activation="sigmoid")
```

Sigmoid saturates, and stacking saturating activations through a deep decoder kills the gradient. Sigmoid belongs on the *final* 1×1 conv (to squash logits into a probability) and nowhere else — the hidden layers want ReLU. **Swapping these three to `relu` is a one-word fix** that would likely make the stacked architecture competitive, and the report's own conclusion ("the idea used for the stacked U-Net is genius") is probably right about the concept even though this implementation of it underdelivers.

> ⚠️ **The reported Dice and IoU numbers in this notebook are not valid.** See the Known Issues section below — the pixel-accuracy and BCE loss are the only trustworthy metrics here.

---

## Results at a glance

| Project | Task | Best approach | Result |
|---|---|---|---|
| Cutting Stock | Minimize rolls, 4 instances | **Simulated Annealing** | 51 / 78 / 96 / 218 — the only algorithm to beat all four targets |
| Pac-Man | Clear a 43-dot maze | **Q-learning, α=0.5, γ=0.5** | Reward **120** in 52 steps (theoretical max 129) |
| Kvasir-SEG | Polyp segmentation | **Single U-Net** (7.9M params) | Test pixel accuracy **0.890**, BCE 0.261 |

---

## Known issues, quirks and caveats

Worth knowing before running or reusing this code.

### U-Net notebook — the evaluation metrics are broken

This is the most important caveat in the repo. **Neither the "Dice Score" nor the "IoU" measures what it claims to.**

- **`dice_coefficient` does not compute Dice.** It returns `np.sum(y_true == y_pred) / (H*W)` — that is **pixel accuracy**, not $\frac{2|P \cap G|}{|P| + |G|}$. The giveaway is in the output: the reported "Dice Score" (0.8899151611) is *identical* to the Keras test accuracy (0.8899151682). Worse, it compares a **binarized** prediction against a **continuous** ground truth (masks were divided by 255 but never thresholded), so the equality test is only ever satisfied on exact-zero background pixels.
- **`calculate_iou` is model-independent.** It calls `np.logical_and(y_true, y_pred)` on **unthresholded float** arrays, where *any* nonzero value counts as `True`. Sigmoid outputs are essentially never exactly 0, so `y_pred` is `True` almost everywhere → the union is the whole image and the intersection is just the ground-truth mask. The result is a constant that depends only on the data. And indeed both models report **exactly** `0.15464396158854166`, to 16 decimal places. That is not a coincidence — it's the same number twice because the model was never actually consulted.

**Fixes:** threshold the masks at load time (`mask = (mask > 0.5).astype(float)`), threshold predictions at 0.5 inside *both* metric functions, and implement Dice as `2 * (P & G).sum() / (P.sum() + G.sum())`. Expect the corrected numbers to look *worse* than the current pixel accuracy — polyps occupy a minority of each image, so a model predicting "all background" already scores ~85% pixel accuracy. This is precisely why Dice and IoU are the standard metrics for segmentation and why getting them right matters.

### Other issues

- **U-Net: sigmoid activations in the stacked decoder** (see Project 3) throttle the second model. Change to `relu`.
- **U-Net: the notebook markdown contradicts the results.** Cell 47 says "the second U-net is predicting the masks much better than the first"; the numbers say the opposite (0.845 vs. 0.890). The written `Report.pdf` gets this right — only the notebook prose is wrong.
- **U-Net: documentation/code drift.** Markdown says images are resized to 224×224 and that the stacked input is 256×256×4; the code uses 128×128 and 128×128×4. Comments also say "50 epochs" where the code passes 30 and 20.
- **The folder is named `UNet-KavirSeg`** — the dataset is *Kvasir*-SEG. (The JSON file in the dataset itself is also misspelled `kavsir_bboxes.json`, so this one isn't entirely the author's fault.)
- **Pac-Man: `choose_next(q_values, random_value)` is misleadingly named.** `random_value` is actually the probability of acting *greedily*, not randomly — `if rnd < random_value: return argmax`. The annealing schedule (0.05 → 0.9) is therefore an *exploitation*-increase schedule, which is correct, but the name says the opposite of what the variable does.
- **Pac-Man: `not_wall` returns 1 only when the target cell is `'E'`.** Dots are handled by an earlier branch so the logic works out, but the function's name promises something broader than it delivers.
- **No random seeds anywhere.** HC, SA and the GA are all stochastic, as is the Q-learning exploration. Re-runs will produce different numbers — the roll counts above are one sample, not a guaranteed reproduction. For a fair comparison the three metaheuristics should each be run *n* times and reported as mean ± std (the GA notebook already does this with `s=5`; HC and SA do not).
- **Cutting stock: no compute budget is equalized.** SA performs ~460k cost evaluations; HC performs ~250k; the GA's budget depends on its stopping rule. SA's win is convincing, but a strict comparison would fix the evaluation budget across all three.
- **Cutting stock: `open_file` / `Get_input` hard-codes four filenames** in an if/elif chain rather than taking a path. Trivially generalizable to `f'input{i}.stock'`.
- **The GA sorts its population with an O(n²) nested loop plus `np.append` in a loop**, which reallocates the whole array every iteration. `np.argsort` would replace ~15 lines and run orders of magnitude faster.

---

## Ideas for improvement

- **Cutting stock:** replace next-fit with **first-fit-decreasing** as the decoder — it's a strictly better greedy and would likely shave rolls off every result for free. Then try 2-opt / insertion neighbourhoods instead of pure swaps, and give the GA a proper permutation mutation (inversion or scramble of a *segment*, not a single transposition — the function is even *named* `scramble_mutation` but implements a swap).
- **Cutting stock:** run each algorithm 30× per instance and report mean ± std with a fixed evaluation budget. Add a lower bound (⌈Σrequests / L⌉) to see how far from optimal these solutions actually are.
- **Pac-Man:** the local-window abstraction is the ceiling. Try adding a coarse "direction to nearest dot" feature to the state, or move to **approximate Q-learning** with a feature vector (distance to nearest dot, number of dots remaining, …) — this is the standard next step and would fix both the loops and the generalization failure.
- **U-Net:** fix the metrics first (see above), then train with **Dice loss** or a BCE+Dice combination rather than pure BCE — segmentation with a small foreground class is exactly the case BCE handles poorly. Add data augmentation (`RandomRotation` is imported but never used), batch normalization in the conv blocks, and swap the stacked decoder's sigmoids for ReLU.

---

## Author

Coursework by **Arad Vazirpanah** (student no. 610399182), for the Artificial Intelligence course taught by **Dr. Sajedi** at the School of Mathematics, Statistics and Computer Science, College of Science, **University of Tehran**.

Written reports accompany two of the three projects: [`AI_HW2.pdf`](HillClimbing-SA-Genetic-stock/AI_HW2.pdf) (the cutting-stock assignment brief) and [`Report.pdf`](UNet-KavirSeg/Report.pdf) (the full U-Net write-up).

---

## License

No license file is currently included, which means the code is under exclusive copyright by default. If you'd like others to reuse it, consider adding a `LICENSE` (MIT is a common choice for coursework).
