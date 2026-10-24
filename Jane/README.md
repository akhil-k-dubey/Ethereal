# Project Jane — Phase 1: The Neural Engine (The Empty Architecture)

> *"Tony didn't write rules for how to talk or how to engineer. Instead, he wrote a learning algorithm. He created a system where the software was capable of forming connections (the dots), but there were no dots there yet."*

Phase 1 constructs Jane's core **Neural Engine**: the foundational mathematical framework, continuous latent manifold, and learning algorithms before populating the knowledge graph (Phase 2) or multimodal NLP layers (Phase 3).

---

## 🧠 Architectural Pillars

To eliminate hallucinations and make math native to Jane's thinking model, standard probability-based neural nodes are modified across four key dimensions:

### 1. Neural Arithmetic Logic Units (NALUs)
- **Problem**: Standard neurons are probability gates that guess number curves rather than computing them.
- **Solution**: Specific clusters of the network are wired with NALU neurons capable of exact $+$, $-$, $\times$, and $\div$.
- **Result**: When parsing numerical inputs (e.g. *"Adjust power to 40%"*), values route into NALU clusters. The calculated mathematical output anchors directly into the continuous latent space as deterministic math nodes.

### 2. Differentiable Neural Computers (DNC Memory Tape)
- **Problem**: Standard networks try to solve multi-step problems in a single pass.
- **Solution**: An internal Turing-machine memory tape ($M \in \mathbb{R}^{S \times W}$) with differentiable content-addressing, erase gates, and read/write heads.
- **Result**: Jane performs step-by-step reasoning inside her own latent space—writing intermediate variables to tape, executing intermediate operations, reading accumulated facts, and progressing sequentially without guessing.

### 3. Energy-Based Constraints (Physics Brick Walls)
- **Problem**: Standard models output whatever has the highest softmax probability, even if physically impossible.
- **Solution**: The continuous manifold is governed by an Energy-Based Model (EBM) with hard physical invariant barriers:
  $$E_{\text{total}}(z) = E_{\text{learned}}(z) + E_{\text{physics\_barrier}}(z)$$
- **Result**: Physical impossibilities (e.g., negative mass, spatial collisions, conservation violations) have infinite potential energy barriers. As energy flows through the manifold, invalid paths act like brick walls: the energy bounces back, forcing vector flow exclusively into mathematically and physically consistent valleys.

### 4. Latent Space Reasoning (System 2 Equilibrium)
- **Problem**: Token-by-token generation guesses output words sequentially without checking work.
- **Solution**: Jane is forbidden from generating text immediately. When stimulated by an input concept, energy bounces iteratively between the Relational Manifold, NALUs, and DNC memory tape while descending the energy landscape.
- **Result**: Jane "thinks" in pure math and concepts inside the latent manifold until energy reaches equilibrium. Only then is the settled latent state decoded into English.

### 5. Deterministic Symbolic Verifier (Z3 & Clingo Gatekeeper)
- An external deterministic proof engine verifies all mathematical assertions and spatial trajectories against formal rules in `configs/solver_rules/` before real-world execution.

---

## 📁 Directory Structure

```
Project_Jane/
├── configs/                         # All hyperparameters and settings
│   ├── transformer_config.yaml     # Settings for the translation model
│   ├── reasoning_engine.yaml        # NALU, DNC, Manifold, and Energy parameters
│   └── solver_rules/                # Human-supplied logic rules (Clingo/Z3)
│       └── physics_rules.yaml       # Inviolable physical invariants
│
├── data/                            # Datasets for training and testing
│   ├── knowledge_graph/             # Engineering/physics principles for manifold mapping
│   ├── text_puzzles/                # Constraint-solving puzzles for benchmarks
│   └── grid_world_maps/            # Sandboxed RL environment maps
│
├── src/                             # Main source code
│   ├── translator/                  # THE TRANSFORMER (Language <-> Vectors ONLY)
│   │   ├── encoder.py               # Parses text into 256D continuous vector space
│   │   ├── decoder.py               # Translates proven vectors back to English
│   │   └── tokenization.py          # Vocabulary & numeric entity extractor
│   │
│   ├── reasoning_engine/           # THE NEURAL BRAIN (Probabilistic Logic + Native Math)
│   │   ├── nalu_arithmetic.py       # Neural Arithmetic Logic Units (Math nodes)
│   │   ├── dnc_memory.py            # Differentiable Neural Computer (Scratchpad tape)
│   │   ├── energy_constraints.py    # Energy-based models with physics brick walls
│   │   └── manifold.py              # 256D Relational Manifold + 3D presentation projection
│   │
│   ├── symbolic_verifier/          # THE DETERMINISTIC PROVER (Zero Hallucinations)
│   │   ├── z3_interface.py         # Math/logic proof engine (Z3)
│   │   ├── clingo_interface.py     # Spatial/ASP proof engine (Clingo)
│   │   └── verifier_orchestrator.py# Verifies if neural output is safe to execute
│   │
│   └── core/                        # Main pipeline tying it all together
│       └── pipeline.py              # Flow: Input -> Translate -> Reason -> Verify -> Output
│
├── environments/                    # Simulation spaces for RL and Benchmarks
│   ├── grid_world/                  # Sandboxed 10x10 spatial grid
│   │   ├── agent_controller.py      # Jane's agent in the grid
│   │   ├── physics_engine.py        # Hard rules of the grid (walls, hazards)
│   │   └── reward_function.py       # RL scoring system
│   │
│   └── text_logic_sim/              # Text-based constraint simulation
│       └── puzzle_generator.py      # Procedural zero-shot constraint puzzles
│
├── training/                        # Scripts to train the models
│   ├── 1_pretrain_manifold.py       # Self-supervised learning on text data
│   ├── 2_train_energy_constraints.py# Train the Energy-Based boundaries
│   ├── 3_train_rl_grid.py           # Reinforcement Learning in Grid World
│   └── 4_neuro_symbolic_align.py    # Train neural net to output Z3-readable logic
│
├── benchmarks/                      # Phase 1 Success Criteria
│   ├── eval_relational_memory.py    # Text constraint-solving tests
│   ├── eval_grid_world.py           # Spatial RL pathfinding tests
│   └── eval_hallucination_rate.py   # Measuring if Z3 rejects neural guesses
│
├── scripts/                         # Utility scripts
│   ├── run_text_benchmark.sh        # Quick run for text logic puzzles
│   └── run_grid_benchmark.sh        # Quick run for RL grid world
│
└── logs/                            # Output metrics and tensorboard logs
    └── training_runs/
```

---

## 🚀 Quickstart

### 1. Interactive Demo
```bash
python3 main.py --text "Adjust reactor output to 40% and calibrate thruster" --device cuda
```

### 2. Run Benchmarks
```bash
./scripts/run_text_benchmark.sh
./scripts/run_grid_benchmark.sh
```

### 3. Run Tests
```bash
pytest tests/test_phase1_neural_engine.py
```
