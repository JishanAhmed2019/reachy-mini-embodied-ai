# From Text to Behavior: Making Probabilistic AI Visible with Embodied Robots

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Weber State](https://img.shields.io/badge/Institution-Weber%20State%20University-purple.svg)](https://www.weber.edu)

> A classroom experiment using Reachy Mini programmable robots and LLMs (OpenAI / Gemini) to make probabilistic AI behavior **observable and measurable** — quantifying prompt sensitivity, behavioral variability, and failure modes in education and production settings.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Key Findings](#key-findings)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Technologies](#technologies)
- [Presentation](#presentation)
- [Author](#author)

---

## Project Overview

LLM outputs are probabilistic — the same prompt can produce different responses. In text-only interfaces this is invisible. When AI controls a physical robot, that variability becomes **observable, repeatable, and measurable**.

This project builds an embodied AI framework to:

1. **Connect** student voice/text prompts → OpenAI/Gemini API → token sampling → motion command parser → Reachy Mini robot
2. **Quantify** prompt sensitivity: how much does a word change produce a behavioral shift?
3. **Analyze** three failure modes: stochastic variability, sensitivity to phrasing, adversarial/out-of-scope inputs
4. **Measure** behavioral overlap in embedding space to characterize the success/failure decision boundary

**Research question:** *Can embodied robot feedback make students move from passive observers of AI outputs to active analysts of AI behavior?*

---

## System Architecture

```
┌──────────────────┐
│ Student Voice /  │  Spoken or typed prompt
│ Text Input       │
└────────┬─────────┘
         │ audio / text
         ▼
┌──────────────────┐
│ Pollen Robotics  │  Speech-to-text + session interface
│ App              │
└────────┬─────────┘
         │ HTTPS POST  prompt + context
         ▼
┌──────────────────┐
│ OpenAI / Gemini  │  Transformer · temperature · P(token)
│ LLM API          │
└────────┬─────────┘
         │ token stream
         ▼
┌──────────────────┐
│ Motion Command   │  Maps text response → robot action
│ Parser           │
└────────┬─────────┘
         │ motor commands
         ▼
┌──────────────────┐
│ Reachy Mini      │  Visible physical behavior
│ (Physical/Sim)   │
└──────────────────┘
```

**Learning loop:** You Speak → AI Decides → Robot Acts → Student Observes → Repeat

---

## Key Findings

| Metric | Value |
|--------|-------|
| Total prompts collected | 26 across 3 rounds |
| Success rate | 50% (13/26) |
| Failure rate | 35% (9/26) |
| Partial rate | 15% (4/26) |
| Within-success cosine similarity | 0.171 |
| Within-failure cosine similarity | 0.061 |
| Cross-group (success vs failure) similarity | 0.110 |

**Key result:** All cross-group mean cosine similarities < 0.22, confirming **no outcome group is semantically well-separated** in TF-IDF embedding space — robot behavior cannot be predicted from surface-level text similarity alone. This demonstrates the inherent probabilistic nature of LLM-driven systems.

### Three Observed Failure Modes

| Type | Example | Finding |
|------|---------|---------|
| **Stochastic Variability** | "Can you dance?" × 3 identical trials | Produced 3 different motions |
| **Prompt Sensitivity** | "Can you spin" → success / "Can you spin your head" → failure | 2 added words broke the token distribution |
| **Adversarial/OOScope** | "Explain quantum physics" → refusal | System cannot translate abstract knowledge → physical motion |

---


## Setup & Installation

```bash
git clone https://github.com/JishanAhmed2019/reachy-mini-embodied-ai.git
cd reachy-mini-embodied-ai

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

jupyter lab
```

---

## Dataset

26 student prompts collected across 3 experiment rounds, each labeled with outcome and observed robot response.

| Column | Description |
|--------|-------------|
| `round` | Experiment round (1 = Variability, 2 = Sensitivity, 3 = Failure Analysis) |
| `prompt` | Raw student text input |
| `outcome` | `success` / `partial` / `failure` |
| `robot_response` | Observed physical behavior |

The dataset is self-contained — no external files required. All 26 prompts are embedded directly in `src/prompt_dataset.py` and both notebooks.

---

## Methodology

- Experimental design (3-round structure: Variability → Sensitivity → Failure)
- TF-IDF embedding strategy (word + character n-grams, L2-normalized)
- LSA/SVD compression to 16-D
- Dimensionality reduction comparison: PCA vs t-SNE (perplexity=7) vs MDS
- Delta score: `Sim(success_centroid) − Sim(failure_centroid)` as a prompt quality signal
- Cross-group similarity matrix interpretation
---

## Technologies

| Category | Tools |
|----------|-------|
| **Robot** | Reachy Mini (Pollen Robotics), simulation + physical |
| **LLM APIs** | OpenAI GPT, Google Gemini |
| **NLP** | `scikit-learn` TF-IDF (word + char n-grams), LSA/SVD |
| **Dimensionality Reduction** | PCA, t-SNE, MDS |
| **Visualization** | `matplotlib`, `seaborn`, convex hull overlays |
| **Interface** | Pollen Robotics App (voice/text input) |

---

## Presentation

This work was presented at **Weber State University** as a classroom experiment in AI education.  
Slides: [`reports/PresentationsJishan.pdf`](reports/PresentationsJishan.pdf)

---

## Author

**Dr. Jishan Ahmed**  
Data Science Assistant Professor · Weber State University

- [LinkedIn](https://www.linkedin.com/in/jishan-ahmed-689aa63b)
- [GitHub](https://github.com/JishanAhmed2019)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

> **Note:** The physical Reachy Mini robot was on order at time of experiments; all trials were conducted in simulation mode via the Pollen Robotics App. Results reflect simulated robot behavior driven by live LLM inference.
