# Logic Engine — Technical Documentation

> A domain-agnostic constraint-aware recommendation framework for personalized learning using Deep Knowledge Tracing.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://xbo78uswdvbsmsnka87e4j.streamlit.app/)
[![HuggingFace](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/Clementio/PLRS)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/clementina-tom/PLRS)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Quick Start](#quick-start)
5. [Knowledge Map Format](#knowledge-map-format)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Model Details](#model-details)
9. [Evaluation Results](#evaluation-results)
10. [Deployment](#deployment)
11. [Extending to a New Domain](#extending-to-a-new-domain)
12. [Limitations](#limitations)
13. [Roadmap](#roadmap)

---

## Overview

The Logic Engine solves two fundamental problems in educational recommendation:

**Problem 1 — The Prerequisite Trap:** Data-driven recommenders suggest advanced content to students who lack foundational knowledge, because they optimise for similarity rather than pedagogical correctness.

**Problem 2 — Domain Lock:** Systems trained on one subject corpus cannot transfer to another without full retraining.

**The solution:** a two-layer architecture that separates *learner behaviour modelling* (handled by a SAKT neural network, domain-agnostic) from *domain knowledge representation* (handled by a configurable DAG, fully swappable).

### Key results

| System | Violation Rate (Math) | Violation Rate (CS) | Domain Agnostic |
|--------|----------------------|---------------------|-----------------|
| Collaborative Filtering | 81.2% | 57.8% | No |
| Matrix Factorisation | 81.3% | 55.2% | No |
| **Logic Engine** | **0.0%** | **0.0%** | **Yes** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Logic Engine Pipeline                 │
│                                                         │
│  Student Interaction Sequence                           │
│           │                                             │
│           ▼                                             │
│  ┌─────────────────┐                                    │
│  │   Layer 1:      │  SAKT Transformer                  │
│  │   Neural Layer  │  Predicts mastery probability      │
│  │   (SAKT Model)  │  per skill — domain agnostic       │
│  └────────┬────────┘                                    │
│           │ Mastery Vector                              │
│           ▼                                             │
│  ┌─────────────────┐  ┌──────────────────────┐         │
│  │   Layer 2:      │◄─│  Knowledge Map (DAG) │         │
│  │   Constraint    │  │  Swappable per domain│         │
│  │   Layer (DAG)   │  └──────────────────────┘         │
│  └────────┬────────┘                                    │
│           │ Approved / Challenging / Vetoed             │
│           ▼                                             │
│  ┌─────────────────┐                                    │
│  │   Layer 3:      │  Multi-objective score:            │
│  │   Ranking       │  mastery gap + readiness           │
│  │   Function      │  + downstream importance           │
│  └────────┬────────┘                                    │
│           │                                             │
│           ▼                                             │
│  Top-N Recommendations with Reasoning Tags             │
└─────────────────────────────────────────────────────────┘
```

### Three-tier constraint system

| Status | Condition | Meaning |
|--------|-----------|---------|
| ✅ Approved | All prerequisites ≥ mastery threshold | Ready to proceed |
| ⚠️ Challenging | All prerequisites ≥ soft threshold, some < mastery threshold | Proceed with caution |
| ❌ Vetoed | Any prerequisite < soft threshold | Not yet ready |

Default thresholds: mastery = 0.70, soft = 0.50 (both configurable).

---

## Repository Structure

```
PLRS/
├── app.py                    # Streamlit dashboard (main entry point)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── knowledge_maps/
│   ├── math_dag.json         # Secondary School Mathematics (JSS3–SS2)
│   └── cs_dag.json           # CS Fundamentals (JSS3–SS2)
│
└── data/
    └── skill_encoder.csv     # OULAD skill ID → activity type mapping
```

**Model files** are hosted on HuggingFace (too large for GitHub):
- `sakt_model.pt` — trained SAKT weights
- `config.json` — model hyperparameters

---

## Quick Start

### Run locally

```bash
# Clone repository
git clone https://github.com/clementina-tom/PLRS.git
cd PLRS

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

### Use the hosted demo

Visit: https://xbo78uswdvbsmsnka87e4j.streamlit.app/

### Use the model programmatically

```python
from huggingface_hub import hf_hub_download
import torch
import json

# Download model
config_path = hf_hub_download(repo_id="Clementio/PLRS", filename="config.json")
model_path  = hf_hub_download(repo_id="Clementio/PLRS", filename="sakt_model.pt")

with open(config_path) as f:
    config = json.load(f)

# Load model (see Model Details for architecture)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Run inference on a student sequence
skill_sequence   = [142, 89, 203, 142, 89]   # OULAD skill IDs
correct_sequence = [1, 0, 1, 1, 0]           # 1=correct, 0=incorrect

mastery_probs = run_sakt_inference(model, config, skill_sequence, correct_sequence, 'cpu')
# Returns: {skill_id: mastery_probability, ...}
```

---

## Knowledge Map Format

Knowledge maps are JSON files with two keys: `nodes` and `edges`.

### Node schema

```json
{
  "id":    "algebraic_expressions",
  "label": "Algebraic Expressions",
  "level": "JSS3",
  "term":  1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier, used in edges |
| `label` | string | Human-readable display name |
| `level` | string | Curriculum level (e.g. JSS3, SS1, SS2) |
| `term` | integer | Term within the level (1, 2, or 3) |

### Edge schema

```json
{"from": "algebraic_expressions", "to": "algebraic_factorization"}
```

An edge `A → B` means **A is a prerequisite of B**. The student must master A before B can be recommended.

### Requirements

- The graph must be a **Directed Acyclic Graph** (no cycles)
- All node IDs referenced in edges must exist in the nodes list
- The system validates DAG integrity on load

### Example: adding a new domain

```json
{
  "domain": "Physics",
  "scope":  "SS1–SS2",
  "version": "1.0",
  "nodes": [
    {"id": "basic_mechanics",  "label": "Basic Mechanics",    "level": "SS1", "term": 1},
    {"id": "energy",           "label": "Energy & Work",      "level": "SS1", "term": 2},
    {"id": "waves",            "label": "Waves & Oscillation","level": "SS2", "term": 1}
  ],
  "edges": [
    {"from": "basic_mechanics", "to": "energy"},
    {"from": "energy",          "to": "waves"}
  ]
}
```

No model retraining required. Drop this file in `knowledge_maps/` and select Physics in the dashboard.

---

## API Reference

### `MasteryVector`

Holds a learner's current mastery state.

```python
mv = MasteryVector(graph, threshold=0.70)
mv.update('algebraic_expressions', 0.82)
mv.is_mastered('algebraic_expressions')  # True
mv.get_mastery('fractions')              # 0.0 (default)
mv.get_mastery_summary()
# {'total_topics': 38, 'mastered': 5, 'mastery_rate': 0.132, 'mastered_topics': [...]}
```

### `DAGConstraintLayer`

Validates candidate topics against prerequisites.

```python
constraint = DAGConstraintLayer(graph, threshold=0.70, soft_threshold=0.50)
status, reasoning = constraint.validate('algebraic_factorization', mastery_vector)
# status:    'approved' | 'challenging' | 'vetoed'
# reasoning: human-readable explanation string
```

### `RankingFunction`

Scores approved topics by multi-objective ranking.

```python
ranker = RankingFunction(graph, threshold=0.70, w_gap=0.40, w_ready=0.35, w_downstream=0.25)
score  = ranker.score('indices', mastery_vector)
# Returns float in [0, 1]
```

### `LearningRecommendationPipeline`

Full pipeline combining all three layers.

```python
pipeline = LearningRecommendationPipeline(
    graph          = math_graph,
    threshold      = 0.70,
    soft_threshold = 0.50,
    top_n          = 5
)
output = pipeline.run(mastery_vector)
```

**Output schema:**

```python
{
    'top_recommendations': [          # approved topics, ranked
        {
            'topic_id':    str,
            'topic_label': str,
            'mastery':     float,     # current mastery [0, 1]
            'score':       float,     # ranking score [0, 1]
            'status':      str,       # 'approved'
            'reasoning':   str,       # human-readable explanation
        },
        ...
    ],
    'challenging': [...],             # challenging topics, top 3
    'total_approved':   int,
    'total_challenging':int,
    'total_vetoed':     int,
    'vetoed_sample':    [...],        # up to 5 vetoed topics
    'prerequisite_violation_rate': float,
}
```

### `what_if_analysis`

Explores DAG structure for a given topic.

```python
result = what_if_analysis('trigonometric_ratios', math_graph)
# {
#   'direct_unlocks':  ['Sine & Cosine Rules', 'Bearings & Distances'],
#   'all_unlocks':     [...],
#   'blocked_by':      ['Pythagoras Theorem'],
#   'total_unlocked':  2
# }
```

---

## Configuration

### Mastery thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.70 | Minimum SAKT mastery probability to consider a topic mastered |
| `soft_threshold` | 0.50 | Below this = hard veto. Between this and threshold = challenging |

Based on Bloom's mastery learning theory (70–80% competency for curriculum progression).

**The threshold is the single most influential configuration parameter in the system.** It is not merely a sensitivity dial — it is a curriculum pacing decision with cascading effects across the entire recommendation pipeline.

Observed effects of threshold adjustment (same learner, seed 1000, length 150):

| Threshold | Topics Mastered | Approved | Challenging | Violation Rate |
|-----------|----------------|----------|-------------|----------------|
| 0.65 | 5 / 38 | 6 | 9 | 47% |
| 0.75 | 1 / 38 | 2 | 17 | 47% |

At 0.65 — topics the learner has at 65–74% mastery are considered mastered, unlocking their dependents throughout the DAG. At 0.75 — those same topics fall below threshold, collapsing the available curriculum toward foundations only.

**Deployment guidance:**
- **0.60–0.65:** Progress-focused. Suitable for self-paced learners who benefit from breadth exploration.
- **0.70** (default): Balanced. Aligns with Bloom's mastery learning consensus.
- **0.75–0.80:** Mastery-focused. Suitable for high-stakes curricula where foundational gaps are costly (e.g. mathematics, programming).

### DAG Mastery Cascade

The system includes a cascade inference mechanism: if a learner demonstrates high mastery on an advanced topic, their prerequisite mastery is inferred proportionally upward through the DAG. This reflects the realistic assumption that a student scoring 80% on Modular Arithmetic almost certainly understands Whole Numbers, even without direct evidence.

```python
# Cascade rule: prerequisite mastery inferred as 85% of descendant mastery
inferred_prereq_mastery = min(descendant_mastery * 0.85, 0.95)
```

The cascade runs iteratively until no further updates occur, propagating through the full ancestor chain. This prevents the cold-start problem where sparse interaction data causes the system to treat experienced learners as complete beginners.

### Ranking weights

| Parameter | Default | Effect |
|-----------|---------|--------|
| `w_gap` | 0.40 | Higher = prioritise topics closest to mastery |
| `w_ready` | 0.35 | Higher = prioritise topics with most prerequisites met |
| `w_downstream` | 0.25 | Higher = prioritise topics that unlock more of the curriculum |

Weights must sum to 1.0.

**Near-mastery boost:** topics where the learner already has 10–69% mastery receive an additional score boost (up to +0.15) proportional to their current mastery. This ensures topics the learner is actively working on rank above completely untouched topics with the same prerequisite readiness.

### Model config (`config.json`)

```json
{
    "num_skills":    5889,
    "embed_dim":     128,
    "num_heads":     8,
    "num_layers":    2,
    "max_seq_len":   200,
    "dropout":       0.2,
    "batch_size":    64,
    "learning_rate": 0.001
}
```

---

## Model Details

### SAKT (Self-Attentive Knowledge Tracing)

Architecture based on Pandey and Karypis (2019) with modifications:

- **Input encoding:** `interaction_id = skill_id + correct * num_skills`
- **Positional encoding:** learned embeddings for sequence position
- **Attention:** 8-head multi-head self-attention with causal masking
- **Stability:** pre-norm (`norm_first=True`), padded positions zeroed before/after transformer
- **Output:** sigmoid over linear projection → mastery probability

**Training data:** OULAD (Open University Learning Analytics Dataset)
- 2,500,366 interaction records
- 17,507 students, 5,736 unique skills
- 70/15/15 train/val/test split
- Balanced correctness signal (50% correct / 50% incorrect)

**Performance:** Val AUC 0.7692 (published SAKT range on OULAD: 0.72–0.80)

---

## Evaluation Results

Full evaluation on 200 test students from OULAD test set.

### Primary metric: Prerequisite Violation Rate

```
What fraction of the top-10 recommendations violate at least 
one direct prerequisite dependency in the knowledge DAG?
```

| System | Math | CS | Severe/Student |
|--------|------|-----|----------------|
| CF | 81.2% | 57.8% | 5.86 |
| MF | 81.3% | 55.2% | 5.83 |
| **Logic Engine** | **0.0%** | **0.0%** | **0.00** |

### Complete evaluation table

| Metric | CF | MF | Logic Engine |
|--------|----|----|-------------|
| Violation Rate (Math) | 81.2% | 81.3% | **0.0%** |
| Violation Rate (CS) | 57.8% | 55.2% | **0.0%** |
| Severe Violations/Student | 5.86 | 5.83 | **0.00** |
| Moderate Violations/Student | 0.51 | 0.54 | **0.00** |
| Minor Violations/Student | 1.56 | 1.67 | **0.00** |
| Curriculum Alignment Score | 0.877 | 0.877 | 0.500* |
| Precision@10 | 0.000 | 0.000 | DAG-filtered |
| Val AUC | N/A | N/A | 0.7692 |
| Domain Agnostic | No | No | **Yes** |
| Prerequisite Aware | No | No | **Yes** |

*Lower curriculum alignment score reflects the Logic Engine recommending the *next* learning step (intentionally one level ahead) rather than familiar same-level content. See Limitations for interpretation.

---

## Deployment

### Streamlit Community Cloud (current)

The app is deployed at: https://xbo78uswdvbsmsnka87e4j.streamlit.app/

Connected to GitHub repo `clementina-tom/PLRS`, branch `main`. Redeploys automatically on push.

Model files served from HuggingFace Hub (`Clementio/PLRS`) at runtime via `hf_hub_download`.

### Local deployment

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Docker (future)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

---

## Extending to a New Domain

Adding a new subject domain requires **no model retraining**. The process:

**Step 1: Define your topic list**
List all topics in the subject with their curriculum level and term.

**Step 2: Define prerequisite relationships**
For each topic, identify which topics must be mastered first.

**Step 3: Create the JSON knowledge map**
Follow the schema in [Knowledge Map Format](#knowledge-map-format).

**Step 4: Validate the DAG**
```python
import networkx as nx, json

with open('your_domain.json') as f:
    data = json.load(f)

G = nx.DiGraph()
for node in data['nodes']:
    G.add_node(node['id'])
for edge in data['edges']:
    G.add_edge(edge['from'], edge['to'])

print(f"Is valid DAG: {nx.is_directed_acyclic_graph(G)}")
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
```

**Step 5: Add to app**
Place the file in `knowledge_maps/` and add the domain name to the `selectbox` in `app.py`.

---

## Limitations

**1. Skill-to-topic bridge is coarse**
The mapping from OULAD VLE activity types to curriculum topic IDs uses activity type as a proxy (e.g. all `oucontent` resources map to `algebraic_expressions`). This produces sparse mastery vectors for most students. A fine-grained mapping would significantly improve mastery estimates.

**2. Trained on UK university data**
OULAD reflects Open University UK students. The learner behaviour patterns may differ from Nigerian secondary school students. Domain-specific interaction data would improve both accuracy and cultural relevance.

**3. No forgetting curve modelling**
SAKT treats all past interactions equally regardless of recency. Interactions from 6 months ago carry the same weight as interactions from yesterday. Integrating exponential decay (Ebbinghaus forgetting curve) into the attention weights would improve mastery estimates.

**4. Cold start**
New students with no interaction history get zero mastery estimates and receive only foundational topic recommendations. A diagnostic assessment module (10–15 adaptive questions) would bootstrap the mastery vector for new users.

**5. No user study for RQ5**
The explainability claim (that reasoning-tagged recommendations are followed more than untagged ones) has not been validated empirically. A controlled user study is required to answer this question.

---

## Roadmap

### Near-term
- [ ] Forgetting curve decay in SAKT attention weights (→ AUC improvement)
- [ ] Spaced repetition signal in ranking function
- [ ] Fine-grained skill-to-topic mapping

### Medium-term
- [ ] FastAPI backend replacing Streamlit for production serving
- [ ] Redis caching for DAG traversals
- [ ] Upgrade to AKT or DTransformer architecture

### Research-level
- [ ] Dynamic prerequisite discovery via GNNs (learning DAG from data)
- [ ] EdBERTa-based automatic cross-curriculum concept alignment
- [ ] Multi-objective Pareto optimisation for ranking

---

## Citation

```bibtex
@misc{logic-engine-2026,
  author = {[Author Name]},
  title  = {Logic Engine: A Domain-Agnostic Constraint-Aware 
            Recommendation Framework for Personalized Learning
            Using Deep Knowledge Tracing},
  year   = {2026},
  url    = {https://github.com/clementina-tom/PLRS}
}
```

---

## References

- Pandey, S. and Karypis, G. (2019) *A self-attentive model for knowledge tracing.* EDM 2019.
- Piech, C. et al. (2015) *Deep knowledge tracing.* NeurIPS 2015.
- Kuzilek, J. et al. (2017) *Open University Learning Analytics dataset.* Scientific Data.
- Corbett, A.T. and Anderson, J.R. (1994) *Knowledge tracing.* UMUAI.

---

*Department of Computer Science | Final Year Project 2026*
