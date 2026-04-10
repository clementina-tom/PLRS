# Logic Engine: Teaching AI to Respect What Students Don't Know Yet

*A complete walkthrough of building a constraint-aware personalized learning recommender using SAKT, DAG prerequisite enforcement, and multi-objective ranking — trained on OULAD, evaluated on the Nigerian secondary school curriculum.*

---

## The Problem Nobody Talks About in Educational AI

Most recommender systems are built to maximize engagement. For instance Netflix recommends what you'll watch next, Spotify recommends what songs you'll enjoy, Amazon recommends what you'll buy.

Some educational recommender systems copy this playbook. How? They find students with similar interaction histories and recommend what those students studied next.

As harmless as that seems, it has its flaw: **a student with no foundation in Basic Algebra should not be recommended Advanced Calculus, no matter how many similar students studied it.**

This is the **prerequisite trap** — and it's baked into every purely data-driven educational recommender that ignores curriculum structure.

This notebook documents how I built a system that refuses to fall into it.

---

## What We're Building

The **Logic Engine** is a three-layer recommendation framework:

```
Layer 1 — Neural:      SAKT model predicts mastery probability per skill
Layer 2 — Constraint:  DAG vetoes recommendations that violate prerequisites  
Layer 3 — Ranking:     Multi-objective scorer orders approved candidates
```

The key architectural insight: **learner behaviour is domain-agnostic, but domain knowledge is not.**

The SAKT model learns *how students learn* — the patterns of success, failure, recency, and forgetting. The DAG encodes *what students should learn* in a specific subject. By separating these concerns, the system can run on any subject by swapping the knowledge map JSON, without retraining the model.

---

## Our Dataset: OULAD

The [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset) contains anonymised data from 32,593 students across 7 courses.

Key files used:
- `studentVle.csv` — VLE interaction logs (clicks per resource per day)
- `studentAssessment.csv` — assessment scores
- `assessments.csv` — assessment metadata
- `vle.csv` — resource metadata including activity type

**The challenge:** OULAD doesn't have explicit correct/incorrect labels per interaction. VLE data is click counts, not right/wrong answers. This required building a proxy correctness signal.

---

## Part 1: Data Pipeline

### Step 1: Load raw files

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

student_vle = pd.read_csv('studentVle.csv')
vle         = pd.read_csv('vle.csv')
student_ass = pd.read_csv('studentAssessment.csv')
assessments = pd.read_csv('assessments.csv')
student_inf = pd.read_csv('studentInfo.csv')

print(f"VLE interactions: {len(student_vle):,}")
print(f"Unique students:  {student_vle['id_student'].nunique():,}")
print(f"Unique resources: {student_vle['id_site'].nunique():,}")
```

**Output:**
```
VLE interactions: 10,655,280
Unique students:  26,074
Unique resources: 6,268
```

---

### Step 2: Build the correctness proxy

The core insight: a student's correctness on a VLE interaction can be estimated by combining two signals:
- **Assessment signal (0.7 weight):** did the student pass assessments in this course?
- **Engagement signal (0.3 weight):** did the student click above median for this resource?

```python
# Assessment correctness per student per course
ass_merged = student_ass.merge(assessments, on='id_assessment', how='left')
ass_merged['correct'] = (ass_merged['score'] >= 50).astype(int)

student_performance = ass_merged.groupby(
    ['id_student', 'code_module', 'code_presentation']
)['correct'].mean().reset_index()
student_performance.rename(columns={'correct': 'pass_rate'}, inplace=True)

# Merge VLE with performance
vle_merged = student_vle.merge(
    vle, on=['id_site', 'code_module', 'code_presentation'], how='left'
)
vle_merged = vle_merged.merge(
    student_performance,
    on=['id_student', 'code_module', 'code_presentation'],
    how='left'
)

# Composite correctness signal
median_clicks = vle_merged['sum_click'].median()
vle_merged['click_signal'] = (vle_merged['sum_click'] >= median_clicks).astype(float)
vle_merged['correct'] = (
    (0.7 * vle_merged['pass_rate'] + 0.3 * vle_merged['click_signal']) >= 0.5
).astype(int)

print(f"Correctness balance:")
print(vle_merged['correct'].value_counts(normalize=True).round(3))
```

**Output:**
```
Correctness balance:
1    0.502
0    0.498
```

The 50/50 balance is important — a DKT model trained on heavily imbalanced data will default to predicting the majority class.

---

### Step 3: Encode skill IDs and filter sequences

```python
# Encode skill IDs
le = LabelEncoder()
vle_merged['skill_id'] = le.fit_transform(vle_merged['id_site'])

# Build final dataset
dkt_final = vle_merged[['id_student', 'skill_id', 'date', 'correct']].copy()
dkt_final = dkt_final.sort_values(['id_student', 'date']).reset_index(drop=True)

# Filter: keep students with 10-500 interactions
seq_lengths = dkt_final.groupby('id_student').size()
valid_students = seq_lengths[(seq_lengths >= 10) & (seq_lengths <= 500)].index
dkt_final = dkt_final[dkt_final['id_student'].isin(valid_students)]

# Truncate to last 200 interactions per student (recency bias)
dkt_final = (
    dkt_final
    .sort_values(['id_student', 'date'])
    .groupby('id_student')
    .tail(200)
    .reset_index(drop=True)
)

print(f"Final dataset: {dkt_final.shape}")
print(f"Unique students: {dkt_final['id_student'].nunique():,}")
print(f"Unique skills:   {dkt_final['skill_id'].nunique():,}")
```

**Output:**
```
Final dataset: (2,500,366, 4)
Unique students: 17,507
Unique skills:   5,736
```

**Why truncate to last 200?** SAKT uses transformers which scale quadratically with sequence length. More importantly, recent interactions are stronger predictors of current mastery than interactions from months ago. The last 200 steps is the right tradeoff.

---

## Part 2: SAKT Model

### Why SAKT over vanilla LSTM-DKT?

Standard DKT uses an LSTM that processes interactions sequentially. By the time it reaches step 150, the signal from step 1 has been compressed through 149 hidden state updates and is largely lost.

SAKT uses self-attention — every interaction can directly attend to every other interaction regardless of distance. A student's struggle with fractions at step 10 is directly accessible when predicting their performance on algebra at step 150.

```
DKT  (LSTM):  h_t = f(h_{t-1}, x_t)           — sequential, forgets
SAKT (Attn):  h_t = attention(x_t, x_1...x_{t-1})  — global, remembers
```

### Architecture

```python
import torch
import torch.nn as nn

class SAKT(nn.Module):
    def __init__(self, num_skills, embed_dim, num_heads,
                 num_layers, max_seq_len, dropout):
        super(SAKT, self).__init__()
        self.num_skills = num_skills
        
        # Encode both WHAT was attempted and WHETHER it was correct
        # interaction_id = skill_id + correct * num_skills
        self.interaction_embed = nn.Embedding(
            num_skills * 2 + 1, embed_dim, padding_idx=0
        )
        self.skill_embed = nn.Embedding(num_skills + 1, embed_dim, padding_idx=0)
        self.pos_embed   = nn.Embedding(max_seq_len + 1, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=embed_dim * 4,
            norm_first=True  # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.dropout = nn.Dropout(dropout)
        self.output  = nn.Linear(embed_dim, 1)

    def forward(self, interactions, target_skills, mask):
        batch_size, seq_len = interactions.shape
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        x = self.interaction_embed(interactions) + self.pos_embed(positions)
        
        # Zero padded positions — cleaner than src_key_padding_mask
        # which causes NaN issues in PyTorch's TransformerEncoder
        x = x * mask.unsqueeze(-1).float()
        x = self.dropout(x)
        
        # Causal mask: model can only see past interactions
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf')), diagonal=1
        )
        x = self.transformer(x, mask=causal_mask, is_causal=False)
        x = x * mask.unsqueeze(-1).float()
        
        # Inject target skill signal before output projection
        x = x + self.skill_embed(target_skills)
        return self.output(x).squeeze(-1)  # raw logits
```

**Config used:**
```python
CONFIG = {
    'num_skills':    5889,   # derived from actual data max
    'embed_dim':     128,
    'num_heads':     8,
    'num_layers':    2,
    'max_seq_len':   200,
    'dropout':       0.2,
    'batch_size':    64,
    'learning_rate': 0.001,
    'epochs':        30,
    'patience':      5,
}
# Total trainable parameters: ~2.6M
```

---

### The input encoding trick

The interaction encoding `skill_id + correct * num_skills` is the standard DKT encoding — it creates a unique token for every (skill, correctness) pair. Getting this wrong (not shifting input/target) is the most common source of data leakage in DKT implementations.

```python
# CORRECT: input is t=0..n-1, target is t=1..n
input_interactions = [
    skills[i] + corrects[i] * num_skills
    for i in range(len(skills) - 1)  # ← exclude last
]
target_skills  = skills[1:]   # ← predict NEXT skill
target_correct = corrects[1:] # ← from NEXT correctness
```

```python
# WRONG (data leakage): using current correctness as input
# AND as target simultaneously
```

---

### Training

```python
criterion = nn.BCEWithLogitsLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, all_labels, all_preds = 0.0, [], []
    
    with torch.set_grad_enabled(train):
        for batch in loader:
            interactions   = batch['interactions'].to(device)
            target_skills  = batch['target_skills'].to(device)
            target_correct = batch['target_correct'].to(device)
            mask           = batch['mask'].to(device)
            
            logits = model(interactions, target_skills, mask)
            
            # Only compute loss on real (non-padded) positions
            loss_per_step = criterion(logits, target_correct)
            mask_float    = mask.float()
            loss = (loss_per_step * mask_float).sum() / mask_float.sum().clamp(min=1)
            
            if torch.isnan(loss): continue  # NaN guard
            
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(logits[mask]).detach().cpu().numpy())
            all_labels.extend(target_correct[mask].cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), auc
```

**Training results:**

| Epoch | Train Loss | Train AUC | Val Loss | Val AUC |
|-------|-----------|-----------|----------|---------|
| 1  | 0.6620 | 0.6257 | 0.6208 | 0.6912 |
| 5  | 0.5710 | 0.7542 | 0.5745 | 0.7561 |
| 10 | 0.5591 | 0.7678 | 0.5658 | 0.7658 |
| 21 | 0.5474 | 0.7802 | 0.5610 | 0.7691 |
| 26 | 0.5409 | 0.7867 | 0.5668 | 0.7662 |

**Best Val AUC: 0.7692** (achieved at epoch 21, early stopping at epoch 26)

Train and Val AUC track closely throughout — no overfitting.

---

## Part 3: Knowledge Maps (DAGs)

The prerequisite constraint layer requires a domain knowledge map. These were manually constructed from official NERDC Nigerian curriculum documents for JSS3–SS2, the Lagos State Education Commission and other online sources.

```python
import json
import networkx as nx

def load_dag(path):
    with open(path) as f:
        data = json.load(f)
    G = nx.DiGraph()
    for node in data['nodes']:
        G.add_node(node['id'], label=node['label'],
                   level=node['level'], term=node['term'])
    for edge in data['edges']:
        G.add_edge(edge['from'], edge['to'])
    assert nx.is_directed_acyclic_graph(G), "Cycle detected in knowledge map!"
    return G

math_graph = load_dag('math_dag.json')
cs_graph   = load_dag('cs_dag.json')

print(f"Math DAG: {math_graph.number_of_nodes()} nodes, {math_graph.number_of_edges()} edges")
print(f"CS DAG:   {cs_graph.number_of_nodes()} nodes,  {cs_graph.number_of_edges()} edges")
```

**Output:**
```
Math DAG: 38 nodes, 45 edges
CS DAG:   31 nodes, 39 edges
```

**Sample knowledge map structure (Math):**
```json
{
  "nodes": [
    {"id": "whole_numbers", "label": "Whole Numbers & Basic Operations", "level": "JSS3", "term": 1},
    {"id": "fractions",     "label": "Fractions",                        "level": "JSS3", "term": 1}
  ],
  "edges": [
    {"from": "whole_numbers", "to": "fractions"}
  ]
}
```

**Longest prerequisite chains:**
```
Math: whole_numbers → fractions → statistics_basic → statistics_frequency 
      → statistics_advanced → cumulative_frequency  (6 steps)

CS:   computer_basics → programming_concepts → variables_operators 
      → control_structures → python_basics → python_projects 
      → data_structures_algorithms  (7 steps)
```

---

## Part 4: Constraint Layer with Soft Constraints

The constraint layer validates each candidate topic against the learner's current mastery state. The key design decision: **hard veto vs soft constraint.**

A purely binary system (pass/fail at 70%) is too rigid. A student at 68% mastery of a prerequisite is essentially ready — blocking them completely is unhelpful.

**Three-tier soft constraint system:**

```python
class DAGConstraintLayer:
    def __init__(self, graph, threshold=0.70, soft_threshold=0.50):
        self.graph          = graph
        self.threshold      = threshold       # full mastery required
        self.soft_threshold = soft_threshold  # below this = hard veto

    def validate(self, topic_id, mastery_vector):
        prerequisites = list(self.graph.predecessors(topic_id))
        if not prerequisites:
            return 'approved', '✅ Foundational topic'
        
        hard_fails = []  # prerequisite < soft_threshold
        soft_fails = []  # soft_threshold <= prerequisite < threshold
        
        for prereq in prerequisites:
            m = mastery_vector.get_mastery(prereq)
            label = self.graph.nodes[prereq].get('label', prereq)
            if m < self.soft_threshold:
                hard_fails.append((label, m))
            elif m < self.threshold:
                soft_fails.append((label, m))
        
        if hard_fails:
            # Student is nowhere near ready
            return 'vetoed', f'❌ Prerequisites not met: ...'
        elif soft_fails:
            # Student is close — recommend with caution flag
            return 'challenging', f'⚠️ Challenging — prerequisites nearly met: ...'
        else:
            return 'approved', f'✅ All prerequisites mastered'
```

**Result:** instead of binary approved/vetoed, learners get three actionable signals — ready to proceed, proceed with caution, or not yet.

---

## Part 5: Multi-Objective Ranking Function

Approved topics are scored by three signals:

```python
ranking_score = (
    0.40 * mastery_gap           +  # how much still to learn
    0.35 * prerequisite_readiness+  # how prepared the student is
    0.25 * downstream_importance    # how many topics this unlocks
)
```

**Mastery gap:** prioritises topics the student is closest to mastering
```python
gap = min(max(0.0, threshold - current_mastery) / threshold, 1.0)
```

**Prerequisite readiness:** fraction of direct prerequisites already mastered
```python
readiness = mastered_prereqs / total_prereqs  # or 1.0 if no prereqs
```

**Downstream importance:** normalised count of descendant topics in the DAG
```python
scores     = {n: len(nx.descendants(graph, n)) for n in graph.nodes}
max_score  = max(scores.values())
downstream = scores[topic_id] / max_score
```

**Near-mastery boost:** topics the student has already started (10–69% mastery) receive an additional score boost proportional to their progress. This prevents completely untouched topics from outranking topics the student is actively working on.

```python
near_mastery_boost = 0.0
if 0.10 <= current_mastery < threshold:
    near_mastery_boost = 0.15 * (current_mastery / threshold)

final_score = w_gap*gap + w_ready*readiness + w_downstream*downstream + near_mastery_boost
```

### The Threshold Effect

The mastery threshold is the single most influential parameter in the system — not a sensitivity dial but a **curriculum pacing decision** with cascading effects.

Observed with the same simulated learner (seed 1000, length 150):

| Threshold | Topics Mastered | Approved | Challenging | Notes |
|-----------|----------------|----------|-------------|-------|
| 0.65 | 5 / 38 | 6 | 9 | Progress-focused, broader exploration |
| 0.75 | 1 / 38 | 2 | 17 | Mastery-focused, strict progression |

At 0.65, topics in the 65–74% mastery range are considered mastered, unlocking their dependents throughout the DAG. At 0.75, those same topics fall below threshold and their descendants are vetoed. A teacher setting 0.75 is saying "I want students truly solid before moving on." A teacher setting 0.60 is saying "progress matters more than perfection." Both are pedagogically valid — which is exactly why the threshold is configurable per deployment.

### DAG Mastery Cascade

A key challenge in sparse interaction environments is the cold-start problem: a student who scores 80% on Modular Arithmetic almost certainly understands Whole Numbers, but if they haven't directly interacted with Whole Numbers resources in the dataset, the system has no evidence of it.

The cascade inference mechanism addresses this:

```python
def cascade_mastery(mastery_vector, graph):
    changed = True
    while changed:
        changed = False
        for node in graph.nodes:
            node_mastery = mastery_vector.get_mastery(node)
            if node_mastery < 0.40:
                continue
            for prereq in graph.predecessors(node):
                prereq_mastery = mastery_vector.get_mastery(prereq)
                # Infer prerequisite mastery as 85% of descendant mastery
                inferred = min(node_mastery * 0.85, 0.95)
                if inferred > prereq_mastery:
                    mastery_vector.update(prereq, inferred)
                    changed = True
    return mastery_vector
```

The cascade runs iteratively until convergence, propagating mastery upward through the full ancestor chain. Effect: Topics Mastered jumped from 1/38 to 5/38 for the same learner state, with the system correctly inferring foundational competence from demonstrated advanced performance.

---

## Part 6: Evaluation Results

Evaluated against Collaborative Filtering (user-based, cosine similarity, top-20 neighbours) and Matrix Factorisation (TruncatedSVD, 50 latent factors) baselines on 200 test students from the OULAD test set.

### Complete Evaluation Summary

| Metric | CF | MF | Logic Engine |
|--------|----|----|-------------|
| Violation Rate (Math) | 81.2% | 81.3% | **0.0%** |
| Violation Rate (CS) | 57.8% | 55.2% | **0.0%** |
| Severe Violations/Student | 5.86 | 5.83 | **0.00** |
| Curriculum Alignment Score | 0.877 | 0.877 | 0.500 |
| Precision@10 | 0.000 | 0.000 | DAG-filtered |
| Val AUC | N/A | N/A | **0.7692** |
| Prerequisite Aware | No | No | **Yes** |
| Domain Agnostic | No | No | **Yes** |

---

### Prerequisite Violation Rate

The primary metric: what fraction of recommendations violate at least one direct prerequisite in the DAG?

**The 81% baseline rate is not a bad implementation — it is an inherent property of similarity-based systems that have no representation of curriculum order.** In a 99.7% sparse matrix, CF and MF have almost no useful signal and fall back to recommending whatever is most popular — which is advanced content by volume.

---

### Violation Severity Analysis

Not all violations are equal. Recommending a topic where a prerequisite is at 68% is very different from recommending an advanced topic to a student with 5% foundational mastery. We categorised violations by severity:

- **Minor:** prerequisite mastery 50–70% (close to threshold, almost ready)
- **Moderate:** prerequisite mastery 25–50% (meaningful gap)
- **Severe:** prerequisite mastery 0–25% (complete beginner being directed to advanced content)

```
Collaborative Filtering:
  Minor    (50–70%): 1.56 violations per student
  Moderate (25–50%): 0.51 violations per student
  Severe    (0–25%): 5.86 violations per student  ← most damaging

Matrix Factorisation:
  Minor    (50–70%): 1.67 violations per student
  Moderate (25–50%): 0.54 violations per student
  Severe    (0–25%): 5.83 violations per student  ← most damaging

Logic Engine:
  Minor:    0.00 per student
  Moderate: 0.00 per student
  Severe:   0.00 per student  ← all eliminated by DAG
```

**The severe violation number is the most important finding.** CF and MF direct complete beginners to advanced content nearly 6 times per student on average. This is the real-world harm that constraint-aware architecture prevents.

---

### Domain Generalisability

The 0% violation rate holds when the Mathematics knowledge map is replaced with the CS Fundamentals knowledge map — no model retraining, no code changes.

```
CS Domain (200 students):
  CF  Violation Rate: 57.8%
  MF  Violation Rate: 55.2%
  Logic Engine:        0.0%  ← plug-and-play confirmed in numbers
```

---

### Curriculum Alignment Score

This metric measures how closely recommendations match the learner's current curriculum level. Higher = closer to current level.

```
CF  score: 0.877
MF  score: 0.877
Logic Engine: 0.500
```

The Logic Engine scores lower here — but this requires careful interpretation. CF and MF score high because they recommend content at the same level as what the student already knows, which is essentially recommending familiar content rather than the next learning step. A system that correctly identifies the *next* valid step will be penalised by a proximity metric because that step is intentionally one level ahead. **This is correct behaviour misread as a weakness by a metric designed for entertainment recommendation.**

---

### SAKT Model Performance

| Metric | Value |
|--------|-------|
| Val AUC | 0.7692 |
| Published SAKT range on OULAD | 0.72–0.80 |
| Training epochs | 26 (early stopped) |
| Parameters | ~2.6M |

---

## Part 7: What Went Wrong (and How We Fixed It)

### Problem 1: CUDA device-side asserts with no useful error message

Symptom: `RuntimeError: CUDA error: device-side assert triggered` with misleading traceback pointing to random lines.

Cause: Embedding index out of bounds. The v2 data builder created more unique skills (5,889) than the original CONFIG value (5,736).

Fix: Always derive `num_skills` from actual data, never hardcode it.

```python
CONFIG['num_skills'] = int(df['skill_id'].max()) + 1
```

### Problem 2: Val AUC = 0.0000 for many epochs

Symptom: Train AUC climbing normally, Val AUC stuck at exactly 0.

Cause: PyTorch's `TransformerEncoder` with mismatched mask types (`src_key_padding_mask` as bool vs causal mask as float) produces NaN in eval mode, causing all val batches to be skipped by the NaN guard.

Fix: Zero out padded positions directly in the embedding tensor before entering the transformer, bypassing `src_key_padding_mask` entirely.

```python
x = x * mask.unsqueeze(-1).float()  # zero padded positions
# Then pass to transformer with NO src_key_padding_mask
```

### Problem 3: Train AUC of 0.99 after 2 epochs = data leakage

Symptom: Suspiciously high train AUC in early epochs.

Cause: The interaction encoding included the current step's correctness as both input and target. The model was predicting what it was already told.

Fix: Shift input and target by one step.

```python
# Input: interactions[0..n-2]
# Target: corrects[1..n-1]
```

### Problem 4: v2 data with 93% correct imbalance

Symptom: After building improved assessment-anchored correctness signal, imbalance went from 50/50 to 93/7.

Cause: Filtering VLE interactions to those near upcoming assessments over-sampled successful students — students who interact with resources before exams tend to pass those exams.

Fix: Reverted to original 50/50 click-based proxy. The lesson: a balanced proxy beats an imbalanced ground truth approximation.

---

## Key Takeaways

**1. Architecture separation beats end-to-end optimisation for educational AI.**
Keeping the learner behaviour model (SAKT) separate from the domain knowledge (DAG) means you can update either independently. Swap the knowledge map without retraining. Upgrade the model without redefining curriculum.

**2. Explicit constraints beat implicit learning for safety-critical recommendations.**
A neural model will never reliably learn to respect prerequisites from interaction data alone — the prerequisite signal is too sparse and too confounded. Hard constraints are not a limitation; they are the contribution.

**3. Class balance matters more than signal quality for DKT.**
A 50/50 balanced proxy correctness signal outperformed a 93/7 imbalanced near-ground-truth signal because the model could actually learn to discriminate. Balance first, quality second.

**4. Mask type mismatches in PyTorch Transformers are silent killers.**
If you are using `TransformerEncoder` with both a causal mask and `src_key_padding_mask`, test on eval mode explicitly before training. The bug manifests only in eval mode due to the interaction between dropout and mask handling.

---

## Links

- **GitHub:** https://github.com/clementina-tom/PLRS
- **HuggingFace Model:** https://huggingface.co/Clementio/PLRS
- **Live Demo:** https://xbo78uswdvbsmsnka87e4j.streamlit.app/
- **Dataset:** [OULAD on Open University](https://analyse.kmi.open.ac.uk/open_dataset)

---

*Built as a Final Year Project in ML track, 3mttxDSNigeria. Nigerian curriculum knowledge maps constructed from NERDC scheme of work documents and online sources. Combined by Gemini Deep Research Tool. Trained on Kaggle with T4 GPU.*
