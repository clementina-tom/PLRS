# Logic Engine — Personalized Learning Recommender

A domain-agnostic constraint-aware recommendation framework for personalized learning using Deep Knowledge Tracing.

## Architecture
- **Layer 1:** SAKT (Self-Attentive Knowledge Tracing) — predicts learner mastery probabilities
- **Layer 2:** DAG Constraint Layer — vetoes recommendations that violate prerequisite dependencies  
- **Layer 3:** Multi-Objective Ranking Function — scores approved topics by mastery gap, readiness, and downstream importance

## Demo Domains
- Secondary School Mathematics (JSS3–SS2, Nigerian curriculum)
- CS Fundamentals (JSS3–SS2, Nigerian curriculum)

## Live Demo
[

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

](https://https://xbo78uswdvbsmsnka87e4j.streamlit.app/#learner-profile)

## Tech Stack
- PyTorch (SAKT model)
- NetworkX (DAG constraint layer)
- Streamlit (dashboard)
- HuggingFace Hub (model hosting)

## Project Presentation 
DS/ML track **3MTTxDSNIGERIA**

Domain-Agnostic Constraint-Aware Recommendation Framework  
for Personalized Learning Using Deep Knowledge Tracing
