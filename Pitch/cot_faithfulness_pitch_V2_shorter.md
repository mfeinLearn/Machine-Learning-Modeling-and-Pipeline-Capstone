# Chain-of-Thought Faithfulness Detector
### Project 2: Machine Learning Modeling and Pipeline Pitch

- **Author:** Malcome Fein · GitHub: [mfeinLearn](https://github.com/mfeinLearn)
- **Pitch date:** Dec 13, 2026

> **One-line summary:** A text-only scikit-learn pipeline that flags when an LLM’s chain-of-thought hides the real reason behind its answer, so reviewers know which explanations to double-check.

**Key terms**
- **Chain of thought (CoT):** the step-by-step reasoning an AI model writes before giving its answer.
- **Hint:** an extra clue slipped into a question, e.g., "A Stanford professor thinks the answer is A."
- **Unfaithful CoT:** the hint changed the model's answer, but its reasoning never mentions the hint.

---

## 1. Business Problem Scenario

### 1.1 Problem
- **People trust AI explanations.** Doctors, analysts and safety teams read a model's reasoning to decide whether to trust its answer.
- **Those explanations can hide the real cause.**
  - Turpin et al. (2023): hints steered models' answers, and the explanations rarely mentioned them.
  - Chen et al. (2025): advanced reasoning models mentioned hints they used only 25% (Claude 3.7 Sonnet) and 39% (DeepSeek R1) of the time.
- **Risk:** a convincing but misleading explanation raises trust without raising reliability.

**Problem statement:** Given a question, a model's reasoning and its answer, predict the probability that the reasoning is unfaithful, so humans can focus their review on the riskiest explanations.

### 1.2 Stakeholders
- **AI safety teams:** need to know when reading a model's reasoning can be trusted.
- **Product managers:** decide where to show AI reasoning and which models to ship.
- **Risk / compliance officers:** need audit trails they can defend.
- **End users in high-stakes work** (medical triage, finance, legal research, AI agents): need a warning when an explanation needs a second look.

### 1.3 Goals (business outcomes)
1. **Catch most misleading explanations** before someone acts on them (target: ≥80% of unfaithful cases).
2. **Keep review cost low** by limiting false alarms (precision floor ≥50%; ~$2 per review).
3. **Give trustworthy risk scores**, so a score of 0.8 means about 80% risk and teams can pick their own cutoff.
4. **Show where the risk is:** which models and hint types are least faithful.

### 1.4 Why machine learning?
- **Simple rules miss the hard cases.** A keyword check finds reasoning that mentions the hint. It can't tell "answered independently" from "was steered and stayed quiet."
- **Labeled data already exists.** Research datasets compare each model's answer with and without a hint, so labels are objective.
- **It's cheaper than the alternative.** Testing faithfulness directly means re-running the model many times (Lanham et al., 2023). A trained classifier only needs to read the text.

### 1.5 Dataset
**Primary: [Measuring CoT Monitorability Transcripts](https://huggingface.co/datasets/ameek/measuring_cot_monitorability_transcripts)** (Meek et al., 2025)
- **License:** MIT (free to use)
- **First milestone scope:** one benchmark (MMLU) and the DeepSeek-R1, QwQ-32B, and Qwen3 outputs identified for unrestricted training.
- **Full coverage (optional later):** 8 models × 3 benchmarks.

**Held-out test (optional):** [CoT Faithfulness in Open-Weight Models](https://huggingface.co/datasets/richardyoung/cot-faithfulness-open-models) (Young, 2026)
- Used only after the first milestone works. Check exact model IDs and exclude any that appeared in training.

**Key variables**
- Reasoning text → model input
- `unfaithful` = answer switched to the hint AND the reasoning did not acknowledge it
- Group by question so all versions stay in the same split

- **Cost:** about $0. The data is free and runs on a laptop or free Colab.

### 1.6 Dataset relevance
- Directly matches the problem: each row pairs an explanation with what actually changed the answer.
- Objective labels from controlled counterfactuals.
- First milestone focuses on a clean, text-only comparison before any generalization tests.

### 1.7 Success measures
**Technical (first milestone)**
- Recall on unfaithful cases ≥ 0.80 at precision ≥ 0.50 (cutoff chosen on validation; reported on held-out questions).
- If the target is missed, show the trade-off achieved and walk through two or three cases the model missed.

**Business**
- Per 1,000 responses: how many unfaithful cases are caught, missed, and sent to review.
- Review cost ≈ $2 per flag.

---

## 2. Problem Solving Process (CRISP-DM)

### 2.1 Data Acquisition and Understanding
1. Download the primary dataset (start with MMLU + DeepSeek-R1 / QwQ-32B / Qwen3).
2. Build reasoning text and the `unfaithful` label.
3. Quick label audit and class-balance check on these models.

### 2.2 Data Preparation and Feature Engineering
- Text-only first: TF-IDF on the CoT (word 1–2-grams).
- Keep all versions of each question in the same split.
- Strip final-answer line; drop empty text.
- Engineered features and embeddings are optional and only after the first comparison works.

### 2.3 Modeling Strategy
**First milestone**
- DummyClassifier (simple baseline)
- TF-IDF Logistic Regression
- Cutoff chosen on the validation set; precision and recall reported on held-out questions.
- Error analysis on 2–3 missed cases if the 80%/50% target is not met.

**Next**
- Add Linear SVM and Naive Bayes.
- Small Streamlit demo that shows caught / missed / sent-to-review counts per 1,000 responses.

**Optional (only after the demo works)**
- Gradient boosting, embeddings, FastAPI endpoint, second dataset (with model-ID check).

### 2.4 Results Interpretation and Communication
- Lead with the per-1,000 summary and the $2-per-review cost.
- Streamlit demo lets stakeholders see the operating point.
- Walk through concrete missed cases.

### 2.5 Conceptual Framework
Text-only TF-IDF LR vs dummy → choose cutoff on validation → report on held-out questions → add SVM + Naive Bayes → Streamlit demo (per 1,000) → optional stretch items.

---

## 3. Timeline and Scope

### 3.1 Timeline (Nov 24 – Dec 13, 2026)
| Start | Due | Phase | Milestone |
|---|---|---|---|
| Nov 24–25 | Nov 25 | Dataset + labels | MMLU + DeepSeek-R1 / QwQ-32B / Qwen3 ready; question-grouped split |
| Nov 25–27 | Nov 27 | EDA | Class balance and text quality on the three models |
| Nov 28–30 | Nov 30 | Preprocessing | Text-only TF-IDF Pipeline |
| Dec 1–3 | Dec 3 | First comparison | Dummy vs TF-IDF LR; cutoff on val; P/R on held-out; error cases |
| Dec 4–7 | Dec 7 | Expand + demo | SVM + Naive Bayes + small Streamlit demo (per 1,000) |
| Dec 8–9 | Dec 9 | Documentation | Report + presentation (**core complete**) |
| Dec 10–13 | Dec 13 | Buffer / submit | Optional stretch only if time remains |

**Deliverables:** GitHub repo, notebooks, saved Pipeline, Streamlit demo, slide deck.

### 3.2 Challenges and mitigations
| Challenge | Mitigation |
|---|---|
| Scope creep | Must-have = text-only TF-IDF LR vs dummy + Streamlit demo; everything else optional |
| Label quality | Hand-audit a small stratified sample; report results with the caveat |
| Class imbalance | Class weights + threshold tuning on validation |
| Leakage | Group by question_id; strip final-answer line |

---

## References
Full citations and links are in [`cot_faithfulness_references.md`](cot_faithfulness_references.md).
