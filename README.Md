# Cognitive Digital Twins for Consumer Behavior Prediction

## Abstract
This repository contains the source code and experimental data for generating highly accurate Cognitive Digital Twins using Large Language Models (LLMs). The research investigates the mitigation of "Center-Collapse" bias in small-parameter models through Retrieval-Augmented Generation (RAG) calibration, Asymmetric Temperature Scaling, and Multi-Agent Persona Debate.

---

## 1. Mathematical Grounding & Evaluation Metrics

To rigorously quantify the fidelity of the synthetic consumers against the real human baselines, this project utilizes the Jensen-Shannon Divergence (JSD). JSD provides a symmetric, finite measure of similarity between two probability distributions.

The divergence between the real human probability distribution $P$ and the synthetic digital twin distribution $Q$ is calculated as:

$$JSD(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M)$$

Where $M$ is the pointwise mean of the two distributions, $M = \frac{1}{2}(P + Q)$, and $D_{KL}$ is the Kullback-Leibler divergence:

$$D_{KL}(P \parallel M) = \sum_{x \in \mathcal{X}} P(x) \log\left(\frac{P(x)}{M(x)}\right)$$

A JSD score of $0.0$ indicates perfect algorithmic alignment with human consumer behavior.

---

## 2. Model Architectures Evaluated

This research cross-examined multiple architectures to observe the impact of parameter scaling on continuous data evaluation.

* **Microsoft Phi-3 (3.8B):** Utilized as the primary edge-capable engine. Showed severe Center-Collapse natively but achieved near-perfect Mid-Tier alignment post-calibration.
* **Llama-3.3-70B (Versatile):** Utilized as the frontier baseline. Natively overcame mathematical center-gravity due to parameter scaling.
* **Google Gemma-2 (9B):** Evaluated for comparative mid-weight architectural variance.

---

## 3. Experimental Methodology: What Failed vs. What Worked

### ❌ Experiment 1: Zero-Shot Uncalibrated Prompting (Failed)
* **Approach:** Passing consumer profiles directly to the LLM and asking for a spending tier prediction.
* **Result:** Complete "Instruction Collapse." Small models defaulted to the safest middle option (Mid-Tier/Silver), ignoring mathematical realities.
* **Conclusion:** LLMs cannot natively map continuous economic variables without contextual anchoring.

### ✅ Experiment 2: 20% RAG Calibration (Success)
* **Approach:** Injecting a 20% calibration split of real human data directly into the prompt as a Bayesian anchor (Lookalike Context).
* **Result:** Successfully mapped global market priors. Reduced JSD significantly across all three tiers.
* **Implementation:** See `src/retail_simulator.py`.

### ✅ Experiment 3: Asymmetric Temperature Scaling (Success)
* **Approach:** Dynamically altering the inference temperature $T$ based on the deterministic constraint of the consumer's income bracket $x$:
    $$T(x) = \begin{cases} 0.2 & \text{if } x \in \text{Low Income} \\ 0.4 & \text{if } x \in \text{Mid Income} \\ 0.65 & \text{if } x \in \text{High Income} \end{cases}$$
* **Result:** Prevented low-income profiles from hallucinating premium purchases while allowing high-income profiles natural behavioral variance.
* **Implementation:** See `src/bank_simulator.py`.

### ✅ Experiment 4: Multi-Agent Persona Debate (Success)
* **Approach:** Resolving "Hyper-Frugality" bias by forcing the LLM to generate opposing arguments (Premium Advocate vs. Frugal Advocate) before a final Judge agent makes the decision.
* **Result:** Restored VIP/Platinum tier adoption rates that a single-agent architecture consistently suppressed.
* **Implementation:** See `run_multi_agent_bank.py`.

---

## 4. Repository Structure & Execution Steps

### Core Engine (`src/`)
* `data_handler.py`: Manages the 20/80 hybrid split and data integrity.
* `retail_simulator.py`: E-commerce digital twin generation using RFM dynamics.
* `bank_simulator.py`: Credit card tier generation using Asymmetric Temperature Scaling.
* `evaluator.py`: Houses the mathematical JSD scoring logic.

### Replication Scripts
To replicate the thesis experiments, execute the following scripts in order:
1.  **`python run_retail_validation.py`**: Validates the baseline RAG architecture against the UCI dataset.
2.  **`python run_bank_validation.py`**: Validates Asymmetric Temperature Scaling.
3.  **`python run_multi_agent_validation.py`**: Executes the Tri-Agent debate architecture.
4.  **`python run_ablation_study.py`**: Runs the head-to-head comparison between Phi-3 (Local) and Llama-3.3-70B (Frontier).
5.  **`python run_comparative_suite.py`**: Generates the final graphical market share comparisons across all models.