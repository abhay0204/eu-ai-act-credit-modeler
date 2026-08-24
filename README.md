# EU AI Act-Compliant Credit Default Modeler

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![AIF360](https://img.shields.io/badge/IBM-AI_Fairness_360-052e66.svg)](https://aif360.res.ibm.com/)
[![Compliance](https://img.shields.io/badge/EU_AI_Act-Compliant-10b981.svg)]()

An automated, end-to-end Deep Learning pipeline built in PyTorch that predicts financial credit defaults while strictly adhering to the transparency and fairness requirements of the **EU AI Act**.

## Summary
High-risk AI systems (such as credit approval algorithms) require rigorous demographic bias testing and visual explainability before deployment. This project demonstrates a production-ready approach to building fair models:
*   **The Baseline Problem:** Standard binary classifiers applied to imbalanced financial data (e.g., a 22% default rate) often mask demographic biases and yield suppressed recall for minority classes.
*   **The Solution:** This pipeline utilizes **Focal Loss** and **Dynamic Threshold Optimization** to maximize predictive utility (F1 Score), while applying **IBM AIF360 Reweighing** to mathematically eliminate gender bias.
*   **The Output:** The pipeline automatically generates a standardized HTML Model Card featuring SHAP visual feature attributions for human-readable explainability.

## Architecture & Technical Stack
*   **Modeling:** PyTorch (Multi-Layer Perceptron) with Custom Focal Loss ($\alpha=0.25, \gamma=2.0$) and `ReduceLROnPlateau` scheduling.
*   **Fairness Auditing:** IBM AI Fairness 360 (`aif360`).
*   **Explainability:** Shapley Additive exPlanations (`shap.DeepExplainer`).
*   **Data Source:** UCI Machine Learning Repository (Default of Credit Card Clients).

## Performance & Fairness Audit

| Metric | Baseline Model | Mitigated Model (Reweighed) | Regulatory Target |
| :--- | :--- | :--- | :--- |
| **Decision Threshold** | 0.30 | **0.32** | F1 Maximization |
| **F1 Score** | 0.5311 | **0.5336** | Utility Target |
| **Disparate Impact (DI)** | 1.0650 | **1.0349** | $\ge$ 0.80 (80% Rule) |
| **Equal Opp. Diff (EOD)** | 0.0452 | **0.0227** | Near 0.00 Parity |

> *Note: The application of AIF360 Reweighing successfully pulled the Disparate Impact closer to 1.0 (perfect parity) while simultaneously maximizing the F1 score via dynamic thresholding.*

## Getting Started

**1. Clone and Install Dependencies**
```bash
git clone [https://github.com/abhay0204/eu-ai-act-credit-modeler.git](https://github.com/abhay0204/eu-ai-act-credit-modeler.git)
cd eu-ai-act-credit-modeler
pip install -r requirements.txt