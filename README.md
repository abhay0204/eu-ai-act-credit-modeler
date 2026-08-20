# EU AI Act-Compliant Credit Default Modeler

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![AIF360](https://img.shields.io/badge/IBM-AI_Fairness_360-052e66.svg)](https://aif360.res.ibm.com/)

An automated, end-to-end Deep Learning pipeline built in PyTorch that predicts financial credit defaults while strictly adhering to the transparency and fairness requirements of the **EU AI Act**.

## 📌 Project Goals
High-risk AI systems (such as credit approval algorithms) require rigorous demographic bias testing and visual explainability before deployment. This project aims to demonstrate a production-ready approach to building fair models:
*   **The Baseline Problem:** Standard binary classifiers applied to imbalanced financial data (e.g., a 22% default rate) often mask demographic biases and yield suppressed recall for minority classes.
*   **The Planned Solution:** Build a pipeline that utilizes **Focal Loss** and **Dynamic Threshold Optimization** to maximize predictive utility (F1 Score), while applying **IBM AIF360 Reweighing** to mathematically eliminate gender bias.
*   **The Output:** Automatically generate a standardized HTML Model Card featuring SHAP visual feature attributions for human-readable explainability.

## ⚙️ Planned Architecture & Technical Stack
*   **Modeling:** PyTorch (Multi-Layer Perceptron) with Custom Focal Loss.
*   **Fairness Auditing:** IBM AI Fairness 360 (`aif360`).
*   **Explainability:** Shapley Additive exPlanations (`shap.DeepExplainer`).
*   **Data Source:** UCI Machine Learning Repository (Default of Credit Card Clients).

*Note: Performance and fairness metrics will be documented here once the pipeline orchestration is fully implemented and tuned.*