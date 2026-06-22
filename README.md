# HunterxData
This repository will gather all the analytics projects for my alter ego, The Data Hunter. I will try my best to show my path of being a great Data Professional

## Projects

### Non-Homogeneous Markov Cluster-Transition Pipeline (`markov_pipeline/`)
A production-grade, config-driven Databricks/PySpark/MLflow pipeline that models
customer value-cluster dynamics as a non-homogeneous Markov chain: a
feature-dependent 6×6 transition kernel per customer, composed as a Markov
Reward Process for expected lifetime, churn absorption and CLV, plus a
closed-loop value layer (action queue, expected actioning volume, modeled growth
expectation vs realized). See [`markov_pipeline/README.md`](markov_pipeline/README.md)
and the operating model in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).
