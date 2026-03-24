# ML Platform Documentation

Developer documentation for the I4G ML Platform.

## Design

- [Architecture](design/architecture.md) — system overview, data flow, Mermaid diagram
- [Monitoring](design/monitoring.md) — BigQuery queries, alerting thresholds, dashboard setup
- [Technical Design Document](design/ml_infrastructure_tdd.md) — detailed design specification

## Runbooks

- [Deployment](runbooks/deployment.md) — step-by-step deployment guide
- [Retraining](runbooks/retraining.md) — continuous retraining trigger and E2E loop

## Developer Bootcamp

Guided exercises for onboarding new contributors to the ML platform. Target audience: developer with Python + GCP basics but no ML platform experience.

1. [Data Flow Walkthrough](bootcamp/01-data-flow-walkthrough.md) — trace a case through the data pipeline
2. [Train a Model Locally](bootcamp/02-train-locally.md) — run XGBoost training in Docker
3. [Submit a Pipeline](bootcamp/03-submit-pipeline.md) — compile and submit to Vertex AI
4. [Evaluate and Promote](bootcamp/04-evaluate-and-promote.md) — interpret metrics, run eval gate
5. [Deploy to Serving](bootcamp/05-deploy-to-serving.md) — deploy, predict, verify logging
6. [Monitor and Retrain](bootcamp/06-monitor-and-retrain.md) — drift, accuracy, retraining triggers
7. [Add a New Capability](bootcamp/07-add-new-capability.md) — end-to-end new ML capability
8. [Graph Features Pipeline](bootcamp/08-graph-features.md) — Dataflow/Beam graph features
9. [Looker Studio Dashboard](bootcamp/09-looker-dashboard.md) — build monitoring dashboard
