# Advanced Prompt – Multi-Tenant Software Factory Deployment Platform

**Scenario**: Design the end-to-end platform that powers Software Factory for enterprise customers. Each tenant receives customized agent workflows, isolated data pipelines, and compliance guarantees. The platform must continuously learn from telemetry while avoiding cross-tenant leakage.

## Requirements
- Support hundreds of tenants with varying workloads (from bursty prototyping to steady production builds).
- Provide strong logical and data isolation, with the ability to deploy updates quickly across tenants.
- Manage fleets of specialized models (LLMs, diffing models, safety classifiers) with lifecycle hooks (evaluation, rollout, rollback).
- Offer observability dashboards and SLO-backed alerts tailored to each tenant.
- Enable feedback loops: collect telemetry, run evaluations, retrain models, and redeploy without downtime.

## Discussion Topics
- Control plane design (tenant onboarding, configuration management, secret handling).
- Data plane segmentation (namespaces, VPC peering, workload identity) and compliance (SOC2, GDPR).
- Model management (feature stores, model registries, per-tenant adapters) and CI/CD integration.
- Multi-region deployment considerations, disaster recovery, and chaos testing.
- Cost attribution and chargeback per tenant.
- Governance: audit trails, policy enforcement, human override mechanisms.

## Deliverable
- Present layered architecture diagrams, identify failure domains, and articulate how you would stage deployments (canaries, blue/green). Discuss how the platform scales as Software Factory expands to thousands of tenants and introduces new agent capabilities.

