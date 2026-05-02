# BigQuery Schema: i4g_ml.actor_features

This view exposes actor-level features for ML consumption, acting as the handoff contract between I4G Core and the ML team.

| Field Name | Type | Mode | Description |
|---|---|---|---|
| `actor_id` | STRING | REQUIRED | Unique identifier for the actor. |
| `co_membership_degree` | INTEGER | NULLABLE | Number of known co-actors in the graph (degree centrality). |
| `cross_campaign_domain_count` | INTEGER | NULLABLE | Number of domains registered or used across multiple distinct campaigns. |
| `leak_count` | INTEGER | NULLABLE | Total number of leak records associated with the actor's identities. |
| `blocklist_hit_count` | INTEGER | NULLABLE | Number of times the actor's infrastructure has hit active blocklists. |
| `damage_confirmed_total` | FLOAT | NULLABLE | Total confirmed financial damage (USD) linked to this actor across all campaigns. |
| `first_seen_timestamp` | TIMESTAMP | NULLABLE | Earliest known activity across all linked identities. |
| `last_seen_timestamp` | TIMESTAMP | NULLABLE | Most recent known activity across all linked identities. |
