# BigQuery Schema: i4g_ml.actor_features

This view exposes actor-level features for ML consumption.

| Field Name | Type | Mode | Description |
|---|---|---|---|
| `actor_id` | STRING | REQUIRED | Unique identifier for the actor. |
| `alias_count` | INTEGER | NULLABLE | Number of known aliases. |
| `active_campaigns_count` | INTEGER | NULLABLE | Number of active campaigns linked to the actor. |
| `primary_motivation` | STRING | NULLABLE | Categorical feature representing motivation (e.g., Financial). |
| `first_seen_timestamp` | TIMESTAMP | NULLABLE | Earliest known activity. |
| `last_seen_timestamp` | TIMESTAMP | NULLABLE | Most recent known activity. |
| `co_actor_count` | INTEGER | NULLABLE | Number of known co-actors in the graph. |
| `total_damage_usd_estimate` | FLOAT | NULLABLE | Estimated total financial damage linked to this actor. |
| `tech_stack_array` | ARRAY<STRING> | NULLABLE | Array of technologies frequently used. |
| `has_leaked_pii` | BOOLEAN | NULLABLE | Boolean flag indicating if PII is known. |
