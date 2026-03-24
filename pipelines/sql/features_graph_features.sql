-- features_graph_features — entity co-occurrence and connected component features
-- Populated by the Dataflow/Beam graph features pipeline (WRITE_TRUNCATE refresh).

CREATE TABLE
IF NOT EXISTS `i4g-ml.i4g_ml.features_graph_features`
(
  case_id                STRING    NOT NULL,
  shared_entity_count    INT64     NOT NULL,
  entity_reuse_frequency FLOAT64   NOT NULL,
  cluster_size           INT64     NOT NULL,
  _computed_at           TIMESTAMP NOT NULL
)
OPTIONS
(
  description = 'Per-case graph features from entity co-occurrence network. Refreshed weekly by Dataflow pipeline.'
);
