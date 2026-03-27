-- Sprint 5: Case embeddings table for similarity search
-- Populated by batch.py embedding capability

CREATE TABLE IF NOT EXISTS `i4g-ml.i4g_ml.case_embeddings` (
  case_id         STRING          NOT NULL,
  embedding       ARRAY<FLOAT64>  NOT NULL,   -- Dense embedding vector
  model_name      STRING          NOT NULL,   -- e.g. 'all-MiniLM-L6-v2'
  computed_at     TIMESTAMP       NOT NULL
)
CLUSTER BY case_id;
