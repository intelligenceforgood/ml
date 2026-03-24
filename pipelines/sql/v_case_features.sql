-- v_case_features: BigQuery SQL view joining raw tables to compute features.
-- Materialized daily into features_case_features via scheduled query.
--
-- Usage:
--   bq query --project_id=i4g-ml --use_legacy_sql=false < pipelines/sql/v_case_features.sql
--
-- Columns align 1-to-1 with features_case_features table schema.
-- NULLs for features whose source data is not yet synced (Phase 1).

CREATE OR REPLACE VIEW `i4g-ml.i4g_ml.v_case_features` AS
SELECT
    c.case_id,
    CAST(NULL AS INT64)   AS text_length,
    CAST(NULL AS INT64)   AS word_count,
    CAST(NULL AS FLOAT64) AS avg_sentence_length,
    COALESCE(ent.entity_count, 0)          AS entity_count,
    COALESCE(ent.unique_entity_types, 0)   AS unique_entity_types,
    COALESCE(ent.has_crypto_wallet, FALSE) AS has_crypto_wallet,
    COALESCE(ent.has_bank_account, FALSE)  AS has_bank_account,
    COALESCE(ent.has_phone, FALSE)         AS has_phone,
    COALESCE(ent.has_email, FALSE)         AS has_email,
    CAST(NULL AS INT64)   AS indicator_count,
    CAST(NULL AS INT64)   AS indicator_diversity,
    CAST(NULL AS FLOAT64) AS max_indicator_confidence,
    JSON_EXTRACT_SCALAR(c.classification_result, '$.classification') AS current_classification_axis,
    COALESCE(SAFE_CAST(JSON_EXTRACT_SCALAR(c.classification_result, '$.confidence') AS FLOAT64), 0.0) AS current_classification_conf,
IF(JSON_EXTRACT(c.classification_result, '$.intent') IS NOT NULL, 1, 0) +
IF(JSON_EXTRACT(c.classification_result, '$.channel') IS NOT NULL, 1, 0) +
IF(JSON_EXTRACT(c.classification_result, '$.techniques') IS NOT NULL, 1, 0) +
IF(JSON_EXTRACT(c.classification_result, '$.actions') IS NOT NULL, 1, 0) +
IF(JSON_EXTRACT(c.classification_result, '$.persona') IS NOT NULL, 1, 0) AS classification_axis_count,
  CAST
(NULL AS INT64) AS document_count,
  CAST
(NULL AS INT64) AS evidence_file_count,
  DATE_DIFF
(CURRENT_DATE
(), DATE
(c.created_at), DAY) AS case_age_days,
  CAST
(NULL AS BOOL) AS has_attachments,
  CURRENT_TIMESTAMP
() AS _computed_at,
  1 AS _feature_version
FROM `i4g-ml.i4g_ml.raw_cases` c
LEFT JOIN
(
  SELECT case_id, COUNT(*) AS entity_count, COUNT(DISTINCT entity_type) AS unique_entity_types, LOGICAL_OR(entity_type = 'crypto_wallet') AS has_crypto_wallet, LOGICAL_OR(entity_type = 'bank_account') AS has_bank_account, LOGICAL_OR(entity_type = 'phone') AS has_phone, LOGICAL_OR(entity_type = 'email') AS has_email
FROM `i4g-ml.i4g_ml.raw_entities`
  GROUP BY case_id
) ent ON c.case_id = ent.case_id
