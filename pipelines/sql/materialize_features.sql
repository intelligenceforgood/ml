-- Materialize feature view into features_case_features table.
-- Run as a BigQuery scheduled query (daily after ETL completes).
--
-- This performs a MERGE to update existing features and insert new ones.

MERGE `i4g-ml.i4g_ml.features_case_features` T
USING `i4g-ml.i4g_ml.v_case_features` S
ON T.case_id = S.case_id

WHEN MATCHED THEN
    UPDATE SET
        T.text_length = S.text_length,
        T.word_count = S.word_count,
        T.avg_sentence_length = S.avg_sentence_length,
        T.entity_count = S.entity_count,
        T.unique_entity_types = S.unique_entity_types,
        T.has_crypto_wallet = S.has_crypto_wallet,
        T.has_bank_account = S.has_bank_account,
        T.has_phone = S.has_phone,
        T.has_email = S.has_email,
        T.indicator_count = S.indicator_count,
        T.indicator_diversity = S.indicator_diversity,
        T.max_indicator_confidence = S.max_indicator_confidence,
        T.current_classification_axis = S.current_classification_axis,
        T.current_classification_conf = S.current_classification_conf,
        T.classification_axis_count = S.classification_axis_count,
        T.document_count = S.document_count,
        T.evidence_file_count = S.evidence_file_count,
        T.case_age_days = S.case_age_days,
        T.has_attachments = S.has_attachments,
        T._computed_at = S._computed_at,
        T._feature_version = S._feature_version

WHEN NOT MATCHED THEN
    INSERT (
        case_id,
        text_length, word_count, avg_sentence_length,
        entity_count, unique_entity_types,
        has_crypto_wallet, has_bank_account, has_phone, has_email,
        indicator_count, indicator_diversity, max_indicator_confidence,
        current_classification_axis, current_classification_conf, classification_axis_count,
        document_count, evidence_file_count,
        case_age_days, has_attachments,
        _computed_at, _feature_version
    )
    VALUES (
        S.case_id,
        S.text_length, S.word_count, S.avg_sentence_length,
        S.entity_count, S.unique_entity_types,
        S.has_crypto_wallet, S.has_bank_account, S.has_phone, S.has_email,
        S.indicator_count, S.indicator_diversity, S.max_indicator_confidence,
        S.current_classification_axis, S.current_classification_conf, S.classification_axis_count,
        S.document_count, S.evidence_file_count,
        S.case_age_days, S.has_attachments,
        S._computed_at, S._feature_version
    );
