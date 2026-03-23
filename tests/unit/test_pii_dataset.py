"""Unit tests for PII redaction in dataset export — Sprint 2 task 2.2."""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import pandas as pd

from ml.data.pii import redact_pii, redact_record

# Common PII patterns used for validation scan (2.2.4)
_PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),  # Credit card
]


class TestRedactPIIInExport:
    """2.2.1 / 2.2.3 — Verify PII redaction during dataset export."""

    def test_redact_record_replaces_email(self):
        record = {"text": "Contact john@example.com", "case_id": "c1"}
        result = redact_record(record)
        assert "john@example.com" not in result["text"]
        assert "[EMAIL]" in result["text"]
        # Non-text fields are untouched
        assert result["case_id"] == "c1"

    def test_redact_record_replaces_phone(self):
        record = {"text": "Call 555-123-4567", "narrative": "Phone: 555-123-4567"}
        result = redact_record(record)
        assert "555-123-4567" not in result["text"]
        assert "555-123-4567" not in result["narrative"]

    def test_redact_record_replaces_ssn(self):
        record = {"text": "SSN is 123-45-6789"}
        result = redact_record(record)
        assert "123-45-6789" not in result["text"]
        assert "[SSN]" in result["text"]

    def test_redact_preserves_non_text_fields(self):
        record = {"text": "some@email.com", "features": {"a": 1}, "labels": "FRAUD"}
        result = redact_record(record)
        assert result["features"] == {"a": 1}
        assert result["labels"] == "FRAUD"

    def test_export_jsonl_applies_redaction(self):
        """Integration: _export_jsonl with redact=True strips PII."""
        from ml.data.datasets import _export_jsonl

        df = pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "text": ["Email me at bob@evil.com", "SSN: 999-88-7777"],
            }
        )

        # Mock GCS
        mock_blob = MagicMock()
        uploaded_content = None

        def _capture(content, **kwargs):
            nonlocal uploaded_content
            uploaded_content = content

        mock_blob.upload_from_string.side_effect = _capture

        with (patch("ml.data.datasets.storage.Client") as mock_storage,):
            mock_storage.return_value.bucket.return_value.blob.return_value = mock_blob
            _export_jsonl(df, "bucket", "path.jsonl", redact=True)

        # Parse exported JSONL and verify no PII
        assert uploaded_content is not None
        for line in uploaded_content.strip().split("\n"):
            record = json.loads(line)
            assert "bob@evil.com" not in record.get("text", "")
            assert "999-88-7777" not in record.get("text", "")

    def test_export_jsonl_no_redaction_preserves_pii(self):
        """Without redact=True, PII is preserved (for debugging/validation)."""
        from ml.data.datasets import _export_jsonl

        df = pd.DataFrame({"text": ["bob@evil.com"]})

        mock_blob = MagicMock()
        uploaded = None

        def _cap(content, **kwargs):
            nonlocal uploaded
            uploaded = content

        mock_blob.upload_from_string.side_effect = _cap

        with patch("ml.data.datasets.storage.Client") as mock_storage:
            mock_storage.return_value.bucket.return_value.blob.return_value = mock_blob
            _export_jsonl(df, "bucket", "path.jsonl", redact=False)

        assert "bob@evil.com" in uploaded


class TestPIIValidationScan:
    """2.2.4 — Validate exported JSONL contains no raw PII patterns."""

    def test_redacted_text_passes_pii_scan(self):
        """A redacted string should not match any PII regex."""
        raw = "Email bob@evil.com or call 555-123-4567 SSN 123-45-6789 CC 4111-1111-1111-1111"
        clean = redact_pii(raw)

        for pat in _PII_PATTERNS:
            assert not pat.search(clean), f"Pattern {pat.pattern} matched in redacted text: {clean}"

    def test_scan_multiple_records(self):
        """Batch scan: multiple records should all be clean after redaction."""
        records = [
            {"text": "victim@email.com sent money", "narrative": "Call 800-555-0199"},
            {"text": "SSN: 078-05-1120", "narrative": "CC 4111 1111 1111 1111"},
        ]
        for rec in records:
            clean = redact_record(rec)
            for field in ("text", "narrative"):
                for pat in _PII_PATTERNS:
                    assert not pat.search(
                        clean.get(field, "")
                    ), f"PII pattern {pat.pattern} found in {field}: {clean[field]}"


class TestDatasetRegistryRedactedField:
    """2.2.2 — Verify redacted field appears in registry metadata."""

    @patch("ml.data.datasets.bigquery.Client")
    @patch("ml.data.datasets.storage.Client")
    def test_metadata_includes_redacted_true(self, mock_storage, mock_bq):
        from ml.data.datasets import create_dataset_version

        # Mock BQ to return a small DataFrame
        bq_client = mock_bq.return_value
        bq_client.query.return_value.to_dataframe.return_value = pd.DataFrame(
            {
                "case_id": [f"c{i}" for i in range(100)],
                "text": [f"text {i}" for i in range(100)],
                "labels": ["INTENT.ROMANCE"] * 100,
            }
        )
        # Mock version query
        mock_row = MagicMock()
        mock_row.next_v = 1
        bq_client.query.return_value.result.return_value = [mock_row]
        bq_client.insert_rows_json.return_value = []

        # Mock GCS
        mock_blob = MagicMock()
        mock_storage.return_value.bucket.return_value.blob.return_value = mock_blob

        metadata = create_dataset_version(
            capability="classification",
            version=1,
            redact=True,
            min_samples_per_class=1,
        )

        assert metadata["redacted"] is True

    @patch("ml.data.datasets.bigquery.Client")
    @patch("ml.data.datasets.storage.Client")
    def test_metadata_includes_redacted_false(self, mock_storage, mock_bq):
        from ml.data.datasets import create_dataset_version

        bq_client = mock_bq.return_value
        bq_client.query.return_value.to_dataframe.return_value = pd.DataFrame(
            {
                "case_id": [f"c{i}" for i in range(100)],
                "text": [f"text {i}" for i in range(100)],
                "labels": ["INTENT.ROMANCE"] * 100,
            }
        )
        mock_row = MagicMock()
        mock_row.next_v = 2
        bq_client.query.return_value.result.return_value = [mock_row]
        bq_client.insert_rows_json.return_value = []

        mock_blob = MagicMock()
        mock_storage.return_value.bucket.return_value.blob.return_value = mock_blob

        metadata = create_dataset_version(
            capability="classification",
            version=2,
            redact=False,
            min_samples_per_class=1,
        )

        assert metadata["redacted"] is False
