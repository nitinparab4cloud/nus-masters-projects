"""
test_generate.py
-----------------
Smoke + regression tests for generate.py. Project 03's deliverable is a
*process* (spec -> six governance documents), so what needs testing isn't
one document's prose -- it's that the process reliably produces all six
documents, non-empty, with the risk matrix computed correctly.
"""

import pathlib
import shutil

from generate import DOCUMENTS, RISK_MATRIX, GENERATED_DIR, load_spec, render_all, write_outputs

EXAMPLE_SPEC = pathlib.Path(__file__).parent / "specs" / "example_toxic_comment_classifier.yaml"


def test_risk_matrix_is_symmetric_in_shape():
    # Every (probability, severity) pair the matrix defines maps to one of
    # the four levels -- catches a typo'd level name before it ships.
    valid_levels = {"Low", "Medium", "High", "Critical"}
    assert set(RISK_MATRIX.values()) <= valid_levels
    assert len(RISK_MATRIX) == 9  # 3x3: Low/Medium/High x Low/Medium/High


def test_load_spec_computes_expected_risk_level():
    spec = load_spec(str(EXAMPLE_SPEC))
    assert spec["risk_probability"] == "Medium"
    assert spec["risk_severity"] == "Medium"
    assert spec["overall_risk_level"] == "Medium"
    assert spec["overall_risk_level"] == RISK_MATRIX[(spec["risk_probability"], spec["risk_severity"])]


def test_render_all_produces_all_six_documents_non_empty():
    spec = load_spec(str(EXAMPLE_SPEC))
    rendered = render_all(spec)
    assert set(rendered.keys()) == set(DOCUMENTS)
    for doc_name, content in rendered.items():
        assert len(content.strip()) > 100, f"{doc_name} rendered suspiciously short"
        # A raw, unrendered Jinja tag left in the output is always a bug.
        assert "{{" not in content, f"{doc_name} has an unrendered template tag"
        assert "{%" not in content, f"{doc_name} has an unrendered template tag"


def test_write_outputs_writes_expected_files():
    spec = load_spec(str(EXAMPLE_SPEC))
    rendered = render_all(spec)
    out_dir = write_outputs(spec["system_name"] + " (ci-test)", rendered)
    try:
        assert out_dir.exists()
        for doc_name in DOCUMENTS:
            f = out_dir / doc_name
            assert f.exists(), f"{doc_name} was not written"
            assert f.stat().st_size > 0
    finally:
        # Clean up the CI-only slug so repeated runs don't accumulate junk
        # in generated/ (the checked-in worked example is a different slug
        # and is untouched by this test).
        if out_dir.exists() and out_dir.parent == GENERATED_DIR:
            shutil.rmtree(out_dir)
