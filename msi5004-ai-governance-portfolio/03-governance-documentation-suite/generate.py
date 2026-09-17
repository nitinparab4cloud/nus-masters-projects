"""
generate.py
-----------
Renders the full six-document governance artifact set from one YAML spec
file. This is the point of Case 03: the deliverable isn't six hand-written
documents, it's a *process* that produces those six documents for any model
you point it at.

Usage:
    python generate.py specs/example_toxic_comment_classifier.yaml

Writes to generated/<system_name>/model_card.md, risk_assessment.md,
data_governance.md, human_oversight.md, incident_response.md, and
vendor_risk_questionnaire.md.
"""

import argparse
import datetime
import pathlib
import re

import yaml
from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"
GENERATED_DIR = pathlib.Path(__file__).parent / "generated"

DOCUMENTS = [
    "model_card.md",
    "risk_assessment.md",
    "data_governance.md",
    "human_oversight.md",
    "incident_response.md",
    "vendor_risk_questionnaire.md",
]

RISK_MATRIX = {
    # (probability, severity) -> overall level. Deliberately simple and
    # legible -- a governance reviewer should be able to sanity-check this
    # table in ten seconds, which is the whole point of using a matrix
    # instead of a black-box score.
    ("Low", "Low"): "Low",
    ("Low", "Medium"): "Low",
    ("Low", "High"): "Medium",
    ("Medium", "Low"): "Low",
    ("Medium", "Medium"): "Medium",
    ("Medium", "High"): "High",
    ("High", "Low"): "Medium",
    ("High", "Medium"): "High",
    ("High", "High"): "Critical",
}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_spec(spec_path: str) -> dict:
    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    spec.setdefault("generated_date", datetime.date.today().isoformat())
    spec["overall_risk_level"] = RISK_MATRIX.get(
        (spec["risk_probability"], spec["risk_severity"]), "Unrated"
    )
    return spec


def render_all(spec: dict) -> dict:
    # Note: trim_blocks is deliberately off. It's tempting to turn on for
    # tighter output, but it silently eats the newline after *any* block
    # tag -- including one at the end of a text line -- which merges lines
    # that should stay separate (bit us during testing on the "regulatory
    # context" block in risk_assessment.md.j2). A few extra blank lines in
    # generated Markdown are harmless; a merged sentence isn't.
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), lstrip_blocks=True)
    rendered = {}
    for doc_name in DOCUMENTS:
        template = env.get_template(f"{doc_name}.j2")
        rendered[doc_name] = template.render(**spec)
    return rendered


def write_outputs(system_name: str, rendered: dict) -> pathlib.Path:
    out_dir = GENERATED_DIR / slugify(system_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_name, content in rendered.items():
        (out_dir / doc_name).write_text(content)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec_path", help="Path to a YAML spec file, e.g. specs/example_toxic_comment_classifier.yaml")
    args = parser.parse_args()

    spec = load_spec(args.spec_path)
    rendered = render_all(spec)
    out_dir = write_outputs(spec["system_name"], rendered)

    print(f"Overall risk level computed as: {spec['overall_risk_level']}")
    print(f"Wrote {len(rendered)} documents to {out_dir}/")


if __name__ == "__main__":
    main()
