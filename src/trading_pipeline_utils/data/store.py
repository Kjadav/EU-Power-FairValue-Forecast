from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

QA_REPORT_FILENAME = "qa_report.json"


#writing the qa report to json
def write_qa_report_json(output_dir: Path, report: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / QA_REPORT_FILENAME
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("QA report saved to %s", path)
    return path

#read qa report from json
def read_qa_report_json(output_dir: Path) -> dict | None:
    path = Path(output_dir) / QA_REPORT_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
