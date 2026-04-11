#!/usr/bin/env python3
"""Download SMARD DE-LU bundle to ``data/processed/smard_bundle`` and write ``qa_report.json``.

**OpenAI / LLM QA**

- Set ``OPENAI_API_KEY`` in the environment or in a **repo-root** ``.env`` file (loaded automatically;
  existing environment variables are not overwritten).
- **LLM is on** when a non-empty ``OPENAI_API_KEY`` is available after loading ``.env``, unless you
  explicitly set ``LLM_QA=0`` / ``false`` / ``no`` / ``off``.
- Set ``LLM_QA=1`` to force on even if you rely on another mechanism for the key.
- Full bundle QA is in ``qa_report.json``; LLM-only payloads are also written to ``qa_llm.json`` when
  the LLM path runs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from eu_power_forecast.ingestion import download_smard_data_de_lu


def _load_repo_dotenv() -> None:
    """Load ``.env`` from repo root without overriding existing environment variables."""
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value


def _resolve_llm_qa_flag() -> bool:
    """Use LLM when a key is present, unless LLM_QA explicitly disables it."""
    raw = os.environ.get("LLM_QA", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def main() -> None:
    """
    Fetches from SMARD (smard.de) for bidding zone **DE-LU**:

    - **day_ahead_prices** — hourly day-ahead wholesale price (**EUR/MWh**), UTC index.
    - **load_forecast** — hourly total load / net consumption forecast (**MW**).
    - **wind_forecast_mw** — hourly on- + offshore **day-ahead wind forecast** (**MW**); prognose filters in config.
    - **solar_forecast** — hourly **day-ahead solar PV forecast** (**MW**); prognose filter in config.
    - **hydro_forecast** — hourly **day-ahead „Sonstige“ generation forecast** (**MW**, hydro-rich; SMARD 715).
    - **wind_generation_actual_mw** — hourly **realized** on- + offshore wind generation (**MW**).
    - **solar_generation_actual_mw** — hourly **realized** solar PV generation (**MW**).
    - **actual_generation_total_mw** — hourly sum of realized net generation by carrier (**MW**); filters in ``configs.configs``.
    - **hydro_pumped_storage_generation_mw** — hourly pumped-storage **generation** (**MW**); filter ``FILTER_GENERATION_PUMPED_STORAGE``.
    - **physical_flows** (optional) — cross-border flows at **quarter-hour** resolution, keyed by SMARD filter id.

    After download, writes Parquet files plus **qa_report.json** (and **qa_llm.json** when LLM QA runs).
    """
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    for p in (repo_root, repo_root / "src"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    _load_repo_dotenv()

    out = repo_root / "data/processed/smard_bundle"
    llm_qa = _resolve_llm_qa_flag()

    if llm_qa:
        try:
            import openai  # noqa: F401
        except ImportError:
            print(
                "LLM QA is enabled but the 'openai' package is missing. Install with:\n"
                "  pip install 'openai>=1.40'",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
        print("LLM QA: on (OpenAI rules will be requested for each core table).\n", flush=True)
    else:
        print(
            "LLM QA: off (no OPENAI_API_KEY after loading .env, or LLM_QA=0). "
            "Deterministic QA only.\n",
            flush=True,
        )

    print(
        "Fetching SMARD DE-LU bundle:\n"
        "  • day_ahead_prices — EUR/MWh, hourly UTC\n"
        "  • load_forecast, wind/solar forecasts + actual generation — MW, hourly UTC\n"
        "  • actual_generation_total_mw, hydro_pumped_storage_generation_mw — MW, hourly UTC\n"
        "  • physical_flows — MW, quarter-hourly (best-effort; failures logged)\n",
        flush=True,
    )
    bundle = download_smard_data_de_lu(out, write_qa_report=True, llm_qa=llm_qa)
    print(f"Wrote bundle to {out.resolve()} (region={bundle.region}, resolution={bundle.resolution}).\n")

    qa_path = out / "qa_report.json"
    if qa_path.is_file():
        report = json.loads(qa_path.read_text(encoding="utf-8"))
        summary = report.get("summary", {})
        print("qa_report.json (validation summary):")
        print(json.dumps({"summary": summary, "tables": list(report.get("tables", {}).keys())}, indent=2))
        for name, block in report.get("tables", {}).items():
            prof = (block or {}).get("profile") or {}
            print(
                f"  {name}: rows={prof.get('n_rows')} cols={prof.get('n_columns')} "
                f"dup_index={prof.get('duplicate_index_count')} overall_ok={block.get('overall_ok')}"
            )

        if llm_qa:
            llm_by_table = {
                name: (block or {}).get("llm_validation")
                for name, block in report.get("tables", {}).items()
            }
            llm_path = out / "qa_llm.json"
            llm_path.write_text(
                json.dumps({"llm_by_table": llm_by_table}, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"\nWrote {llm_path.relative_to(repo_root)} (LLM payloads per table).", flush=True)
            print("\nqa_llm.json content:", flush=True)
            print(json.dumps({"llm_by_table": llm_by_table}, indent=2, default=str), flush=True)
    else:
        print(f"Warning: expected {qa_path} but file is missing.")

    try:
        from backtest.regression.pipeline import deploy_regression_backtest
    except ImportError as e:
        print(f"Regression backtest skipped (import): {e}", flush=True)
    else:
        reg_json = out / "regression_backtest.json"
        try:
            deploy_regression_backtest(
                out,
                use_synthetic_fallback=False,
                print_summary=True,
                json_path=reg_json,
            )
        except ValueError as e:
            print(f"Regression backtest skipped: {e}", flush=True)
        except ImportError as e:
            print(f"Regression backtest skipped (e.g. matplotlib): {e}", flush=True)


if __name__ == "__main__":
    main()
