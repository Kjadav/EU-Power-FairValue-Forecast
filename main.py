"""Pipeline entry point."""

import sys

from pipeline import (
    fetch_data,
    llm_validate_data,
    run_base_model,
    run_forecasting,
    validate_data,
)
from utils.io import load_config, merge_env_secrets
from utils.logging import setup_logging


def main() -> int:
    setup_logging()
    config = load_config()
    config = merge_env_secrets(config)

    data = fetch_data(config)
    validation = validate_data(data, config)
    llm_result = llm_validate_data(data, config)
    model = run_base_model(data, config)
    result = run_forecasting(model, data, config)

    signal = result["signal"]
    metrics = result["metrics"]

    print("\n" + "=" * 60)
    print("PIPELINE RESULT")
    print("=" * 60)
    print(f"  Signal:     {signal.direction}")
    print(f"  Outlook:    {signal.outlook}")
    print(f"  Confidence: {signal.confidence:.3f}")
    print("-" * 60)
    total_rows = sum(len(df) for _, df in data.iter_core_tables())
    print(f"  fetch_data:       {data.region}, {total_rows} rows")
    print(f"  validate_data:    {'PASS' if validation.ok else 'FAIL'}")
    print(f"  llm_validate:     {llm_result.verdict} (conf={llm_result.confidence:.2f})")
    print(f"  base_model:       {model['method']}, horizon={model['horizon']}")
    print(f"  forecasting:      MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
