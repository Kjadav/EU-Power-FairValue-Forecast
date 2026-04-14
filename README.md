# European Trading Pipeline : DE-LU

This repository is demonstrating a trading pipeline prototype for the electricity market : DE-LU. 

The pipeline follows a simple structure, enhanced by LLM to reduce manual effort that would be required. The workflow step follows as below:

## Quick start

```bash
# Step 1
pip install -e ".[full]"

# Step 2
Load Environment Variables

# Step 3
python -m trading_pipeline_utils --config config/pipeline.yaml
```

Default config path is `config/pipeline.yaml`.

## What it does

1. Data is fetched from [SMARD](https://www.smard.de/)
2. Data QA is conducted (including LLM)
3. Runs next-day forecast + next-week path simulation
4. Builds product fair values and confidence-weighted signals
5. Writes plots, `submission.csv`, `signal_summary.txt`, LLM JSON, and run metadata under `results/runs/<timestamp>/` (`results/latest` symlink)

## LLM Failures & Points to note:

- Ensure it is loaded into .env -> run source .env  
- Ensure key has credit though Gemini provides free LLM accessing compared to OpenAI SDK

## Configuration


| Section       | Role                     |
| ------------- | ------------------------ |
| `data_vendor` | SMARD endpoints, filters |
| `llm`         | Gemini model, limits     |
| `pipeline`    | Output dir, horizons     |


### ENTSO-E Transparency (optional)

Set `ENTSOE_TOKEN` in `.env` for [ENTSO-E Transparency](https://transparency.entsoe.eu/) API access. Fetch helpers live in `trading_pipeline_utils.entsoe_data` (e.g. `fetch_day_ahead_prices`, `fetch_actual_total_load`). The API often returns **15-minute** series; by default those functions **average to hourly** rows so the cadence matches the SMARD hourly bundle. Pass `aggregate_to_hourly=False` for native resolution, or use `aggregate_entsoe_series_to_hourly` on a raw frame.

## Outputs


| Artifact                                | Description                         |
| --------------------------------------- | ----------------------------------- |
| `da_hourly_forecast.png` / `.csv`       | Next-day fan chart                  |
| `weekly_distribution.png` / `.csv`      | Next-week average across paths      |
| `submission.csv`                        | `id`, `y_pred` for forecast horizon |
| `signal_summary.txt`                    | Rule-based signal lines             |
| `llm_payload.json`, `llm_insight.json`  | Desk commentary payload + result    |
| `config_snapshot.json`, `run_summary.`* | Run audit trail                     |


## Docker

```bash
docker compose build
docker compose run --rm app
```

## Tests

```bash
pip install -e ".[dev,full]"
pytest
```

## Documentation

`docs/` — [report.md](http://report.md) , high-level-diagram.png