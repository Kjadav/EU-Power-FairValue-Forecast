import json
import logging
from typing import Any

from trading_pipeline_utils.llm.client import create_gemini_client, generate_json_content, repair_truncated_json, GENAI_AVAILABLE
from trading_pipeline_utils.llm.prompts import DESK_INSIGHT_SYSTEM_PROMPT, DESK_INSIGHT_USER_TEMPLATE
from trading_pipeline_utils.llm.schemas import LLMInsightResult
from trading_pipeline_utils.settings import PostModelConfig
from trading_pipeline_utils.market.curve import TranslatedProduct
from trading_pipeline_utils.market.signals import Signal
from trading_pipeline_utils.market.invalidation import InvalidationResult
from trading_pipeline_utils.market.shapes import ShapeView

logger = logging.getLogger(__name__)


def build_signal_payload(
    products: dict[str, TranslatedProduct],
    signals: dict[str, Signal],
    shapes: list[ShapeView],
    invalidations: dict[str, InvalidationResult],
    backtest_summary: dict[str, Any],
    config: PostModelConfig,
    as_of_date: str,
) -> dict[str, Any]:
    """Build the structured payload that the LLM receives to create the commentary
    """
    outright_signals = []
    for code, sig in sorted(signals.items(), key=lambda kv: -abs(kv[1].signal_score)):
        prod = products.get(code)
        inv = invalidations.get(code)
        outright_signals.append({
            "product": code,
            "direction": sig.direction,
            "edge": round(sig.edge, 2),
            "edge_z": round(sig.edge_z, 2),
            "confidence": round(sig.confidence, 3),
            "signal_score": round(sig.signal_score, 3),
            "fair_value": round(prod.fair_value, 2) if prod else None,
            "market_forward": round(prod.market_forward, 2) if prod else None,
            "std": round(prod.std, 2) if prod else None,
            "p10": round(prod.p10, 2) if prod and prod.p10 else None,
            "p50": round(prod.p50, 2) if prod and prod.p50 else None,
            "p90": round(prod.p90, 2) if prod and prod.p90 else None,
            "risk_premium_proxy": round(sig.risk_premium_proxy, 2),
            "translation_method": prod.diagnostics.method_used if prod else None,
            "translation_quality": prod.diagnostics.translation_quality if prod else None,
            "invalidation_flag": inv.invalidation_flag if inv else False,
            "invalidation_reasons": inv.invalidation_reasons if inv else [],
        })

    shape_summaries = [
        {
            "shape": s.shape_name,
            "model_shape": round(s.model_shape, 2),
            "market_shape": round(s.market_shape, 2),
            "shape_edge": round(s.shape_edge, 2),
            "z_score": round(s.shape_zscore, 2),
            "direction": s.signal_direction,
            "expression": s.suggested_expression,
        }
        for s in shapes
    ]

    bt: dict[str, Any] = {}
    for key in ["overall_mae_eur_mwh", "overall_rmse_eur_mwh", "q10_q90_coverage",
                 "n_oof_hours", "n_folds", "fold_mae_mean", "fold_mae_std"]:
        if key in backtest_summary:
            v = backtest_summary[key]
            bt[key] = round(v, 2) if isinstance(v, float) else v

    return {
        "as_of_date": as_of_date,
        "market_name": config.market_name,
        "outright_signals": outright_signals,
        "shape_signals": shape_summaries,
        "backtest_summary": bt,
    }


def _deterministic_fallback(
    payload: dict[str, Any],
) -> LLMInsightResult:
    """Generate a structured commentary in the case the llm has failed"""
    outright = payload.get("outright_signals", [])
    shapes = payload.get("shape_signals", [])
    bt = payload.get("backtest_summary", {})

    top_signals = [s for s in outright if s["direction"] != "flat"]
    top_signals.sort(key=lambda s: -abs(s["signal_score"]))

    if top_signals:
        top = top_signals[0]
        exec_summary = (
            f"As of {payload['as_of_date']}, the strongest signal is "
            f"{top['direction']} {top['product']} with edge = {top['edge']:+.2f} EUR/MWh "
            f"(z = {top['edge_z']:+.2f}, confidence = {top['confidence']:.2f}). "
        )
        mae = bt.get("overall_mae_eur_mwh")
        if mae:
            exec_summary += f"Model backtest MAE = {mae:.1f} EUR/MWh."
    else:
        exec_summary = f"As of {payload['as_of_date']}, no strong outright signals."

    outright_views = []
    for s in outright:
        inv_note = ""
        if s.get("invalidation_flag"):
            inv_note = f" [INVALIDATED: {'; '.join(s.get('invalidation_reasons', []))}]"
        outright_views.append(
            f"{s['direction'].upper()} {s['product']}: edge={s['edge']:+.2f}, "
            f"z={s['edge_z']:+.2f}, conf={s['confidence']:.2f}, "
            f"FV={s.get('fair_value', '?')}, mkt={s.get('market_forward', '?')}{inv_note}"
        )

    shape_views = [s["expression"] for s in shapes if s["direction"] != "flat"]

    risks = []
    for s in outright:
        if s.get("invalidation_flag"):
            risks.extend(s.get("invalidation_reasons", []))
    if not risks:
        risks = ["No active invalidation triggers."]

    coverage = bt.get("q10_q90_coverage")
    conf_notes = f"Model q10–q90 coverage = {coverage:.1%}." if coverage else "No backtest coverage data."

    return LLMInsightResult(
        executive_summary=exec_summary,
        outright_views=outright_views,
        shape_views=shape_views,
        why_now=f"Signal payload generated on {payload['as_of_date']} for {payload['market_name']}.",
        what_the_desk_would_do=(
            f"Consider {top_signals[0]['direction']} {top_signals[0]['product']}"
            if top_signals else "No action — all signals flat."
        ),
        key_risks=risks,
        invalidation_triggers=[
            s["product"] for s in outright if s.get("invalidation_flag")
        ] or ["None active."],
        what_would_change_the_view="Large shift in fundamental forecasts or market forward repricing.",
        confidence_notes=conf_notes,
        data_quality_notes=f"Backtest: {bt.get('n_oof_hours', '?')} OOF hours, {bt.get('n_folds', '?')} folds.",
        questions_for_trader="Do you have a view on upcoming policy or weather events?",
        source="deterministic_fallback",
    )


def _get_list(d: dict[str, Any], k: str) -> list[str]:
    v = d.get(k, [])
    return [str(x) for x in v] if isinstance(v, list) else [str(v)]


def generate_llm_insights(
    payload: dict[str, Any],
    config: PostModelConfig,
) -> LLMInsightResult:
    """Call Gemini for additional analysis"""
    llm_cfg = config.llm_insight
    if not llm_cfg.enabled:
        return _deterministic_fallback(payload)

    api_key = llm_cfg.api_key
    if not api_key:
        logger.info("Gemini API key not set — using deterministic fallback")
        return _deterministic_fallback(payload)

    if not GENAI_AVAILABLE:
        logger.warning("google-genai package not installed — using deterministic fallback")
        return _deterministic_fallback(payload)

    user_message = DESK_INSIGHT_USER_TEMPLATE.format(payload=json.dumps(payload, indent=2, default=str))
    full_prompt = f"{DESK_INSIGHT_SYSTEM_PROMPT}\n\n{user_message}"

    client = create_gemini_client(api_key)
    try:
        raw_text = generate_json_content(
            client,
            llm_cfg.model,
            full_prompt,
            temperature=llm_cfg.temperature,
            max_output_tokens=llm_cfg.max_tokens,
        )
        data = repair_truncated_json(raw_text)

        return LLMInsightResult(
            executive_summary=str(data.get("executive_summary", "")),
            outright_views=_get_list(data, "outright_views"),
            shape_views=_get_list(data, "shape_views"),
            why_now=str(data.get("why_now", "")),
            what_the_desk_would_do=str(data.get("what_the_desk_would_do", "")),
            key_risks=_get_list(data, "key_risks"),
            invalidation_triggers=_get_list(data, "invalidation_triggers"),
            what_would_change_the_view=str(data.get("what_would_change_the_view", "")),
            confidence_notes=str(data.get("confidence_notes", "")),
            data_quality_notes=str(data.get("data_quality_notes", "")),
            questions_for_trader=_get_list(data, "questions_for_trader"),
            source="gemini",
        )
    except Exception as e:
        logger.warning("Gemini insight generation failed: %s — using fallback", e)
        result = _deterministic_fallback(payload)
        result.error = str(e)
        return result
