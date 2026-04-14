from typing import Any

import pandas as pd

from trading_pipeline_utils.market.adapter import ForecastSnapshot
from trading_pipeline_utils.market.curve import TranslatedProduct
from trading_pipeline_utils.market.invalidation import InvalidationResult
from trading_pipeline_utils.market.signals import Signal


def build_desk_action_table(
    products: dict[str, TranslatedProduct],
    signals: dict[str, Signal],
    invalidations: dict[str, InvalidationResult],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for code in sorted(products.keys()):
        prod = products[code]
        sig = signals.get(code)
        inv = invalidations.get(code)
        if sig is None:
            continue

        rows.append({
            "product_code": code,
            "strip_type": prod.strip_type,
            "market_forward": round(prod.market_forward, 2),
            "fair_value": round(prod.fair_value, 2),
            "std": round(prod.std, 2),
            "p10": round(prod.p10, 2) if prod.p10 is not None else None,
            "p50": round(prod.p50, 2) if prod.p50 is not None else None,
            "p90": round(prod.p90, 2) if prod.p90 is not None else None,
            "edge": round(sig.edge, 2),
            "edge_z": round(sig.edge_z, 2),
            "risk_premium_proxy": round(sig.risk_premium_proxy, 2),
            "confidence": round(sig.confidence, 3),
            "suggested_direction": sig.direction,
            "suggested_expression": sig.suggested_expression,
            "suggested_position_units": round(sig.suggested_position_units, 2),
            "translation_method": prod.diagnostics.method_used,
            "translation_quality": prod.diagnostics.translation_quality,
            "invalidation_flag": inv.invalidation_flag if inv else False,
            "invalidation_summary": "; ".join(inv.invalidation_reasons) if inv else "",
            "hold_or_exit": inv.hold_or_exit if inv else "hold",
        })

    return pd.DataFrame(rows)


def build_deterministic_summaries(
    signals: dict[str, Signal],
    products: dict[str, TranslatedProduct],
    invalidations: dict[str, InvalidationResult],
) -> list[str]:
    summaries: list[str] = []
    for code, sig in sorted(signals.items(), key=lambda kv: -abs(kv[1].signal_score)):
        prod = products.get(code)
        inv = invalidations.get(code)
        if prod is None:
            continue

        if not prod.has_market_forward:
            summaries.append(_forecast_only_summary(code, prod, sig))
            continue

        if sig.direction == "flat":
            reason = "edge is small"
            if inv and inv.invalidation_flag and inv.invalidation_reasons:
                reason += f" and {inv.invalidation_reasons[0]}"
            summaries.append(f"Flat {code} because {reason}.")
            continue

        verb = "Long" if sig.direction == "long" else "Short"
        edge_desc = f"translated fair value is {abs(sig.edge_z):.1f} sigma {'above' if sig.edge > 0 else 'below'} market"

        qual_note = ""
        if prod.diagnostics.translation_quality not in ("direct", "good"):
            qual_note = f" (translation quality: {prod.diagnostics.translation_quality})"

        mkt_position = ""
        if prod.p10 is not None and prod.p90 is not None:
            if prod.market_forward < prod.p10:
                pct = 10
            elif prod.market_forward > prod.p90:
                pct = 90
            else:
                spread = prod.p90 - prod.p10
                pct = int(10 + 80 * (prod.market_forward - prod.p10) / max(spread, 1e-6))
            mkt_position = f" and market trades near model p{pct}"

        summaries.append(f"{verb} {code} because {edge_desc}{mkt_position}{qual_note}.")

    return summaries


def _forecast_only_summary(code: str, prod: TranslatedProduct, sig: Signal) -> str:
    fv = prod.fair_value
    parts = [f"Model forecasts {code} at {fv:.1f} EUR/MWh"]
    if prod.p10 is not None and prod.p90 is not None:
        parts.append(f"p10={prod.p10:.1f}, p90={prod.p90:.1f}")
    parts.append(f"std={prod.std:.1f}")
    conf = sig.confidence
    if conf > 0.01:
        parts.append(f"confidence={conf:.2f}")
    return f"{' | '.join(parts)}. No market forward provided for edge comparison."
