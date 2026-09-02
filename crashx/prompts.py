"""Shared prompt / structured-response helpers for CrashX."""

from __future__ import annotations

USER_PROMPT = (
    "Analyze this car crash video spatiotemporally. Provide crash severity, "
    "impact point, exact timestamp window, and detailed causal explanation."
)


def format_target_text(
    severity: str,
    impact: str,
    start_sec: float,
    end_sec: float,
    vehicles: str,
    weather: str,
    explanation: str,
    n_vehicles: str | int | None = None,
) -> str:
    """Serialize ground-truth fields into the structured assistant response."""
    parts = [
        f"Severity: {severity}",
        f"Impact: {impact}",
        f"Start: {start_sec:.3f}s",
        f"End: {end_sec:.3f}s",
        f"Vehicles: {vehicles}",
    ]
    if n_vehicles is not None and str(n_vehicles).strip() not in ("", "nan", "None"):
        parts.append(f"NumVehicles: {n_vehicles}")
    parts.append(f"Weather: {weather}")
    parts.append(f"Explanation: {explanation}")
    return " | ".join(parts)
