"""
theme.py
========
Enterprise/cybersecurity-appropriate palette for Palo Alto Networks HR
analytics.

Bug-fix note: an earlier version of this file forced the KPI card
*label* text color dark but never forced the *value* text color. When a
viewer's browser is in dark mode, Streamlit auto-selects a light value
color for st.metric -- which is invisible against this app's explicitly
white card background. Every color-bearing rule below now explicitly
targets Streamlit's actual metric sub-elements
(stMetricValue/stMetricLabel/stMetricDelta), not just the outer wrapper,
so this can't silently recur.

Scope note: this still does NOT override Streamlit's built-in sidebar
navigation styling -- a prior project found that internal HTML to be the
part of Streamlit most likely to change between versions.
"""
from __future__ import annotations

import streamlit as st

COLORS = {
    "navy": "#0F2A4A",
    "navy_light": "#1E4272",
    "slate": "#334155",
    "orange": "#FA582D",
    "orange_light": "#FF8A65",
    "bg": "#F6F7F9",
    "card": "#FFFFFF",
    "border": "#E2E6EA",
    "ink": "#111827",
    "muted": "#64748B",
    "low_risk": "#16A34A",
    "medium_risk": "#F59E0B",
    "high_risk": "#DC2626",
}

RISK_COLOR_MAP = {
    "Low Risk": COLORS["low_risk"],
    "Medium Risk": COLORS["medium_risk"],
    "High Risk": COLORS["high_risk"],
}

PLOTLY_COLORWAY = [COLORS["navy"], COLORS["orange"], COLORS["slate"], COLORS["orange_light"], COLORS["navy_light"]]


def rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' + 0-1 alpha into 'rgba(r,g,b,a)'. Plotly's color
    validator has rejected 8-digit hex-with-alpha strings on some
    versions in the past; rgba() is accepted by every version."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def inject_global_css() -> None:
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 2rem;
            max-width: 1200px;
        }}

        /* --- KPI / metric cards --- */
        div[data-testid="stMetric"] {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-left: 4px solid {COLORS['navy']};
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
        }}
        /* Force every metric sub-element's color explicitly -- fixes a
           bug where only the label was forced dark and the value text
           inherited Streamlit's dark-mode default (light gray),
           becoming invisible against this card's white background. */
        div[data-testid="stMetric"] * {{
            color: {COLORS['ink']} !important;
        }}
        div[data-testid="stMetricLabel"] * {{
            color: {COLORS['muted']} !important;
            font-weight: 600 !important;
        }}
        div[data-testid="stMetricValue"] * {{
            color: {COLORS['navy']} !important;
            font-weight: 700 !important;
        }}

        /* --- Badges (hero eyebrow row) --- */
        .nc-badge {{
            display: inline-block;
            padding: 3px 12px;
            border-radius: 100px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            background: {COLORS['bg']};
            border: 1px solid {COLORS['border']};
            color: {COLORS['navy']};
            margin-right: 6px;
        }}
        .nc-badge.accent {{
            background: {rgba(COLORS['orange'], 0.12)};
            border-color: {COLORS['orange']};
            color: #9A3412;
        }}

        /* --- Risk badges (colored pill, always white text -- safe in
           both light/dark since the background itself is always a
           solid, sufficiently dark color) --- */
        .risk-badge {{
            display: inline-block;
            padding: 3px 14px;
            border-radius: 100px;
            font-size: 0.78rem;
            font-weight: 700;
            color: white !important;
        }}

        /* --- Insight callout box --- */
        .nc-insight {{
            background: {COLORS['bg']};
            border-left: 4px solid {COLORS['orange']};
            border-radius: 8px;
            padding: 0.9rem 1.2rem;
            margin-bottom: 0.6rem;
        }}
        .nc-insight, .nc-insight * {{
            color: {COLORS['ink']} !important;
        }}
        .nc-insight b {{
            color: {COLORS['navy']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(category: str) -> str:
    color = RISK_COLOR_MAP.get(category, COLORS["muted"])
    return f'<span class="risk-badge" style="background:{color};">{category}</span>'


def badge_row_html(badges: list[str], accent_last: bool = True) -> str:
    spans = []
    for i, b in enumerate(badges):
        cls = "nc-badge accent" if (accent_last and i == len(badges) - 1) else "nc-badge"
        spans.append(f'<span class="{cls}">{b}</span>')
    return "".join(spans)


def insight_html(text: str) -> str:
    return f'<div class="nc-insight">{text}</div>'


def hero(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    st.title(title)
    st.caption(subtitle)
    if badges:
        st.markdown(badge_row_html(badges), unsafe_allow_html=True)
    st.divider()
