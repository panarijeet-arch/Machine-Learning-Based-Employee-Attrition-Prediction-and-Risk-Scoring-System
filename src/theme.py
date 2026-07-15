"""
theme.py
========
Enterprise/cybersecurity-appropriate palette for Palo Alto Networks HR
analytics. Deliberately conservative in scope compared to a prior
project's theming: the base light/dark theme and background/text colors
are set via .streamlit/config.toml (the officially supported mechanism,
stable across Streamlit versions), and this module adds only a small,
tested amount of extra CSS for KPI cards and risk badges. It does NOT
override Streamlit's built-in sidebar navigation styling -- a prior
project found that this is the part of Streamlit's internal HTML most
likely to change between versions, so it is left at Streamlit's own
default rather than risking it going invisible again.
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
        }}
        div[data-testid="stMetric"] {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
        }}
        div[data-testid="stMetric"] label {{
            color: {COLORS['muted']} !important;
        }}
        .risk-badge {{
            display: inline-block;
            padding: 2px 12px;
            border-radius: 100px;
            font-size: 0.78rem;
            font-weight: 700;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(category: str) -> str:
    color = RISK_COLOR_MAP.get(category, COLORS["muted"])
    return f'<span class="risk-badge" style="background:{color};">{category}</span>'


def hero(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
    st.divider()
