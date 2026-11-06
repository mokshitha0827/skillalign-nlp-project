"""Presentation helpers for the Streamlit dashboard: CSS, HTML components and Plotly charts."""
from __future__ import annotations

from html import escape

import plotly.graph_objects as go

from .gap_analyzer import MATCHED, MISSING, PARTIAL, SkillResult, category_breakdown

NAVY, BLUE, SLATE, LIGHT, BORDER = "#0B2545", "#1D4ED8", "#475569", "#F1F5F9", "#E2E8F0"
STATUS_COLOR = {MATCHED: "#1D4ED8", PARTIAL: "#D97706", MISSING: "#DC2626"}
STATUS_BG = {MATCHED: "#DBEAFE", PARTIAL: "#FEF3C7", MISSING: "#FEE2E2"}
STATUS_TEXT = {MATCHED: "#1E3A8A", PARTIAL: "#92400E", MISSING: "#991B1B"}

CSS = f"""
<style>
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }}
h1, h2, h3 {{ color: {NAVY}; letter-spacing: -0.01em; }}
section[data-testid="stSidebar"] {{ background: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: #E2E8F0; }}
section[data-testid="stSidebar"] .stRadio label p {{ font-size: 0.95rem; }}
section[data-testid="stSidebar"] [data-baseweb="select"] *, section[data-testid="stSidebar"] input {{ color: #0F172A !important; }}
section[data-testid="stSidebar"] .stButton button {{ background: transparent; border: 1px solid #64748B; color: #E2E8F0; }}
.sa-brand {{ font-size: 1.35rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0; }}
.sa-brand-sub {{ font-size: 0.8rem; color: #94A3B8; margin-bottom: 1rem; }}
.sa-hero {{ background: {NAVY}; color: #FFFFFF; border-radius: 16px; padding: 2.2rem 2.4rem; margin-bottom: 1.2rem; }}
.sa-hero h1 {{ color: #FFFFFF; margin: 0 0 .4rem 0; font-size: 2.1rem; }}
.sa-hero p {{ color: #CBD5E1; font-size: 1.05rem; margin: 0; max-width: 56rem; }}
.sa-card {{ background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 14px; padding: 1.1rem 1.25rem; height: 100%; }}
.sa-card h4 {{ margin: 0 0 .35rem 0; color: {NAVY}; font-size: 1.02rem; }}
.sa-card p, .sa-card li {{ color: {SLATE}; font-size: .93rem; line-height: 1.5; margin: 0; }}
.sa-step {{ display:inline-block; background:{BLUE}; color:#fff; border-radius:999px; width:1.7rem; height:1.7rem; line-height:1.7rem; text-align:center; font-weight:700; margin-bottom:.5rem; }}
.sa-kpi {{ background:#FFFFFF; border:1px solid {BORDER}; border-radius:14px; padding:1rem 1.2rem; border-top: 4px solid var(--c); }}
.sa-kpi .v {{ font-size:2rem; font-weight:700; color:{NAVY}; line-height:1.1; }}
.sa-kpi .l {{ font-size:.82rem; color:{SLATE}; text-transform:uppercase; letter-spacing:.05em; }}
.sa-kpi .s {{ font-size:.8rem; color:{SLATE}; margin-top:.15rem; }}
.sa-pill {{ display:inline-block; padding:.12rem .6rem; border-radius:999px; font-size:.78rem; font-weight:600; }}
.sa-note {{ background:{LIGHT}; border:1px solid {BORDER}; border-radius:12px; padding:.8rem 1rem; color:{SLATE}; font-size:.9rem; }}
.sa-evidence {{ border-left: 3px solid {BLUE}; background:{LIGHT}; padding:.5rem .8rem; border-radius:0 8px 8px 0; margin:.35rem 0; color:#0F172A; font-size:.92rem; }}
.sa-muted {{ color:{SLATE}; font-size:.85rem; }}
div[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:12px; }}
.stButton button[kind="primary"] {{ background:{BLUE}; border:none; border-radius:10px; font-weight:600; }}
.stButton button, .stDownloadButton button {{ border-radius:10px; }}
</style>
"""


def kpi(label: str, value, sub: str = "", color: str = BLUE) -> str:
    return (f'<div class="sa-kpi" style="--c:{color}"><div class="l">{escape(label)}</div>'
            f'<div class="v">{escape(str(value))}</div><div class="s">{escape(sub)}</div></div>')


def pill(status: str) -> str:
    return (f'<span class="sa-pill" style="background:{STATUS_BG.get(status, LIGHT)};color:{STATUS_TEXT.get(status, NAVY)}">'
            f'{escape(status)}</span>')


def card(title: str, body_html: str) -> str:
    return f'<div class="sa-card"><h4>{escape(title)}</h4>{body_html}</div>'


def _layout(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#0F172A", size=13),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def status_donut(stats: dict) -> go.Figure:
    labels = [MATCHED, PARTIAL, MISSING]
    values = [stats["matched"], stats["partial"], stats["missing"]]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, sort=False, textinfo="value+percent",
                           marker=dict(colors=[STATUS_COLOR[s] for s in labels], line=dict(color="#fff", width=2))))
    fig.update_layout(title="Requirement status")
    return _layout(fig)


def coverage_gauge(coverage: float | None) -> go.Figure:
    if coverage is None:
        fig = go.Figure()
        fig.add_annotation(text="Coverage not available<br>(no requirements identified)", showarrow=False, font=dict(size=15))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        return _layout(fig)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=coverage, number=dict(suffix="%", font=dict(color=NAVY)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=BLUE), bgcolor=LIGHT,
                   steps=[dict(range=[0, 40], color="#E2E8F0"), dict(range=[40, 70], color="#CBD5E1"), dict(range=[70, 100], color="#BFDBFE")])))
    fig.update_layout(title="Requirement coverage (illustrative)")
    return _layout(fig)


def category_chart(results: list[SkillResult]) -> go.Figure:
    df = category_breakdown(results)
    fig = go.Figure()
    for status in (MATCHED, PARTIAL, MISSING):
        fig.add_bar(y=df["Category"], x=df[status], name=status, orientation="h", marker_color=STATUS_COLOR[status])
    fig.update_layout(barmode="stack", title="Skills by category and status", yaxis=dict(autorange="reversed"),
                      xaxis=dict(title="Number of skills", dtick=1))
    return _layout(fig, height=max(280, 60 * max(len(df), 1) + 90))


def roadmap_chart(items) -> go.Figure:
    names = {"High": "Learn first", "Medium": "Learn next", "Low": "Learn later"}
    colors = {"High": "#DC2626", "Medium": "#D97706", "Low": "#1D4ED8"}
    start, rows = 0, []
    for it in items:
        rows.append((it.skill, start, it.estimated_weeks, it.priority))
        start += it.estimated_weeks
    fig = go.Figure()
    shown = set()
    for skill, s0, w, p in rows:
        fig.add_bar(y=[skill], x=[w], base=[s0], orientation="h", marker_color=colors[p], name=names[p], legendgroup=p,
                    showlegend=p not in shown, text=[f"{w} wk"], textposition="inside", insidetextanchor="middle",
                    hovertemplate=f"{skill}: weeks {s0 + 1} to {s0 + w}<extra></extra>")
        shown.add(p)
    fig.update_layout(title="Suggested study order", yaxis=dict(autorange="reversed"), xaxis=dict(title="Week (approximate)"))
    return _layout(fig, height=max(260, 34 * len(rows) + 120))


# ------------------------------------------------------------------ skill cloud (no extra library)
CLOUD_SIZE = {"required": 40, "unspecified": 30, "preferred": 22}


def cloud_layout(results: list[SkillResult], width: int = 1000, height: int = 420) -> list[dict]:
    """Place skill names on a spiral so that no two words overlap. Size = importance in the job; sorted largest first."""
    import math
    words = sorted(results, key=lambda r: (-CLOUD_SIZE[r.priority], r.skill.lower()))
    placed: list[dict] = []
    cx, cy, pad = width / 2, height / 2, 26
    for r in words:
        size = float(CLOUD_SIZE[r.priority])
        for _shrink in range(8):
            w, h = len(r.skill) * size * 0.60, size * 1.15
            theta, spot = 0.0, None
            while theta < 90:
                rad = 3.2 * theta
                x, y = cx + rad * math.cos(theta) * 2.0, cy + rad * math.sin(theta) * 0.9
                theta += 0.18
                if x - w / 2 < 8 or x + w / 2 > width - 8 or y - h / 2 < 8 or y + h / 2 > height - 8:
                    continue
                if all(abs(x - q["x"]) * 2 >= (w + q["w"]) + pad or abs(y - q["y"]) * 2 >= (h + q["h"]) + pad for q in placed):
                    spot = (x, y); break
            if spot:
                placed.append({"skill": r.skill, "status": r.status, "priority": r.priority, "size": size, "x": spot[0], "y": spot[1], "w": w, "h": h})
                break
            size *= 0.88
    return placed


def skill_cloud_html(results: list[SkillResult], width: int = 1000, height: int = 420) -> str:
    """Responsive HTML word cloud: bigger word = more important in the job, colour = result (Found / Partly found / Not found)."""
    items = cloud_layout(results, width, height)
    if not items:
        return ""
    names = {MATCHED: "Found in resume", PARTIAL: "Partly found", MISSING: "Not found"}
    imp = {"required": "Must have", "preferred": "Nice to have", "unspecified": "Mentioned"}
    spans = "".join(
        f'<span title="{escape(i["skill"])}: {names[i["status"]]} ({imp[i["priority"]]})" style="position:absolute;left:{100 * i["x"] / width:.2f}%;top:{100 * i["y"] / height:.2f}%;'
        f'transform:translate(-50%,-50%);white-space:nowrap;font-weight:700;color:{STATUS_COLOR[i["status"]]};font-size:{100 * i["size"] / width:.3f}cqw;line-height:1">{escape(i["skill"])}</span>'
        for i in items)
    legend = "".join(f'<span style="margin-right:1.2rem"><span style="display:inline-block;width:.7rem;height:.7rem;border-radius:50%;background:{STATUS_COLOR[s]};margin-right:.35rem"></span>{names[s]}</span>' for s in (MATCHED, PARTIAL, MISSING))
    return (f'<div style="border:1px solid {BORDER};border-radius:14px;background:#fff;padding:.5rem">'
            f'<div style="position:relative;width:100%;aspect-ratio:{width}/{height};container-type:inline-size;overflow:hidden">{spans}</div>'
            f'<div style="padding:.4rem .8rem;color:{SLATE};font-size:.85rem">{legend}&nbsp;·&nbsp;Bigger word = more important in the job</div></div>')
