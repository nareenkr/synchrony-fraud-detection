"""Build a polished, self-contained hackathon presentation PDF."""

# Presentation copy is intentionally kept as complete string literals.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
BG = "#061512"
SURFACE = "#0d2722"
SURFACE_2 = "#12342c"
TEXT = "#E8F6F2"
MUTED = "#8CB0A7"
GREEN = "#35D0A7"
AMBER = "#F4B860"
RED = "#EF6B73"
BLUE = "#73BDEC"
PURPLE = "#A595F3"


def canvas(page: int, section: str = "SYNCHRONY") -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(13.333, 7.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 13.333)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.add_patch(patches.Rectangle((0, 7.36), 13.333, 0.14, color=GREEN, lw=0))
    ax.text(0.45, 0.25, section, color=MUTED, fontsize=7, weight="bold", va="center")
    ax.text(12.87, 0.25, f"{page:02d}", color=MUTED, fontsize=7, ha="right", va="center")
    return fig, ax


def heading(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.text(0.62, 6.76, title, color=TEXT, fontsize=24, weight="bold", va="top")
    if subtitle:
        ax.text(0.64, 6.27, subtitle, color=MUTED, fontsize=10.5, va="top")


def card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    accent: str = GREEN,
    title_size: float = 12,
    body_size: float = 9,
) -> None:
    wrap_width = max(20, int(width * 12))
    wrapped_body = "\n".join(textwrap.fill(part, wrap_width) for part in body.splitlines())
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.10",
            facecolor=SURFACE,
            edgecolor="#224A40",
            linewidth=0.8,
        )
    )
    ax.add_patch(patches.Rectangle((x, y + height - 0.045), width, 0.045, color=accent, lw=0))
    ax.text(
        x + 0.22, y + height - 0.25, title, color=TEXT, fontsize=title_size, weight="bold", va="top"
    )
    ax.text(
        x + 0.22,
        y + height - 0.68,
        wrapped_body,
        color=MUTED,
        fontsize=body_size,
        va="top",
        linespacing=1.45,
    )


def bullets(
    ax: plt.Axes, items: list[str], x: float, y: float, width: int = 64, size: float = 11
) -> None:
    cursor = y
    for item in items:
        wrapped = textwrap.fill(item, width)
        ax.text(x, cursor, "•", color=GREEN, fontsize=size + 2, weight="bold", va="top")
        ax.text(x + 0.28, cursor, wrapped, color=TEXT, fontsize=size, va="top", linespacing=1.45)
        cursor -= 0.47 + 0.24 * wrapped.count("\n")


def save(pdf: PdfPages, fig: plt.Figure) -> None:
    for ax in fig.axes:
        ax.set_xlim(0, 13.333)
        ax.set_ylim(0, 7.5)
    pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches=None)
    plt.close(fig)


def title_slide(pdf: PdfPages, roll: str) -> None:
    fig, ax = canvas(1, "HACKATHON PITCH")
    ax.add_patch(patches.Circle((11.15, 4.05), 2.15, color="#0D322A", alpha=0.9))
    ax.add_patch(patches.Circle((11.15, 4.05), 1.35, fill=False, edgecolor=GREEN, linewidth=2.2))
    ax.text(11.15, 4.08, "S", color=GREEN, fontsize=56, weight="bold", ha="center", va="center")
    ax.text(0.72, 6.12, "SYNCHRONY", color=GREEN, fontsize=11, weight="bold")
    ax.text(
        0.72,
        5.56,
        "Real-time hybrid fraud\ndecision support",
        color=TEXT,
        fontsize=32,
        weight="bold",
        va="top",
        linespacing=1.08,
    )
    ax.text(
        0.75, 3.55, "Digital lending · explainable risk · human oversight", color=MUTED, fontsize=13
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.72, 2.52),
            4.2,
            0.56,
            boxstyle="round,pad=.02,rounding_size=.12",
            facecolor=SURFACE_2,
            edgecolor="#2A554B",
        )
    )
    ax.text(0.95, 2.80, f"ROLL NUMBER  {roll}", color=TEXT, fontsize=11, weight="bold", va="center")
    ax.text(
        0.75,
        1.55,
        "Prototype using synthetic data. HIGH_RISK means investigate—not deny credit.",
        color=AMBER,
        fontsize=10,
    )
    save(pdf, fig)


def problem_slide(pdf: PdfPages) -> None:
    fig, ax = canvas(2, "PROBLEM & OUTCOME")
    heading(
        ax,
        "From isolated applications to connected fraud signals",
        "A real-time investigator workflow, not another opaque binary classifier",
    )
    card(
        ax,
        0.65,
        3.55,
        3.75,
        2.15,
        "01 · THE GAP",
        "Static rules miss new patterns. A supervised model alone struggles with novel behavior, bursts, and identities connected by shared devices, IPs, or bank accounts.",
        accent=RED,
    )
    card(
        ax,
        4.78,
        3.55,
        3.75,
        2.15,
        "02 · THE RISK",
        "False positives delay access to credit; false negatives create loss. The system must expose evidence, preserve privacy, and keep a human in control.",
        accent=AMBER,
    )
    card(
        ax,
        8.91,
        3.55,
        3.75,
        2.15,
        "03 · THE OUTCOME",
        "One 0–100 risk score, three operational bands, four transparent components, curated reasons, investigator feedback, and a live dashboard.",
        accent=GREEN,
    )
    ax.text(0.68, 3.08, "Design question", color=GREEN, fontsize=9, weight="bold")
    ax.text(
        0.68,
        2.27,
        "How can we detect known fraud, novel anomalies, suspicious velocity, and coordinated rings—without turning a prototype into an autonomous lending gate?",
        color=TEXT,
        fontsize=17,
        weight="bold",
        wrap=True,
    )
    save(pdf, fig)


def architecture_slide(pdf: PdfPages) -> None:
    fig, ax = canvas(3, "ARCHITECTURE")
    ax.text(0.62, 6.38, "A modular, API-first vertical slice", color=TEXT, fontsize=24, weight="bold", va="top")
    ax.text(0.64, 5.92, "The simulator and dashboard use the same production-shaped assessment path", color=MUTED, fontsize=10.5, va="top")
    boxes = [
        (0.55, "React", "monitor · investigate"),
        (2.73, "FastAPI", "validate · authorize"),
        (4.91, "Features", "point-in-time · v1"),
        (7.09, "Hybrid AI", "4 component scores"),
        (9.27, "Risk policy", "score · explain"),
        (11.45, "SQL", "persist · analyze"),
    ]
    for index, (x, label, detail) in enumerate(boxes):
        accent = [GREEN, BLUE, PURPLE, AMBER, RED, GREEN][index]
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 4.36),
                1.42,
                1.18,
                boxstyle="round,pad=.02,rounding_size=.11",
                facecolor=SURFACE,
                edgecolor=accent,
                linewidth=1.25,
            )
        )
        ax.text(x + 0.71, 5.08, label, color=TEXT, fontsize=11, weight="bold", ha="center")
        ax.text(x + 0.71, 4.68, detail, color=MUTED, fontsize=7.7, ha="center")
        if index < len(boxes) - 1:
            ax.text(x + 1.81, 4.95, "→", color=GREEN, fontsize=16, ha="center", va="center")
    card(
        ax,
        0.72,
        1.42,
        3.62,
        1.77,
        "LOCAL",
        "SQLite + locked in-memory prior state. One-command startup keeps judging and demos reliable.",
        accent=GREEN,
    )
    card(
        ax,
        4.86,
        1.42,
        3.62,
        1.77,
        "DEPLOYMENT",
        "PostgreSQL 16 + atomic Redis state in non-root Docker containers with readiness checks.",
        accent=BLUE,
    )
    card(
        ax,
        9.00,
        1.42,
        3.62,
        1.77,
        "SECURITY",
        "Reader/admin API keys, HMAC pseudonyms, request limits, restricted CORS, and environment-only secrets.",
        accent=PURPLE,
    )
    ax.text(0.72, 0.86, "Optional AI stack:", color=AMBER, fontsize=9, weight="bold")
    ax.text(
        2.40,
        0.86,
        "Temporal tabular scoring—not semantic search or generation. Deterministic models are safer and measurable here.",
        color=MUTED,
        fontsize=8.6,
    )
    save(pdf, fig)


def data_slide(pdf: PdfPages) -> None:
    fig, ax = canvas(4, "DATA & LEAKAGE")
    heading(
        ax,
        "Honest data boundaries before impressive metrics",
        "PaySim supplies labels; lending context is deterministic synthetic enrichment with field-level provenance",
    )
    bullets(
        ax,
        [
            "Canonical strict event schema; identifiers and labels never enter the 26-feature model contract.",
            "Chronological train / validation / test split before fitting preprocessing or selecting thresholds.",
            "Offline replay uses the same state semantics as online inference: snapshot prior events, score current event, then record it.",
            "A manifest records source, hash, seed, time boundaries, schema, row counts, and entity-overlap checks.",
            "The bundled 600-row development fixture proves the pipeline—it is not representative lending evidence.",
        ],
        0.72,
        5.66,
        width=70,
        size=10.5,
    )
    card(
        ax,
        8.82,
        4.28,
        3.75,
        1.48,
        "SOURCE",
        "PaySim synthetic mobile-money transactions and their fraud label.",
        accent=BLUE,
    )
    card(
        ax,
        8.82,
        2.48,
        3.75,
        1.48,
        "ENRICHMENT",
        "Loan, income, device, login, geography, and application-history features generated deterministically.",
        accent=PURPLE,
    )
    card(
        ax,
        8.82,
        0.68,
        3.75,
        1.48,
        "PARITY INVARIANT",
        "No current/future event can leak into velocity or shared-entity features.",
        accent=GREEN,
    )
    save(pdf, fig)


def model_slide(pdf: PdfPages) -> None:
    fig, ax = canvas(5, "DETECTION LAYER")
    heading(
        ax,
        "Four specialists, one configurable decision",
        "Each component emits a normalized 0–1 score and safe signal codes",
    )
    card(
        ax,
        0.7,
        3.82,
        2.72,
        1.82,
        "38% · SUPERVISED",
        "XGBoost challenger selected against a class-weighted logistic baseline using validation PR-AUC and operating constraints.",
        accent=GREEN,
    )
    card(
        ax,
        3.78,
        3.82,
        2.72,
        1.82,
        "12% · ANOMALY",
        "Isolation Forest detects unusual combinations; empirical calibration maps raw scores into a stable 0–1 range.",
        accent=PURPLE,
    )
    card(
        ax,
        6.86,
        3.82,
        2.72,
        1.82,
        "35% · BEHAVIOR",
        "Velocity, failed login, device transition, amount deviation, and loan-to-income risk signals.",
        accent=AMBER,
    )
    card(
        ax,
        9.94,
        3.82,
        2.72,
        1.82,
        "15% · GRAPH",
        "Shared device, IP, and bank-account cardinality expose coordinated identity rings without a heavy graph platform.",
        accent=BLUE,
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (1.43, 1.15),
            10.45,
            1.55,
            boxstyle="round,pad=.03,rounding_size=.14",
            facecolor="#0C211D",
            edgecolor="#31594F",
        )
    )
    ax.text(2.08, 2.14, "0", color=GREEN, fontsize=18, weight="bold", ha="center")
    ax.text(6.66, 2.14, "40", color=AMBER, fontsize=18, weight="bold", ha="center")
    ax.text(10.60, 2.14, "70", color=RED, fontsize=18, weight="bold", ha="center")
    ax.plot([2.08, 11.15], [1.84, 1.84], color="#294B43", lw=8, solid_capstyle="round")
    ax.plot([2.08, 6.66], [1.84, 1.84], color=GREEN, lw=8, solid_capstyle="round")
    ax.plot([6.66, 10.60], [1.84, 1.84], color=AMBER, lw=8)
    ax.plot([10.60, 11.15], [1.84, 1.84], color=RED, lw=8, solid_capstyle="round")
    ax.text(3.65, 1.42, "APPROVE", color=GREEN, fontsize=9, weight="bold", ha="center")
    ax.text(8.58, 1.42, "MANUAL REVIEW", color=AMBER, fontsize=9, weight="bold", ha="center")
    ax.text(10.86, 1.42, "HIGH RISK", color=RED, fontsize=9, weight="bold", ha="center")
    save(pdf, fig)


def explain_slide(pdf: PdfPages) -> None:
    fig, ax = canvas(6, "EXPLAINABILITY")
    heading(
        ax,
        "Actionable evidence without leaking fraud logic",
        "Raw model contributions are translated through a curated reason catalog",
    )
    card(
        ax,
        0.72,
        3.68,
        3.55,
        2.08,
        "MODEL CONTRIBUTION",
        "SHAP for the tree model (or coefficients for the baseline) identifies stable directional features. Raw values never leave the service boundary.",
        accent=PURPLE,
    )
    card(
        ax,
        4.89,
        3.68,
        3.55,
        2.08,
        "SIGNAL CATALOG",
        "Model and rules map to controlled codes such as HIGH_LOAN_INCOME_RATIO, APPLICATION_BURST, and SHARED_DEVICE_IDENTITIES.",
        accent=BLUE,
    )
    card(
        ax,
        9.06,
        3.68,
        3.55,
        2.08,
        "INVESTIGATOR VIEW",
        "The API returns ranked human-readable reasons, four component scores, decision, timestamp, and versioned audit metadata.",
        accent=GREEN,
    )
    ax.text(0.74, 2.68, "Example high-risk explanation", color=GREEN, fontsize=9, weight="bold")
    example = [
        "1  Multiple applicant identities share this device.",
        "2  Application activity increased sharply within a short window.",
        "3  Requested amount is high relative to reported annual income.",
    ]
    for i, line in enumerate(example):
        y = 2.12 - i * 0.47
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.74, y - 0.12),
                8.0,
                0.38,
                boxstyle="round,pad=.015,rounding_size=.06",
                facecolor=SURFACE,
                edgecolor="#23483F",
            )
        )
        ax.text(0.94, y + 0.07, line, color=TEXT, fontsize=10, va="center")
    ax.text(9.25, 2.32, "TRANSPARENCY BOUNDARY", color=AMBER, fontsize=8.5, weight="bold")
    ax.text(
        9.25,
        1.92,
        textwrap.fill(
            "No raw identifiers, sensitive proxy explanations, user-supplied text, contribution magnitudes, or internal rule thresholds.",
            39,
        ),
        color=MUTED,
        fontsize=9.2,
        va="top",
        linespacing=1.45,
    )
    save(pdf, fig)


def product_slide(pdf: PdfPages) -> None:
    fig, ax = canvas(7, "PRODUCT EXPERIENCE")
    heading(
        ax,
        "Monitor, investigate, and replay—not just call a model",
        "React + TanStack Query + Recharts; server state remains the source of truth",
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.72, 1.02),
            7.7,
            4.85,
            boxstyle="round,pad=.02,rounding_size=.10",
            facecolor="#081D19",
            edgecolor="#294C43",
        )
    )
    ax.add_patch(patches.Rectangle((0.72, 5.38), 7.7, 0.49, color="#102C26", lw=0))
    ax.text(
        1.02, 5.62, "Synchrony  /  Monitoring", color=TEXT, fontsize=9, weight="bold", va="center"
    )
    for index, (label, value, color) in enumerate(
        [
            ("TOTAL", "11", BLUE),
            ("APPROVE", "1", GREEN),
            ("REVIEW", "4", AMBER),
            ("HIGH RISK", "6", RED),
        ]
    ):
        x = 1.02 + index * 1.78
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 4.39),
                1.52,
                0.68,
                boxstyle="round,pad=.015,rounding_size=.06",
                facecolor=SURFACE,
                edgecolor=color,
            )
        )
        ax.text(x + 0.12, 4.86, label, color=MUTED, fontsize=6.3)
        ax.text(x + 0.12, 4.52, value, color=TEXT, fontsize=15, weight="bold")
    ax.text(1.03, 3.88, "LIVE APPLICATION FEED", color=GREEN, fontsize=7.3, weight="bold")
    rows = [
        ("APP-NORMAL-01", "18", "APPROVE", GREEN),
        ("APP-BURST-02", "57", "MANUAL REVIEW", AMBER),
        ("APP-RING-03", "84", "HIGH RISK", RED),
    ]
    for index, (application, score, decision, color) in enumerate(rows):
        y = 3.27 - index * 0.62
        ax.add_patch(patches.Rectangle((1.02, y), 7.05, 0.48, color="#0C241F", lw=0))
        ax.text(1.22, y + 0.24, application, color=TEXT, fontsize=7.5, va="center")
        ax.text(5.65, y + 0.24, score, color=color, fontsize=8.5, weight="bold", va="center")
        ax.text(6.35, y + 0.24, decision, color=color, fontsize=7.2, weight="bold", va="center")
    ax.text(9.08, 5.42, "DEMO PROOF", color=GREEN, fontsize=9, weight="bold")
    bullets(
        ax,
        [
            "Seeded scenarios span web, mobile, partner API, and agent channels.",
            "Every event calls the real /predict path.",
            "One-second polling shows persisted results.",
            "Investigations expose an action and verified-outcome controls.",
            "Reset and replay are deterministic and tested.",
        ],
        9.05,
        4.91,
        width=39,
        size=10.2,
    )
    save(pdf, fig)


def evidence_slide(pdf: PdfPages) -> None:
    fig, ax = canvas(8, "MODEL FINDINGS")
    heading(
        ax,
        "Promising evidence—with the caveat in the headline",
        "Chronological test partition of the deliberately small 600-row development fixture",
    )
    plot = ROOT / "reports" / "plots" / "precision_recall_curve.png"
    if plot.exists():
        image = mpimg.imread(plot)
        ax.imshow(image, extent=(0.65, 7.05, 1.05, 5.85), aspect="auto")
    card(
        ax,
        7.55,
        4.28,
        2.34,
        1.48,
        "RECALL",
        "1.00\nClassifier at frozen threshold",
        accent=GREEN,
        title_size=10,
        body_size=9.5,
    )
    card(
        ax,
        10.15,
        4.28,
        2.34,
        1.48,
        "FPR",
        "0.0909\nClassifier only",
        accent=AMBER,
        title_size=10,
        body_size=9.5,
    )
    card(
        ax,
        7.55,
        2.42,
        2.34,
        1.48,
        "HYBRID RECALL",
        "1.00\nFlagged band",
        accent=BLUE,
        title_size=10,
        body_size=9.5,
    )
    card(
        ax,
        10.15,
        2.42,
        2.34,
        1.48,
        "HYBRID FPR",
        "0.00\nDesigned fixture",
        accent=PURPLE,
        title_size=10,
        body_size=9.5,
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (7.55, 0.82),
            4.94,
            1.06,
            boxstyle="round,pad=.02,rounding_size=.08",
            facecolor="#342815",
            edgecolor=AMBER,
        )
    )
    ax.text(7.78, 1.57, "INTERPRETATION", color=AMBER, fontsize=8, weight="bold")
    ax.text(
        7.78,
        1.28,
        textwrap.fill(
            "These numbers verify code, leakage controls, and demo separability. They do not estimate real lending performance.",
            55,
        ),
        color="#E8C98E",
        fontsize=9,
        va="top",
    )
    save(pdf, fig)


def responsible_slide(pdf: PdfPages) -> None:
    fig, ax = canvas(9, "RESPONSIBLE AI")
    heading(
        ax,
        "Human oversight is part of the architecture",
        "Safety controls are product behavior—not a disclaimer added at the end",
    )
    bullets(
        ax,
        [
            "No race, religion, gender, or other protected traits are model inputs; geography is excluded from public reasons and segment claims.",
            "A broad MANUAL_REVIEW band avoids turning uncertainty into automatic rejection.",
            "Every decision includes model, risk-config, and feature-schema versions plus curated reasons.",
            "Verified outcomes measure false positives and missed fraud; enough evidence can recommend an offline retraining review, never a silent live update.",
            "Segment reports include support counts and suppress interpretation where samples are sparse.",
            "Known risks remain: domain shift, concept drift, label bias, false-positive harm, and unmeasured intersectional disparities.",
        ],
        0.72,
        5.65,
        width=75,
        size=10.5,
    )
    card(
        ax,
        8.9,
        4.37,
        3.7,
        1.38,
        "SYSTEM OUTPUT",
        "Investigation priority and evidence",
        accent=GREEN,
    )
    ax.annotate(
        "",
        xy=(10.75, 3.84),
        xytext=(10.75, 4.34),
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.6},
    )
    card(
        ax,
        8.9,
        2.42,
        3.7,
        1.38,
        "HUMAN ACTION",
        "Review context, verify identity, document outcome",
        accent=AMBER,
    )
    ax.annotate(
        "",
        xy=(10.75, 1.89),
        xytext=(10.75, 2.39),
        arrowprops={"arrowstyle": "->", "color": AMBER, "lw": 1.6},
    )
    card(
        ax,
        8.9,
        0.47,
        3.7,
        1.38,
        "GOVERNED DECISION",
        "Credit outcome, notice, and appeal outside this prototype",
        accent=RED,
    )
    save(pdf, fig)


def security_slide(pdf: PdfPages) -> None:
    fig, ax = canvas(10, "SECURITY & PRIVACY")
    heading(
        ax,
        "Minimize data, fail closed, separate privileges",
        "Prototype controls are explicit; production upgrades are documented rather than implied",
    )
    card(
        ax,
        0.72,
        4.20,
        3.55,
        1.55,
        "ACCESS",
        "Production requires distinct 32+ character reader and administrator keys. Reader is GET-only; admin may score and control demos.",
        accent=GREEN,
    )
    card(
        ax,
        4.89,
        4.20,
        3.55,
        1.55,
        "PRIVACY",
        "HMAC pseudonyms and a derived-field allowlist; no raw user, device, IP, bank, income, amount, or request body in storage/logs.",
        accent=PURPLE,
    )
    card(
        ax,
        9.06,
        4.20,
        3.55,
        1.55,
        "API HARDENING",
        "Strict schema and bounds, 64 KiB body cap, safe error responses, request IDs, constrained CORS, parameterized ORM access.",
        accent=BLUE,
    )
    card(
        ax,
        0.72,
        1.88,
        3.55,
        1.55,
        "SUPPLY CHAIN",
        "Pinned dependency ranges, artifact SHA-256/schema checks before deserialization, and no automatic dataset downloads.",
        accent=AMBER,
    )
    card(
        ax,
        4.89,
        1.88,
        3.55,
        1.55,
        "RUNTIME",
        "Non-root containers, no-new-privileges, service health checks, internal-only Redis, environment-backed secrets.",
        accent=GREEN,
    )
    card(
        ax,
        9.06,
        1.88,
        3.55,
        1.55,
        "NEXT GATE",
        "Enterprise OIDC, short-lived tokens, TLS termination, immutable audit logs, secret manager, rate limits, SIEM/telemetry.",
        accent=RED,
    )
    save(pdf, fig)


def code_slide(pdf: PdfPages) -> None:
    fig, ax = canvas(11, "CODE WALKTHROUGH")
    heading(
        ax,
        "Small contracts keep the difficult parts testable",
        "Training replay and online requests converge on the same feature and decision boundaries",
    )
    code = """event -> state.snapshot(event, now)\nfeatures = FeatureBuilder().transform(event, snapshot)\ncomponents = {\n  "supervised": classifier.score(features),\n  "anomaly": anomaly.score(features),\n  "behavioral": behavior.score(features),\n  "graph": graph.score(features),\n}\nassessment = risk_engine.combine(components)\nrepository.save(event, assessment)\nstate.record(event, assessment)"""
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.72, 1.12),
            6.15,
            4.76,
            boxstyle="round,pad=.03,rounding_size=.10",
            facecolor="#071B17",
            edgecolor="#295047",
        )
    )
    ax.text(
        1.03,
        5.55,
        code,
        color="#C6E7DF",
        family="monospace",
        fontsize=10,
        va="top",
        linespacing=1.55,
    )
    bullets(
        ax,
        [
            "Strict LoanApplicationEvent and FraudAssessment schemas define the transport contract.",
            "Prior-only snapshot ordering prevents self-counting and future leakage.",
            "Scorers are independently swappable and return normalized, typed outputs.",
            "Risk weights and thresholds live in validated YAML—not frontend logic.",
            "Repository and state Protocols support local and deployed adapters.",
            "Artifact loaders verify digest, size, schema, version, and estimator type.",
        ],
        7.45,
        5.62,
        width=51,
        size=9.7,
    )
    save(pdf, fig)


def engineering_slide(pdf: PdfPages) -> None:
    fig, ax = canvas(12, "ENGINEERING EVIDENCE")
    heading(
        ax,
        "Evidence gates, not screenshot-driven confidence",
        "The project is reproducible locally and production-shaped in Compose",
    )
    card(
        ax,
        0.72,
        4.13,
        3.55,
        1.62,
        "BACKEND",
        "125 tests including feedback + random stream\n81% backend/training coverage\nRuff clean",
        accent=GREEN,
        body_size=11,
    )
    card(
        ax,
        4.89,
        4.13,
        3.55,
        1.62,
        "FRONTEND",
        "Vitest component suite\nTypeScript production build\n0 npm audit vulnerabilities",
        accent=BLUE,
        body_size=11,
    )
    card(
        ax,
        9.06,
        4.13,
        3.55,
        1.62,
        "END TO END",
        "Real artifacts loaded\n11-event replay deterministic\nHTTP/API and bundle smoke checks",
        accent=PURPLE,
        body_size=11,
    )
    bullets(
        ax,
        [
            "README: one-command bootstrap, local run, full-data training, Compose, tests, and troubleshooting.",
            "Versioned model manifests and generated evaluation, hybrid, plot, and responsible-AI reports.",
            "Docker Compose: React, FastAPI, PostgreSQL, Redis, health checks, non-root processes.",
            "Git is initialized on main; the student must commit with their identity and push to their own GitHub repository.",
        ],
        0.75,
        2.96,
        width=110,
        size=8.9,
    )
    ax.text(
        0.75,
        0.62,
        "Verification counts should be refreshed immediately before submission; see docs/VERIFICATION.md.",
        color=AMBER,
        fontsize=9.2,
    )
    save(pdf, fig)


def roadmap_slide(pdf: PdfPages) -> None:
    fig, ax = canvas(13, "ROADMAP")
    heading(
        ax,
        "What moves this from prototype to controlled pilot",
        "Every next step is an evidence gate, not a feature wish list",
    )
    steps = [
        ("01", "REAL DATA", "Representative lending labels, consent and retention review", GREEN),
        (
            "02",
            "VALIDATE",
            "Calibration, temporal backtests, cost curves, fairness uncertainty",
            BLUE,
        ),
        ("03", "GOVERN", "OIDC, roles, appeal workflow, model approval and rollback", PURPLE),
        ("04", "OPERATE", "Drift/SLO monitoring, rate limits, audit logs, incident runbook", AMBER),
        ("05", "PILOT", "Shadow mode, investigator feedback, controlled threshold changes", RED),
    ]
    for index, (number, title, detail, color) in enumerate(steps):
        x = 0.62 + index * 2.55
        ax.add_patch(patches.Circle((x + 0.9, 4.95), 0.38, color=color))
        ax.text(
            x + 0.9, 4.95, number, color=BG, fontsize=10, weight="bold", ha="center", va="center"
        )
        if index < len(steps) - 1:
            ax.plot([x + 1.3, x + 2.48], [4.95, 4.95], color="#315149", lw=2)
        ax.text(x + 0.9, 4.22, title, color=TEXT, fontsize=10, weight="bold", ha="center")
        ax.text(
            x + 0.9,
            3.70,
            textwrap.fill(detail, 23),
            color=MUTED,
            fontsize=8.5,
            ha="center",
            va="top",
            linespacing=1.35,
        )
    ax.add_patch(
        patches.FancyBboxPatch(
            (1.65, 1.18),
            10.03,
            1.12,
            boxstyle="round,pad=.025,rounding_size=.12",
            facecolor=SURFACE,
            edgecolor=GREEN,
        )
    )
    ax.text(
        6.66, 1.80, "KEEP THE CORE INVARIANT", color=GREEN, fontsize=8.5, weight="bold", ha="center"
    )
    ax.text(
        6.66,
        1.48,
        "Point-in-time data → transparent components → configurable policy → human decision",
        color=TEXT,
        fontsize=13,
        weight="bold",
        ha="center",
    )
    save(pdf, fig)


def close_slide(pdf: PdfPages, roll: str) -> None:
    fig, ax = canvas(14, "CLOSE")
    ax.text(6.666, 5.72, "SYNCHRONY", color=GREEN, fontsize=12, weight="bold", ha="center")
    ax.text(
        6.666,
        4.68,
        "Detect connected risk.\nExplain the evidence.\nKeep humans accountable.",
        color=TEXT,
        fontsize=31,
        weight="bold",
        ha="center",
        va="top",
        linespacing=1.16,
    )
    ax.text(
        6.666,
        2.18,
        "React · FastAPI · PostgreSQL · Redis · XGBoost · Isolation Forest · SHAP",
        color=MUTED,
        fontsize=11,
        ha="center",
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (4.75, 1.23),
            3.83,
            0.54,
            boxstyle="round,pad=.02,rounding_size=.11",
            facecolor=SURFACE_2,
            edgecolor="#2A554B",
        )
    )
    ax.text(
        6.666,
        1.50,
        f"ROLL NUMBER  {roll}",
        color=TEXT,
        fontsize=10.5,
        weight="bold",
        ha="center",
        va="center",
    )
    save(pdf, fig)


def build(output: Path, roll_number: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    with PdfPages(output) as pdf:
        title_slide(pdf, roll_number)
        problem_slide(pdf)
        architecture_slide(pdf)
        data_slide(pdf)
        model_slide(pdf)
        explain_slide(pdf)
        product_slide(pdf)
        evidence_slide(pdf)
        responsible_slide(pdf)
        security_slide(pdf)
        code_slide(pdf)
        engineering_slide(pdf)
        roadmap_slide(pdf)
        close_slide(pdf, roll_number)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roll-number", default="[ADD ROLL NUMBER]")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "submission" / "preview" / "Synchrony-presentation-preview.pdf",
    )
    args = parser.parse_args()
    build(args.output.resolve(), args.roll_number)
    print(f"Created {args.output.resolve()}")


if __name__ == "__main__":
    main()
