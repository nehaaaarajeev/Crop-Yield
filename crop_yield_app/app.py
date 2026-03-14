# ============================================================
# CROP YIELD DASHBOARD — app.py
# Streamlit + Plotly interactive dashboard
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix
)

# ─────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 Crop Yield Intelligence Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
# GLOBAL STYLES  (light / white theme)
# ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Root background ── */
    .stApp { background-color: #F8F9FA; }
    section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }

    /* ── KPI card ── */
    .kpi-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px 18px 16px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid var(--accent);
        text-align: center;
    }
    .kpi-label { font-size: 13px; color: #6B7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 32px; font-weight: 800; color: #111827; margin: 6px 0 4px; }
    .kpi-sub   { font-size: 12px; color: #9CA3AF; }

    /* ── Section header ── */
    .section-header {
        font-size: 19px; font-weight: 700; color: #1F2937;
        border-left: 4px solid #16A34A; padding-left: 10px;
        margin-bottom: 6px;
    }

    /* ── Tab font ── */
    button[data-baseweb="tab"] { font-size: 14px !important; font-weight: 600 !important; }

    /* ── Expander ── */
    details summary { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# COLOUR PALETTES
# ─────────────────────────────────────────────────
SEASON_COLORS   = {"Kharif": "#16A34A", "Rabi": "#2563EB", "Zaid": "#F97316"}
SOIL_COLORS     = {"Loamy": "#92400E", "Clay": "#7F1D1D", "Sandy": "#FBBF24", "Silty": "#6B7280", "Black": "#111827"}
SUCCESS_COLORS  = {1: "#16A34A", 0: "#DC2626"}
SUCCESS_LABELS  = {1: "Success", 0: "Failure"}
PLOTLY_TEMPLATE = "plotly_white"

# ─────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING  (cached)
# ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Crop_Yield.csv", encoding="latin1")
    return df

@st.cache_data
def preprocess_and_train(df_raw):
    df = df_raw.copy()

    # --- Null handling ---
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # --- Encode ---
    df_enc = df.copy()
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col])

    # --- Split ---
    X = df_enc.drop(columns=["Yield Success"])
    y = df_enc["Yield Success"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # --- Train ---
    dt  = DecisionTreeClassifier(random_state=42)
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    gbt = GradientBoostingClassifier(n_estimators=100, random_state=42)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gbt.fit(X_train, y_train)

    trained = {"Decision Tree": dt, "Random Forest": rf, "Gradient Boosted Trees": gbt}

    # --- Eval ---
    eval_rows = []
    for name, mdl in trained.items():
        y_pred = mdl.predict(X_test)
        eval_rows.append({
            "Model": name,
            "Train Acc": round(accuracy_score(y_train, mdl.predict(X_train)), 4),
            "Test Acc":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "CM": confusion_matrix(y_test, y_pred),
            "FI": dict(zip(X.columns, mdl.feature_importances_))
        })
    return trained, eval_rows, X_test, y_test, X.columns.tolist()

# ─────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────
df_raw = load_data()
trained_models, eval_rows, X_test, y_test, feature_names = preprocess_and_train(df_raw)
df = df_raw.copy()
df["Yield Label"] = df["Yield Success"].map(SUCCESS_LABELS)

# ─────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/wheat.png", width=60)
    st.title("🌾 Crop Yield\nDashboard")
    st.caption("Powered by Machine Learning")
    st.divider()

    st.markdown("### 🔍 Filters")

    crop_options  = sorted(df["Crop Type"].unique())
    season_options = sorted(df["Season"].unique())
    soil_options   = sorted(df["Soil Type"].unique())
    irr_options    = sorted(df["Irrigation Type"].unique())
    seed_options   = sorted(df["Seed Quality"].unique())
    farm_options   = sorted(df["Farming Practice"].unique())

    sel_crop    = st.multiselect("Crop Type",        crop_options,    default=crop_options)
    sel_season  = st.multiselect("Season",           season_options,  default=season_options)
    sel_soil    = st.multiselect("Soil Type",        soil_options,    default=soil_options)
    sel_irr     = st.multiselect("Irrigation Type",  irr_options,     default=irr_options)
    sel_seed    = st.multiselect("Seed Quality",     seed_options,    default=seed_options)
    sel_farm    = st.multiselect("Farming Practice", farm_options,    default=farm_options)

    st.markdown("#### 📊 Numeric Ranges")
    rf_min, rf_max = float(df["Rainfall (mm)"].min()), float(df["Rainfall (mm)"].max())
    sel_rain = st.slider("Rainfall (mm)", rf_min, rf_max, (rf_min, rf_max), step=1.0)

    ph_min, ph_max = float(df["Soil Ph"].min()), float(df["Soil Ph"].max())
    sel_ph = st.slider("Soil pH", ph_min, ph_max, (ph_min, ph_max), step=0.1)

    fert_min, fert_max = float(df["Fertilizer Used (kg)"].min()), float(df["Fertilizer Used (kg)"].max())
    sel_fert = st.slider("Fertilizer Used (kg)", fert_min, fert_max, (fert_min, fert_max), step=1.0)

    st.divider()
    st.caption("© 2025 Crop Yield Intelligence")

# Apply filters
mask = (
    df["Crop Type"].isin(sel_crop) &
    df["Season"].isin(sel_season) &
    df["Soil Type"].isin(sel_soil) &
    df["Irrigation Type"].isin(sel_irr) &
    df["Seed Quality"].isin(sel_seed) &
    df["Farming Practice"].isin(sel_farm) &
    df["Rainfall (mm)"].between(*sel_rain) &
    df["Soil Ph"].between(*sel_ph) &
    df["Fertilizer Used (kg)"].between(*sel_fert)
)
dff = df[mask].copy()

# ─────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#16A34A,#15803D);padding:28px 32px;border-radius:14px;margin-bottom:20px;'>
    <h1 style='color:white;margin:0;font-size:30px;'>🌾 Crop Yield Intelligence Dashboard</h1>
    <p style='color:#BBF7D0;margin:6px 0 0;font-size:15px;'>End-to-end crop performance analytics and ML model insights</p>
</div>
""", unsafe_allow_html=True)

st.caption(f"📦 Showing **{len(dff):,}** records after filters  |  Total dataset: **{len(df):,}** records")

# ─────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────
tab_overview, tab_eda, tab_model, tab_fi = st.tabs([
    "🏠 Overview", "📊 EDA", "🤖 Model Performance", "🌟 Feature Importance"
])

# ════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════
with tab_overview:

    # ── KPI Cards ──────────────────────────────
    st.markdown('<p class="section-header">📌 Key Performance Indicators</p>', unsafe_allow_html=True)

    success_rate  = round(dff["Yield Success"].mean() * 100, 1) if len(dff) else 0
    avg_exp_yield = round(dff["Expected Yield (kg per acre)"].mean(), 1) if len(dff) else 0
    avg_act_yield = round(dff["Actual Yield (kg per acre)"].mean(), 1) if len(dff) else 0
    avg_rainfall  = round(dff["Rainfall (mm)"].mean(), 1) if len(dff) else 0
    avg_ph        = round(dff["Soil Ph"].mean(), 2) if len(dff) else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "#16A34A", "✅ Yield Success Rate",       f"{success_rate}%",      "of filtered crops"),
        (c2, "#2563EB", "📦 Avg Expected Yield",        f"{avg_exp_yield:,} kg",  "per acre"),
        (c3, "#7C3AED", "📦 Avg Actual Yield",          f"{avg_act_yield:,} kg",  "per acre"),
        (c4, "#0891B2", "🌧️ Avg Rainfall",              f"{avg_rainfall} mm",    "per season"),
        (c5, "#D97706", "🧪 Avg Soil pH",               f"{avg_ph}",             "pH units"),
    ]
    for col, accent, label, value, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:{accent}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparative Bar Charts ──────────────────
    st.markdown('<p class="section-header">📊 Yield Success by Category</p>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        grp = dff.groupby(["Crop Type", "Yield Label"]).size().reset_index(name="Count")
        fig = px.bar(grp, x="Crop Type", y="Count", color="Yield Label",
                     barmode="group",
                     color_discrete_map={"Success": "#16A34A", "Failure": "#DC2626"},
                     title="Yield Success by Crop Type",
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(legend_title_text="", title_font_size=14, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        grp2 = dff.groupby(["Season", "Yield Label"]).size().reset_index(name="Count")
        season_colors_map = {}
        for s, c in SEASON_COLORS.items():
            season_colors_map[s] = c
        fig2 = px.bar(grp2, x="Season", y="Count", color="Season",
                      facet_col="Yield Label",
                      color_discrete_map=SEASON_COLORS,
                      title="Yield Success by Season",
                      template=PLOTLY_TEMPLATE)
        fig2.update_layout(title_font_size=14, height=360, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col_c:
        grp3 = dff.groupby(["Soil Type", "Yield Label"]).size().reset_index(name="Count")
        fig3 = px.bar(grp3, x="Soil Type", y="Count", color="Soil Type",
                      facet_col="Yield Label",
                      color_discrete_map=SOIL_COLORS,
                      title="Yield Success by Soil Type",
                      template=PLOTLY_TEMPLATE)
        fig3.update_layout(title_font_size=14, height=360, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── Socio-economic Pie Charts ───────────────
    st.markdown('<p class="section-header">💳 Socio-Economic Factors</p>', unsafe_allow_html=True)
    pie_c1, pie_c2 = st.columns(2)

    with pie_c1:
        credit_df = dff.groupby(["Access to Credit", "Yield Label"]).size().reset_index(name="Count")
        credit_df["Credit Label"] = credit_df["Access to Credit"].map({1: "With Credit", 0: "No Credit"})
        fig_p1 = px.sunburst(credit_df, path=["Credit Label", "Yield Label"], values="Count",
                              color="Yield Label",
                              color_discrete_map={"Success": "#16A34A", "Failure": "#DC2626"},
                              title="Yield Success by Access to Credit",
                              template=PLOTLY_TEMPLATE)
        fig_p1.update_layout(title_font_size=14, height=380)
        st.plotly_chart(fig_p1, use_container_width=True)

    with pie_c2:
        subsidy_df = dff.groupby(["Govt. Subsidy Received", "Yield Label"]).size().reset_index(name="Count")
        subsidy_df["Subsidy Label"] = subsidy_df["Govt. Subsidy Received"].map({1: "Got Subsidy", 0: "No Subsidy"})
        fig_p2 = px.sunburst(subsidy_df, path=["Subsidy Label", "Yield Label"], values="Count",
                              color="Yield Label",
                              color_discrete_map={"Success": "#16A34A", "Failure": "#DC2626"},
                              title="Yield Success by Govt. Subsidy",
                              template=PLOTLY_TEMPLATE)
        fig_p2.update_layout(title_font_size=14, height=380)
        st.plotly_chart(fig_p2, use_container_width=True)

    st.divider()

    # ── Trend: Farmer Experience vs Yield ──────
    st.markdown('<p class="section-header">📈 Yield Success vs Farmer Experience</p>', unsafe_allow_html=True)

    exp_grp = dff.groupby("Farmer Experience (years)")["Yield Success"].mean().reset_index()
    exp_grp.columns = ["Experience (years)", "Success Rate"]
    exp_grp["Success Rate (%)"] = (exp_grp["Success Rate"] * 100).round(1)

    fig_trend = px.line(
        exp_grp, x="Experience (years)", y="Success Rate (%)",
        markers=True,
        line_shape="spline",
        color_discrete_sequence=["#16A34A"],
        title="Yield Success Rate (%) by Farmer Experience",
        template=PLOTLY_TEMPLATE
    )
    fig_trend.update_traces(line_width=2.5, marker_size=7)
    fig_trend.add_hline(y=50, line_dash="dash", line_color="#DC2626",
                        annotation_text="50% Baseline", annotation_position="top left")
    fig_trend.update_layout(title_font_size=15, height=380,
                             xaxis_title="Farmer Experience (years)",
                             yaxis_title="Success Rate (%)")
    st.plotly_chart(fig_trend, use_container_width=True)

    with st.expander("📋 View Raw Data Sample"):
        st.dataframe(dff.head(50), use_container_width=True, height=280)


# ════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════
with tab_eda:

    # ── Distribution Plots ──────────────────────
    st.markdown('<p class="section-header">📦 Distribution Analysis</p>', unsafe_allow_html=True)
    dist_c1, dist_c2 = st.columns(2)

    with dist_c1:
        fig_box = px.box(
            dff, x="Yield Label", y="Rainfall (mm)",
            color="Yield Label",
            color_discrete_map={"Success": "#16A34A", "Failure": "#DC2626"},
            title="Rainfall Distribution vs Yield Success",
            template=PLOTLY_TEMPLATE,
            points="outliers"
        )
        fig_box.update_layout(title_font_size=14, height=400, showlegend=False,
                               xaxis_title="Yield Outcome", yaxis_title="Rainfall (mm)")
        st.plotly_chart(fig_box, use_container_width=True)

    with dist_c2:
        fig_hist = px.histogram(
            dff, x="Soil Ph", color="Yield Label",
            nbins=30, barmode="overlay", opacity=0.7,
            color_discrete_map={"Success": "#16A34A", "Failure": "#DC2626"},
            title="Soil pH Distribution: Success vs Failure",
            template=PLOTLY_TEMPLATE
        )
        fig_hist.update_layout(title_font_size=14, height=400,
                                xaxis_title="Soil pH", yaxis_title="Count",
                                legend_title_text="Yield Outcome")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ── Correlation Heatmap ─────────────────────
    st.markdown('<p class="section-header">🔥 Correlation Heatmap (Numeric Features)</p>', unsafe_allow_html=True)

    num_cols = ["Rainfall (mm)", "Soil Ph", "Fertilizer Used (kg)", "Pesticide Used (kg)",
                "Expected Yield (kg per acre)", "Actual Yield (kg per acre)",
                "Farm Size", "Soil Moisture (%)", "Avg Temperature (°C)",
                "Farmer Experience (years)", "Yield Success"]
    corr = dff[num_cols].corr().round(2)

    fig_hm = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        textfont_size=9,
        hoverongaps=False
    ))
    fig_hm.update_layout(
        title="Correlation Matrix — Numeric Variables",
        title_font_size=15,
        height=560,
        template=PLOTLY_TEMPLATE,
        xaxis_tickangle=-40,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()

    # ── Extra EDA ───────────────────────────────
    st.markdown('<p class="section-header">🌦️ Scatter: Rainfall vs Actual Yield</p>', unsafe_allow_html=True)
    fig_sc = px.scatter(
        dff, x="Rainfall (mm)", y="Actual Yield (kg per acre)",
        color="Yield Label",
        symbol="Season",
        color_discrete_map={"Success": "#16A34A", "Failure": "#DC2626"},
        opacity=0.6,
        title="Rainfall vs Actual Yield — coloured by outcome, shaped by season",
        template=PLOTLY_TEMPLATE
    )
    fig_sc.update_layout(title_font_size=14, height=420, legend_title_text="Outcome / Season")
    st.plotly_chart(fig_sc, use_container_width=True)

    with st.expander("📊 Descriptive Statistics"):
        st.dataframe(dff[num_cols].describe().T.style.background_gradient(cmap="Greens"),
                     use_container_width=True)


# ════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════
with tab_model:

    st.markdown('<p class="section-header">📐 Model Evaluation Summary</p>', unsafe_allow_html=True)

    metric_rows = [{k: v for k, v in r.items() if k not in ("CM", "FI")} for r in eval_rows]
    eval_disp = pd.DataFrame(metric_rows)
    eval_disp.columns = ["Model", "Train Accuracy", "Test Accuracy", "Precision", "Recall"]

    # Metric cards
    m_cols = st.columns(len(eval_rows))
    for col, row in zip(m_cols, eval_rows):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:#2563EB;margin-bottom:8px;">
                <div class="kpi-label">{row['Model']}</div>
                <div class="kpi-value" style="font-size:22px;">{row['Test Acc']*100:.1f}%</div>
                <div class="kpi-sub">Test Accuracy</div>
            </div>""", unsafe_allow_html=True)

    # Grouped bar chart
    fig_eval = go.Figure()
    metrics  = ["Train Acc", "Test Acc", "Precision", "Recall"]
    colors   = ["#2563EB", "#16A34A", "#7C3AED", "#D97706"]
    for metric, color in zip(metrics, colors):
        fig_eval.add_trace(go.Bar(
            name=metric,
            x=[r["Model"] for r in eval_rows],
            y=[r[metric] for r in eval_rows],
            marker_color=color,
            text=[f"{r[metric]:.3f}" for r in eval_rows],
            textposition="outside"
        ))
    fig_eval.update_layout(
        barmode="group",
        title="Model Comparison — All Metrics",
        template=PLOTLY_TEMPLATE,
        yaxis=dict(range=[0, 1.1], title="Score"),
        xaxis_title="Model",
        height=420,
        title_font_size=15,
        legend_title_text="Metric"
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    # Detailed table
    with st.expander("📋 Full Evaluation Table"):
        styled = eval_disp.set_index("Model").style.background_gradient(
            cmap="Greens", subset=["Test Accuracy", "Precision", "Recall"]
        ).format("{:.4f}", subset=["Train Accuracy", "Test Accuracy", "Precision", "Recall"])
        st.dataframe(styled, use_container_width=True)

    st.divider()

    # ── Confusion Matrices ──────────────────────
    st.markdown('<p class="section-header">🔲 Confusion Matrices</p>', unsafe_allow_html=True)
    cm_cols = st.columns(3)
    class_labels = ["Failure", "Success"]

    for col, row in zip(cm_cols, eval_rows):
        cm = row["CM"]
        with col:
            annotations = []
            quadrant_labels = [["TN", "FP"], ["FN", "TP"]]
            for i in range(2):
                for j in range(2):
                    annotations.append(dict(
                        x=class_labels[j], y=class_labels[i],
                        text=f"<b>{cm[i,j]}</b><br><span style='font-size:11px;color:#888'>{quadrant_labels[i][j]}</span>",
                        showarrow=False,
                        font=dict(size=16, color="#111827"),
                        xref="x", yref="y"
                    ))
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_labels,
                y=class_labels,
                colorscale=[[0, "#FEF2F2"], [0.5, "#BFDBFE"], [1, "#1D4ED8"]],
                showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            ))
            fig_cm.update_layout(
                title=dict(text=row["Model"], font_size=13, x=0.5),
                annotations=annotations,
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=320,
                template=PLOTLY_TEMPLATE,
                margin=dict(l=10, r=10, t=40, b=30)
            )
            st.plotly_chart(fig_cm, use_container_width=True)


# ════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════
with tab_fi:

    st.markdown('<p class="section-header">🌟 Feature Importance — All Models</p>', unsafe_allow_html=True)

    fi_colors = {"Decision Tree": "#2563EB", "Random Forest": "#16A34A", "Gradient Boosted Trees": "#F97316"}

    for row in eval_rows:
        name = row["Model"]
        fi   = row["FI"]
        fi_series = pd.Series(fi).sort_values(ascending=True)

        with st.container():
            fig_fi = go.Figure(go.Bar(
                x=fi_series.values,
                y=fi_series.index.tolist(),
                orientation="h",
                marker=dict(
                    color=fi_series.values,
                    colorscale="Greens",
                    showscale=False
                ),
                text=[f"{v:.4f}" for v in fi_series.values],
                textposition="outside",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>"
            ))
            fig_fi.update_layout(
                title=dict(text=f"Feature Importance — {name}", font_size=15, x=0),
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=430,
                template=PLOTLY_TEMPLATE,
                margin=dict(l=20, r=80, t=50, b=40)
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.divider()

    # ── Combined Comparison ─────────────────────
    with st.expander("📊 Side-by-side Feature Importance Comparison"):
        fi_data = {}
        for row in eval_rows:
            fi_data[row["Model"]] = row["FI"]

        fi_df = pd.DataFrame(fi_data).fillna(0)
        fi_df = fi_df.loc[fi_df.mean(axis=1).sort_values(ascending=False).index]

        fig_compare = go.Figure()
        for model_name, color in fi_colors.items():
            fig_compare.add_trace(go.Bar(
                name=model_name,
                x=fi_df.index.tolist(),
                y=fi_df[model_name].tolist(),
                marker_color=color
            ))
        fig_compare.update_layout(
            barmode="group",
            title="Feature Importance Comparison Across All Models",
            template=PLOTLY_TEMPLATE,
            height=460,
            xaxis_title="Feature",
            yaxis_title="Importance Score",
            xaxis_tickangle=-35,
            legend_title_text="Model",
            title_font_size=15
        )
        st.plotly_chart(fig_compare, use_container_width=True)
