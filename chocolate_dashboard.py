import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="🍫 Chocolate Bars Explorer",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta de cores chocolate ───────────────────────────────────────────────
COLORS = {
    "dark":       "#2C1503",
    "espresso":   "#3E1C00",
    "cocoa":      "#5C3317",
    "milk":       "#7B4F2E",
    "caramel":    "#A0522D",
    "gold":       "#C8860A",
    "cream":      "#F5DEB3",
    "white":      "#FFF8F0",
}

PALETTE = [
    "#5C3317", "#A0522D", "#C8860A", "#7B4F2E",
    "#3E1C00", "#D2691E", "#8B4513", "#F4A460",
    "#DEB887", "#CD853F",
]

# ── CSS customizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #1A0A00;
    color: #F5DEB3;
}

.main { background-color: #1A0A00; }

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #C8860A !important;
}

.stMetric {
    background: linear-gradient(135deg, #2C1503, #3E1C00);
    border: 1px solid rgba(200, 134, 10, 0.25);
    border-radius: 12px;
    padding: 16px !important;
}

.stMetric label { color: #DEB887 !important; font-size: 0.85rem !important; }
.stMetric [data-testid="stMetricValue"] { color: #C8860A !important; font-size: 2rem !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2C1503 0%, #1A0A00 100%);
    border-right: 1px solid rgba(200, 134, 10, 0.25);
}

section[data-testid="stSidebar"] * { color: #F5DEB3 !important; }

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background-color: #3E1C00;
    border-color: rgba(200, 134, 10, 0.4);
    color: #F5DEB3;
}

hr { border-color: rgba(200, 134, 10, 0.25); }
.stPlotlyChart { border-radius: 12px; overflow: hidden; }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: #C8860A;
    text-align: center;
    letter-spacing: 2px;
    margin-bottom: 0;
}
.hero-sub {
    text-align: center;
    color: #DEB887;
    font-size: 1.1rem;
    margin-top: 4px;
    letter-spacing: 1px;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #C8860A;
    border-left: 4px solid #C8860A;
    padding-left: 12px;
    margin-top: 32px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Configurações de tema Plotly (Nativo) ──────────────────────────────────
pio.templates["choc_theme"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#1A0A00",
        plot_bgcolor="#2C1503",
        font=dict(color="#F5DEB3", family="Lato"),
        title=dict(font=dict(color="#C8860A", family="Playfair Display", size=18)),
        legend=dict(bgcolor="rgba(44, 21, 3, 0.5)", bordercolor="rgba(200, 134, 10, 0.25)"),
        xaxis=dict(gridcolor="rgba(200, 134, 10, 0.15)", linecolor="rgba(200, 134, 10, 0.25)", tickcolor="#DEB887"),
        yaxis=dict(gridcolor="rgba(200, 134, 10, 0.15)", linecolor="rgba(200, 134, 10, 0.25)", tickcolor="#DEB887"),
        colorway=PALETTE,
    )
)
# Aplica o nosso tema de chocolate por padrão a todos os gráficos!
pio.templates.default = "choc_theme"

# ── Carregamento dos dados ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("chocolate_bars.csv")
    df.columns = df.columns.str.strip()
    df["cocoa_percent"] = pd.to_numeric(df["cocoa_percent"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["year_reviewed"] = pd.to_numeric(df["year_reviewed"], errors="coerce")
    df["num_ingredients"] = pd.to_numeric(df["num_ingredients"], errors="coerce")
    return df.dropna(subset=["rating", "cocoa_percent"])

df = load_data()

# ── Sidebar — Filtros ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍫 Filtros")
    st.markdown("---")

    anos = sorted(df["year_reviewed"].dropna().unique().astype(int))
    ano_range = st.slider(
        "Ano de avaliação",
        min_value=int(min(anos)), max_value=int(max(anos)),
        value=(int(min(anos)), int(max(anos)))
    )

    rating_range = st.slider(
        "Rating mínimo / máximo",
        min_value=float(df["rating"].min()),
        max_value=float(df["rating"].max()),
        value=(float(df["rating"].min()), float(df["rating"].max())),
        step=0.25
    )

    top_paises = df["company_location"].value_counts().head(20).index.tolist()
    pais_sel = st.multiselect(
        "País do fabricante",
        options=sorted(df["company_location"].unique()),
        default=[]
    )

    st.markdown("---")
    st.markdown("##### Sobre os dados")
    st.markdown("Base: avaliações de barras de chocolate artesanal de todo o mundo.")

# ── Filtragem ───────────────────────────────────────────────────────────────
mask = (
    df["year_reviewed"].between(ano_range[0], ano_range[1]) &
    df["rating"].between(rating_range[0], rating_range[1])
)
if pais_sel:
    mask = mask & df["company_location"].isin(pais_sel)

dff = df[mask]

# ── Cabeçalho ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🍫 Chocolate Bars Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Uma jornada sensorial pelos dados do chocolate artesanal</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── KPIs ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🍫 Barras avaliadas", f"{len(dff):,}")
k2.metric("⭐ Rating médio", f"{dff['rating'].mean():.2f}")
k3.metric("🌿 % Cacau médio", f"{dff['cocoa_percent'].mean():.1f}%")
k4.metric("🏭 Fabricantes", f"{dff['manufacturer'].nunique():,}")
k5.metric("🌍 Origens de grão", f"{dff['bean_origin'].nunique():,}")

st.markdown("<hr>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# GRÁFICO 1 — Distribuição de ratings
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📊 Distribuição dos Ratings</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    rating_counts = dff["rating"].value_counts().sort_index()
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=rating_counts.index,
        y=rating_counts.values,
        marker=dict(
            color=rating_counts.values,
            colorscale=[[0, "#3E1C00"], [0.5, "#A0522D"], [1, "#C8860A"]],
            showscale=False,
            line=dict(color="rgba(245, 222, 179, 0.15)", width=0.5),
        ),
        text=rating_counts.values,
        textposition="outside",
        textfont=dict(color="#DEB887", size=11),
        hovertemplate="Rating: %{x}<br>Quantidade: %{y}<extra></extra>",
    ))
    fig1.update_layout(
        title="Quantidade de barras por rating",
        xaxis_title="Rating",
        yaxis_title="Nº de barras",
        bargap=0.15,
        height=360,
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    bins = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    labels = ["1–2", "2–2.5", "2.5–3", "3–3.5", "3.5–4", "4–4.5", "4.5–5"]
    dff2 = dff.copy()
    dff2["faixa"] = pd.cut(dff2["rating"], bins=bins, labels=labels, right=True)
    pie_data = dff2["faixa"].value_counts().sort_index()
    fig2 = go.Figure(go.Pie(
        labels=pie_data.index,
        values=pie_data.values,
        hole=0.45,
        marker=dict(colors=PALETTE, line=dict(color="#1A0A00", width=2)),
        textinfo="label+percent",
        textfont=dict(color="#F5DEB3", size=11),
        hovertemplate="%{label}: %{value} barras (%{percent})<extra></extra>",
    ))
    fig2.update_layout(
        title="Faixas de rating",
        height=360,
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# GRÁFICO 2 — Cacau % vs Rating (scatter)
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🌿 Teor de Cacau × Rating</div>', unsafe_allow_html=True)

fig3 = px.scatter(
    dff,
    x="cocoa_percent",
    y="rating",
    color="rating",
    color_continuous_scale=[[0, "#3E1C00"], [0.4, "#7B4F2E"], [0.7, "#C8860A"], [1, "#F5DEB3"]],
    hover_data={"manufacturer": True, "bean_origin": True, "bar_name": True,
                "cocoa_percent": True, "rating": True},
    labels={"cocoa_percent": "% de Cacau", "rating": "Rating"},
    opacity=0.75,
    size_max=8,
)
bins_x = pd.cut(dff["cocoa_percent"], bins=20)
trend = dff.groupby(bins_x, observed=False)["rating"].mean().reset_index()
trend["mid"] = trend["cocoa_percent"].apply(lambda x: x.mid if hasattr(x, "mid") else None)
trend = trend.dropna(subset=["mid"])
fig3.add_trace(go.Scatter(
    x=trend["mid"], y=trend["rating"],
    mode="lines",
    line=dict(color="#C8860A", width=2.5, dash="dot"),
    name="Tendência média",
))
fig3.update_layout(
    title="Relação entre teor de cacau e qualidade percebida",
    xaxis_title="% de Cacau",
    yaxis_title="Rating",
    height=420,
    coloraxis_showscale=False,
)
st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# GRÁFICO 3 — Top países fabricantes e origens do grão
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🌍 Fabricantes & Origens do Grão</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    top_fab = (
        dff.groupby("company_location")["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .query("count >= 5")
        .sort_values("mean", ascending=True)
        .tail(15)
    )
    fig4 = go.Figure(go.Bar(
        x=top_fab["mean"],
        y=top_fab["company_location"],
        orientation="h",
        marker=dict(
            color=top_fab["mean"],
            colorscale=[[0, "#3E1C00"], [0.5, "#A0522D"], [1, "#C8860A"]],
            showscale=False,
            line=dict(color="rgba(245, 222, 179, 0.15)", width=0.5),
        ),
        text=[f"{v:.2f} ({c} barras)" for v, c in zip(top_fab["mean"], top_fab["count"])],
        textposition="outside",
        textfont=dict(color="#DEB887", size=10),
        hovertemplate="%{y}<br>Rating médio: %{x:.2f}<extra></extra>",
    ))
    fig4.update_layout(
        title="Top 15 países fabricantes<br>(por rating médio, mín. 5 barras)",
        xaxis_title="Rating médio",
        yaxis_title="",
        height=460,
        margin=dict(l=10),
    )
    st.plotly_chart(fig4, use_container_width=True)

with col4:
    top_ori = (
        dff.groupby("bean_origin")["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .query("count >= 8")
        .sort_values("mean", ascending=True)
        .tail(15)
    )
    fig5 = go.Figure(go.Bar(
        x=top_ori["mean"],
        y=top_ori["bean_origin"],
        orientation="h",
        marker=dict(
            color=top_ori["mean"],
            colorscale=[[0, "#2C1503"], [0.5, "#7B4F2E"], [1, "#DEB887"]],
            showscale=False,
            line=dict(color="rgba(245, 222, 179, 0.15)", width=0.5),
        ),
        text=[f"{v:.2f} ({c} barras)" for v, c in zip(top_ori["mean"], top_ori["count"])],
        textposition="outside",
        textfont=dict(color="#DEB887", size=10),
        hovertemplate="%{y}<br>Rating médio: %{x:.2f}<extra></extra>",
    ))
    fig5.update_layout(
        title="Top 15 origens do grão de cacau<br>(rating médio, mín. 8 barras)",
        xaxis_title="Rating médio",
        yaxis_title="",
        height=460,
        margin=dict(l=10),
    )
    st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# GRÁFICO 4 — Evolução do rating ao longo dos anos
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📅 Evolução Temporal das Avaliações</div>', unsafe_allow_html=True)

ano_stats = (
    dff.groupby("year_reviewed")["rating"]
    .agg(["mean", "median", "count"])
    .reset_index()
    .sort_values("year_reviewed")
)

fig6 = go.Figure()
fig6.add_trace(go.Scatter(
    x=ano_stats["year_reviewed"], y=ano_stats["mean"],
    mode="lines+markers",
    name="Média",
    line=dict(color="#C8860A", width=3),
    marker=dict(size=8, color="#C8860A", line=dict(color="#F5DEB3", width=1.5)),
    hovertemplate="Ano: %{x}<br>Rating médio: %{y:.2f}<extra></extra>",
))
fig6.add_trace(go.Scatter(
    x=ano_stats["year_reviewed"], y=ano_stats["median"],
    mode="lines+markers",
    name="Mediana",
    line=dict(color="#DEB887", width=2, dash="dash"),
    marker=dict(size=6, color="#DEB887"),
    hovertemplate="Ano: %{x}<br>Rating mediano: %{y:.2f}<extra></extra>",
))
fig6.add_trace(go.Bar(
    x=ano_stats["year_reviewed"], y=ano_stats["count"],
    name="Nº de avaliações",
    yaxis="y2",
    marker=dict(color="rgba(92, 51, 23, 0.4)"),
    hovertemplate="Ano: %{x}<br>Avaliações: %{y}<extra></extra>",
))
fig6.update_layout(
    title="Rating médio e volume de avaliações por ano",
    xaxis_title="Ano",
    yaxis=dict(title="Rating"),
    yaxis2=dict(title="Nº de avaliações", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=380,
)
st.plotly_chart(fig6, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# GRÁFICO 5 — Distribuição por nº de ingredientes
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🧪 Complexidade da Receita</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    ingr_stats = (
        dff.dropna(subset=["num_ingredients"])
        .groupby("num_ingredients")["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("num_ingredients")
    )
    fig7 = go.Figure()
    fig7.add_trace(go.Bar(
        x=ingr_stats["num_ingredients"].astype(int).astype(str),
        y=ingr_stats["count"],
        name="Nº de barras",
        marker=dict(color="rgba(92, 51, 23, 0.7)"),
        hovertemplate="Ingredientes: %{x}<br>Barras: %{y}<extra></extra>",
    ))
    fig7.add_trace(go.Scatter(
        x=ingr_stats["num_ingredients"].astype(int).astype(str),
        y=ingr_stats["mean"],
        mode="lines+markers+text",
        name="Rating médio",
        yaxis="y2",
        line=dict(color="#C8860A", width=2.5),
        marker=dict(size=9, color="#C8860A"),
        text=[f"{v:.2f}" for v in ingr_stats["mean"]],
        textposition="top center",
        textfont=dict(color="#C8860A", size=11),
        hovertemplate="Ingredientes: %{x}<br>Rating médio: %{y:.2f}<extra></extra>",
    ))
    fig7.update_layout(
        title="Nº de ingredientes × rating",
        xaxis_title="Nº de ingredientes",
        yaxis=dict(title="Nº de barras"),
        yaxis2=dict(title="Rating médio", overlaying="y", side="right", showgrid=False,
                    range=[2.5, 4.5]),
        height=380,
    )
    st.plotly_chart(fig7, use_container_width=True)

with col6:
    dff3 = dff.copy()
    dff3["faixa_cacau"] = pd.cut(
        dff3["cocoa_percent"],
        bins=[50, 60, 65, 70, 75, 80, 85, 100],
        labels=["≤60%", "60–65%", "65–70%", "70–75%", "75–80%", "80–85%", "≥85%"],
        right=True
    )
    fig8 = go.Figure()
    for i, faixa in enumerate(["≤60%", "60–65%", "65–70%", "70–75%", "75–80%", "80–85%", "≥85%"]):
        subset = dff3[dff3["faixa_cacau"] == faixa]["rating"]
        if len(subset) == 0:
            continue
        fig8.add_trace(go.Box(
            y=subset,
            name=faixa,
            boxmean=True,
            marker=dict(color=PALETTE[i % len(PALETTE)]),
            line=dict(color=PALETTE[i % len(PALETTE)]),
            hovertemplate=f"{faixa}<br>Rating: %{{y}}<extra></extra>",
        ))
    fig8.update_layout(
        title="Distribuição do rating por faixa de % cacau",
        yaxis_title="Rating",
        xaxis_title="% de Cacau",
        showlegend=False,
        height=380,
    )
    st.plotly_chart(fig8, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TABELA — Top barras
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🏆 As Barras Mais Bem Avaliadas</div>', unsafe_allow_html=True)

top_bars = (
    dff.sort_values("rating", ascending=False)
    [["manufacturer", "bar_name", "bean_origin", "company_location",
      "cocoa_percent", "rating", "year_reviewed", "review"]]
    .head(20)
    .rename(columns={
        "manufacturer": "Fabricante",
        "bar_name": "Barra",
        "bean_origin": "Origem do grão",
        "company_location": "País",
        "cocoa_percent": "% Cacau",
        "rating": "⭐ Rating",
        "year_reviewed": "Ano",
        "review": "Notas de degustação",
    })
)

st.dataframe(
    top_bars.reset_index(drop=True),
    use_container_width=True,
    height=380,
)

# ── Rodapé ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#7B4F2E; font-size:0.85rem;">'
    "🍫 Chocolate Bars Explorer · Dados: Flavors of Cacao rating database"
    "</div>",
    unsafe_allow_html=True,
)
