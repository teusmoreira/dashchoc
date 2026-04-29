import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings("ignore")

# ── Configuração da Página Streamlit ───────────────────────────────────────────
st.set_page_config(page_title="Chocolate Bar Explorer", layout="wide", page_icon="🍫")
st.title("🍫 Chocolate Bar Explorer")
st.markdown("Um dashboard escuro e elegante feito com Matplotlib dentro do Streamlit.")

# ── DEFINA O CAMINHO DO SEU ARQUIVO AQUI ───────────────────────────────────────
file_path = 'chocolate_bars.csv'

# ── Cores ──────────────────────────────────────────────────────────────────────
BG      = "#1a0f0a"
CARD    = "#2d1a0e"
BORDER  = "#5c3317"
GOLD    = "#c49a6c"
AMBER   = "#e8b87d"
RUST    = "#8c6847"
CREAM   = "#f5e6d3"
DARK    = "#3d2415"

CHOC_CMAP = LinearSegmentedColormap.from_list(
    "choc", ["#2d1a0e", "#8c6847", "#c49a6c", "#f5e6d3"]
)

# ── Estilo global ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   CREAM,
    "axes.titlecolor":   GOLD,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     12,
    "axes.grid":         True,
    "grid.color":        DARK,
    "grid.linewidth":    0.6,
    "xtick.color":       RUST,
    "ytick.color":       RUST,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "text.color":        CREAM,
    "legend.facecolor":  CARD,
    "legend.edgecolor":  BORDER,
    "legend.labelcolor": CREAM,
    "figure.dpi":        130,
})

def spine_style(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)

def title_bar(ax, text):
    ax.set_title(text, color=GOLD, fontsize=13, fontweight="bold", pad=12)

# ── Carregar dados (com cache do Streamlit) ────────────────────────────────────
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    df = load_data(file_path)
    
    # ── Construção da Figura Matplotlib ────────────────────────────────────────
    # Usar st.container para centralizar ou organizar a tela
    with st.container():
        fig1 = plt.figure(figsize=(18, 11))
        fig1.patch.set_facecolor(BG)

        gs = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.4, wspace=0.3, top=0.95, bottom=0.07)

        # 1. Distribuição de Ratings
        ax1 = fig1.add_subplot(gs[0, 0])
        counts, bins, patches = ax1.hist(df["rating"].dropna(), bins=20, color=GOLD, edgecolor=BORDER, alpha=0.8)
        title_bar(ax1, "Distribuição de Ratings")
        spine_style(ax1)

        # 2. Rating Médio por Ano
        ax2 = fig1.add_subplot(gs[0, 1])
        yearly = df.groupby("year_reviewed")["rating"].mean()
        ax2.plot(yearly.index, yearly.values, color=AMBER, lw=2, marker="o")
        title_bar(ax2, "Tendência de Qualidade")
        spine_style(ax2)

        # 3. Top Origens
        ax3 = fig1.add_subplot(gs[0, 2])
        origin_df = df.groupby("bean_origin")["rating"].agg(["mean", "count"]).query("count >= 10").nlargest(10, "mean")
        ax3.barh(origin_df.index, origin_df["mean"], color=GOLD, edgecolor=BORDER)
        title_bar(ax3, "Top 10 Origens (Cacau)")
        spine_style(ax3)

        # 4. Fabricantes por País (Expandido)
        ax5 = fig1.add_subplot(gs[1, 0:2])
        country_df = df["company_location"].value_counts().head(12)
        ax5.bar(country_df.index, country_df.values, color=RUST, edgecolor=BORDER)
        plt.setp(ax5.get_xticklabels(), rotation=30, ha="right")
        title_bar(ax5, "Principais Países Fabricantes")
        spine_style(ax5)

        # 5. Mix de Ingredientes (Com Legenda Lateral)
        ax6 = fig1.add_subplot(gs[1, 2])
        ing_counts = df["ingredients"].value_counts().head(6)
        labels_clean = [i.replace(",", " + ") for i in ing_counts.index]

        wedges, texts, autotexts = ax6.pie(ing_counts.values,
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           wedgeprops=dict(width=0.5, edgecolor=BG),
                                           colors=[CHOC_CMAP(i/6) for i in range(6)])

        for autotext in autotexts:
            autotext.set_color(CREAM)
            autotext.set_weight('bold')

        ax6.legend(wedges, labels_clean, title="Combinações", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        title_bar(ax6, "Mix de Ingredientes")

        # Em vez de plt.show(), passamos o objeto figure para o Streamlit renderizar
        st.pyplot(fig1)

except FileNotFoundError:
    st.error(f"❌ Erro ao carregar: Arquivo `{file_path}` não encontrado. Faça o upload no Colab.")
except Exception as e:
    st.error(f"❌ Ocorreu um erro inesperado: {e}")
