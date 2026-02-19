import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]
FACTS = BASE / "processed" / "facts_normalized.parquet"

st.set_page_config(page_title="LASTRO — Piloto", layout="wide")

@st.cache_data(show_spinner=False)
def load_data():
    if not FACTS.exists():
        st.error("Não achei: processed/facts_normalized.parquet. Rode o passo do Parquet primeiro.")
        st.stop()

    df = pd.read_parquet(FACTS)

    # garante numéricos
    df["net_royalty"] = pd.to_numeric(df.get("net_royalty"), errors="coerce").fillna(0.0)
    df["units"] = pd.to_numeric(df.get("units"), errors="coerce").fillna(0.0)

    # garante period como data
    df["period"] = pd.to_datetime(df.get("period"), errors="coerce")
    df = df.dropna(subset=["period"])
    df["period_date"] = df["period"].dt.date  # <- isso resolve o erro do date_input

    # preenche vazios para filtros não quebrarem
    for c, fill in [
        ("distributor", "(sem distribuidora)"),
        ("artist", "(sem artista)"),
        ("track_title", "(sem faixa)"),
        ("release_title", "(sem release)"),
        ("store", "(sem plataforma)"),
        ("country", "(sem país)"),
        ("currency", "(sem moeda)"),
        ("classification", "incerto"),
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": np.nan, "None": np.nan}).fillna(fill)
        else:
            df[c] = fill

    return df

df = load_data()

st.title("📊 LASTRO — Dashboard (Piloto)")
st.caption("Versão para apresentar no celular e desktop.")

# ======================
# SIDEBAR (FILTROS)
# ======================
st.sidebar.header("Filtros")

# Período
dmin = df["period_date"].min()
dmax = df["period_date"].max()
date_range = st.sidebar.date_input("Período", value=(dmin, dmax), min_value=dmin, max_value=dmax)

# Segurança: se o Streamlit retornar 1 data só
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = dmin, dmax

# Multiselects
all_dist = sorted(df["distributor"].unique().tolist())
all_class = sorted(df["classification"].unique().tolist())
all_store = sorted(df["store"].unique().tolist())
all_country = sorted(df["country"].unique().tolist())

sel_dist = st.sidebar.multiselect("Distribuidora", all_dist, default=all_dist)
sel_class = st.sidebar.multiselect("Classificação", all_class, default=all_class)
sel_store = st.sidebar.multiselect("Plataforma", all_store, default=[])
sel_country = st.sidebar.multiselect("País", all_country, default=[])

search = st.sidebar.text_input("Busca (faixa / release / artista / ISRC / UPC)", value="").strip().lower()
min_rev = st.sidebar.number_input("Receita mínima", min_value=0.0, value=0.0, step=10.0)
min_units = st.sidebar.number_input("Units mínimas", min_value=0.0, value=0.0, step=100.0)

# ======================
# APLICA FILTROS
# ======================
f = df.copy()

f = f[(f["period_date"] >= start_date) & (f["period_date"] <= end_date)]

if sel_dist:
    f = f[f["distributor"].isin(sel_dist)]
if sel_class:
    f = f[f["classification"].isin(sel_class)]
if sel_store:
    f = f[f["store"].isin(sel_store)]
if sel_country:
    f = f[f["country"].isin(sel_country)]

if search:
    cols = ["track_title", "release_title", "artist", "isrc", "upc"]
    mask = False
    for c in cols:
        if c in f.columns:
            mask = mask | f[c].astype(str).str.lower().str.contains(search, na=False)
    f = f[mask]

if min_rev > 0:
    f = f[f["net_royalty"] >= min_rev]
if min_units > 0:
    f = f[f["units"] >= min_units]

# ======================
# KPIs
# ======================
total_rev = float(f["net_royalty"].sum())
total_units = float(f["units"].sum())
n_tracks = int(f["track_title"].nunique())

c1, c2, c3 = st.columns(3)
c1.metric("Receita Total (net_royalty)", f"{total_rev:,.2f}")
c2.metric("Units", f"{int(total_units):,}")
c3.metric("Faixas únicas", f"{n_tracks:,}")

st.divider()

# ======================
# GRÁFICOS / TABELAS
# ======================
st.subheader("Evolução mensal (Receita)")

# cria coluna "YYYY-MM" para agrupar
f["period_ym"] = pd.to_datetime(f["period"]).dt.strftime("%Y-%m")
timeline = f.groupby("period_ym")["net_royalty"].sum().sort_index()
st.line_chart(timeline)

colA, colB = st.columns(2)

with colA:
    st.subheader("Receita por Distribuidora")
    by_dist = f.groupby("distributor")[["net_royalty", "units"]].sum().sort_values("net_royalty", ascending=False)
    st.dataframe(by_dist, use_container_width=True)

with colB:
    st.subheader("Receita por Classificação")
    by_class = f.groupby("classification")[["net_royalty", "units"]].sum().sort_values("net_royalty", ascending=False)
    st.dataframe(by_class, use_container_width=True)

colC, colD = st.columns(2)

with colC:
    st.subheader("Top Plataformas")
    by_store = f.groupby("store")[["net_royalty", "units"]].sum().sort_values("net_royalty", ascending=False).head(20)
    st.dataframe(by_store, use_container_width=True)

with colD:
    st.subheader("Top Países")
    by_country = f.groupby("country")[["net_royalty", "units"]].sum().sort_values("net_royalty", ascending=False).head(20)
    st.dataframe(by_country, use_container_width=True)

st.divider()

st.subheader("Top Faixas (por receita)")
top_tracks = (
    f.groupby(["track_title"], as_index=False)
     .agg(revenue=("net_royalty", "sum"), units=("units", "sum"),
          distributor=("distributor", lambda s: ", ".join(sorted(set(s)))))
     .sort_values("revenue", ascending=False)
     .head(30)
)
st.dataframe(top_tracks, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Exportar (CSV filtrado)")
csv_bytes = f.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Baixar CSV (com filtros)", data=csv_bytes, file_name="lastro_filtrado.csv", mime="text/csv")
