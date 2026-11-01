"""Streamlit dashboard:  streamlit run app.py"""
import sys; sys.path.insert(0, "src")
import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
import optimize as op

st.set_page_config(page_title="Supply Chain Optimizer", page_icon="🚚", layout="wide")
st.markdown("<style>.block-container{padding-top:1.4rem} [data-testid=stMetricValue]{font-size:1.6rem}</style>", unsafe_allow_html=True)

@st.cache_data
def load():
    wh, cu, sh, dm = op.load(); return wh, cu, sh, dm
wh0, cu, sh, dm = load()

st.title("🚚 Supply Chain Analytics & Transportation Optimization")
st.caption("PuLP finds the cheapest warehouse-to-customer allocation · OR-Tools builds truck routes · all data is synthetic")

# ---------- sidebar controls ----------
yr = int(sh.order_date.dt.year.max()); d = sh[sh.order_date.dt.year == yr]
mon = d.groupby([d.order_date.dt.month, "customer"]).quantity.sum().unstack()
st.sidebar.header("Planning inputs")
month = st.sidebar.selectbox("Demand month", list(mon.index), index=list(mon.index).index(int(mon.sum(axis=1).idxmax())),
                             format_func=lambda m: pd.Timestamp(2025, m, 1).strftime("%B") + (" (peak)" if m == int(mon.sum(axis=1).idxmax()) else ""))
uplift = st.sidebar.slider("Demand change (%)", -20, 30, 0)
st.sidebar.subheader("Warehouse capacity (%)")
wh = wh0.copy()
for w in wh.index: wh.loc[w, "monthly_capacity"] = wh0.loc[w, "monthly_capacity"] * st.sidebar.slider(w, 0, 100, 100, step=10) / 100
dem = mon.loc[month] * (1 + uplift / 100)

# ---------- solve ----------
opt = op.solve_transport(wh, dem, dm)
base = op.greedy_baseline(wh, dem, dm)
if opt is None:
    st.error(f"Infeasible: demand ({dem.sum():,.0f} units) is above total capacity ({wh.monthly_capacity.sum():,.0f}). Raise capacity or lower demand.")
    st.stop()
co, cb = op.plan_cost(opt, dm), op.plan_cost(base, dm)
k = st.columns(4)
k[0].metric("Demand to serve", f"{dem.sum():,.0f} units"); k[1].metric("Optimised cost", f"INR {co:,.0f}")
k[2].metric("Greedy baseline", f"INR {cb:,.0f}"); k[3].metric("Saving vs baseline", f"{(cb - co) / cb * 100:.1f}%", f"INR {cb - co:,.0f}")

t1, t2, t3, t4, t5, t6 = st.tabs(["📈 Overview", "🏭 Allocation", "🗺️ Routes", "🧪 Scenarios", "📦 Inventory", "🗃️ Data"])

with t1:
    s2 = sh.assign(month=sh.order_date.dt.to_period("M").astype(str)).groupby("month").quantity.sum().reset_index()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.area(s2, x="month", y="quantity", title="Monthly demand (units)", color_discrete_sequence=["#1f4e79"]), width="stretch")
    g = sh.groupby("warehouse").agg(cost=("transport_cost", "sum"), delay=("delayed", "mean")).reset_index()
    c2.plotly_chart(px.bar(g, x="warehouse", y="delay", title="Delay rate by warehouse", color_discrete_sequence=["#f39c12"]).update_yaxes(tickformat=".0%"), width="stretch")
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.pie(sh.groupby("product_category").quantity.sum().reset_index(), names="product_category", values="quantity", title="Units by product category", hole=.45), width="stretch")
    c4.plotly_chart(px.bar(sh.groupby("priority").delayed.mean().reset_index(), x="priority", y="delayed", title="Delay rate: Standard vs Express", color_discrete_sequence=["#c0392b"]).update_yaxes(tickformat=".0%"), width="stretch")

with t2:
    M = pd.DataFrame(0.0, index=wh.index, columns=cu.index)
    for (w, c), q in opt.items(): M.loc[w, c] = q
    st.plotly_chart(px.imshow(M.round(), text_auto=True, aspect="auto", color_continuous_scale="Blues", title="Units shipped: warehouse → customer"), width="stretch")
    util = (M.sum(axis=1) / wh0.monthly_capacity * 100).round(1).rename("Utilisation of full capacity (%)")
    st.dataframe(util.to_frame(), width="stretch")

@st.cache_data(show_spinner="Solving truck routes with OR-Tools...")
def routes_for(plan_items, capacity_key):
    plan = dict(plan_items); out = {}; tot = 0; direct = 0
    for w in wh.index:
        cs = sorted({c for (ww, c) in plan if ww == w})
        if not cs: continue
        sub = pd.Series({c: plan[(w, c)] for c in cs}); r, km = op.solve_vrp(w, cs, sub, wh, cu, time_limit=1)
        out[w] = r; tot += km; direct += op.direct_trips_km(w, cs, sub, wh, cu)
    return out, tot, direct

with t3:
    routes, vkm, dkm = routes_for(tuple(sorted(opt.items())), tuple(wh.monthly_capacity))
    a, b, c = st.columns(3); a.metric("Direct-trip distance", f"{dkm:,.0f} km"); b.metric("Optimised routes", f"{vkm:,.0f} km"); c.metric("Distance saved", f"{(dkm - vkm) / dkm * 100:.1f}%")
    fig = go.Figure(); pal = px.colors.qualitative.Bold
    for i, (w, rs) in enumerate(routes.items()):
        for r in rs:
            pts = [(wh.loc[n, "lat"], wh.loc[n, "lon"]) if n in wh.index else (cu.loc[n, "lat"], cu.loc[n, "lon"]) for n in r["path"]]
            fig.add_trace(go.Scattergeo(lat=[p[0] for p in pts], lon=[p[1] for p in pts], mode="lines", line=dict(width=2, color=pal[i % len(pal)]),
                                        hoverinfo="text", text=f"{w}: {r['load']} units, {r['km']} km", showlegend=False))
    fig.add_trace(go.Scattergeo(lat=cu.lat, lon=cu.lon, text=cu.index, mode="markers+text", textposition="top right", marker=dict(size=6, color="#333"), name="Customers"))
    fig.add_trace(go.Scattergeo(lat=wh.lat, lon=wh.lon, text=wh.index, mode="markers", marker=dict(size=13, symbol="square", color="#c0392b", line=dict(width=1, color="black")), name="Warehouses"))
    fig.update_geos(scope="asia", fitbounds="locations", showcountries=True, showland=True, landcolor="#f4f4f0"); fig.update_layout(height=640, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

with t4:
    st.caption("Re-solves the model for demand growth and for losing half of one warehouse's capacity (at current sidebar demand).")
    sc = op.scenarios(wh0, dem, dm); st.dataframe(sc, width="stretch", hide_index=True)
    ok = sc[sc.feasible]; st.plotly_chart(px.bar(ok, x="cost_inr", y="scenario", orientation="h", title="Optimised cost by scenario (INR)", color_discrete_sequence=["#2e86c1"]).update_yaxes(autorange="reversed"), width="stretch")

with t5:
    st.caption("Reorder point = average lead-time demand + safety stock (1.65 × σ × √lead time, ≈95% service level).")
    st.dataframe(op.inventory_policy(sh, cu, dm, opt), width="stretch", hide_index=True)

with t6:
    st.caption("Synthetic data generated by src/generate_data.py (seed 42).")
    st.dataframe(sh.head(500), width="stretch", hide_index=True)
    st.download_button("Download shipments.csv", sh.to_csv(index=False), "shipments.csv")
