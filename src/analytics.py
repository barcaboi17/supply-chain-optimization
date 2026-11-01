"""EDA + charts (static PNGs for README/PDF) for the supply-chain project."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp, numpy as np, pandas as pd

OUT = "outputs/charts/"
C = ["#1f4e79", "#2e86c1", "#f39c12", "#27ae60", "#c0392b", "#8e44ad", "#16a085"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 9})

def _save(fig, name): fig.tight_layout(); fig.savefig(OUT + name, dpi=160); plt.close(fig)

def eda(sh):
    sh = sh.copy(); sh["ym"] = sh.order_date.dt.to_period("M")
    m = sh.groupby("ym").quantity.sum(); x = m.index.astype(str)
    fig, ax = plt.subplots(figsize=(8, 3.4)); ax.plot(x, m.values, marker="o", ms=4, color=C[0], lw=2)
    ax.fill_between(range(len(m)), m.values, alpha=.12, color=C[0]); ax.set_xticks(range(0, len(m), 2)); ax.set_xticklabels(x[::2], rotation=45)
    ax.set(title="Monthly demand (units): festive peak every Oct-Dec, ~8% growth in 2025", ylabel="Units"); _save(fig, "01_monthly_demand.png")

    g = sh.groupby("warehouse").agg(cost=("transport_cost", "sum"), delay=("delayed", "mean"))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4)); n = g.index.str.replace("WH-", "")
    ax[0].bar(n, g.cost / 1e5, color=C[1]); ax[0].set(title="Transport cost by warehouse", ylabel="INR lakh")
    ax[1].bar(n, g.delay * 100, color=C[2]); ax[1].set(title="Delayed deliveries by warehouse", ylabel="% of orders")
    for a in ax: a.tick_params(axis="x", rotation=30)
    _save(fig, "02_cost_delay_by_warehouse.png")

    cat = sh.groupby("product_category").quantity.sum(); pr = sh.groupby("priority").delayed.mean() * 100
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].barh(cat.index, cat.values / 1e3, color=C[3]); ax[0].set(title="Units shipped by product category", xlabel="Thousand units")
    ax[1].bar(pr.index, pr.values, color=[C[1], C[4]]); ax[1].set(title="Delay rate: Standard vs Express", ylabel="%")
    _save(fig, "08_category_priority.png")
    return dict(orders=len(sh), columns=sh.shape[1] - 1, total_units=int(sh.quantity.sum()), total_cost=round(sh.transport_cost.sum()),
                delay_rate=round(sh.delayed.mean() * 100, 1),
                delay_standard=round(pr["Standard"], 1), delay_express=round(pr["Express"], 1),
                top_customers=sh.groupby("customer").quantity.sum().nlargest(3).to_dict(),
                warehouses=int(sh.warehouse.nunique()), customers=int(sh.customer.nunique()),
                period=f"{sh.order_date.min().date()} to {sh.order_date.max().date()}")

def cost_chart(res):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    v = [res["historical_cost"], res["baseline_cost"], res["optimised_cost"]]
    b = ax.bar(["Simulated\nhistorical pattern", "Greedy nearest-\nwarehouse", "PuLP\noptimised"], v, color=[C[4], C[2], C[3]], width=.6)
    ax.bar_label(b, labels=[f"INR {x:,.0f}" for x in v], fontsize=8.5, padding=3)
    ax.set(title="Transport cost in the peak month", ylim=(0, max(v) * 1.18), ylabel="INR"); _save(fig, "03_cost_comparison.png")

def heatmap(opt_alloc, wh, cu):
    M = pd.DataFrame(0.0, index=wh.index, columns=cu.index)
    for (w, c), q in opt_alloc.items(): M.loc[w, c] = q
    fig, ax = plt.subplots(figsize=(10, 3.4)); im = ax.imshow(M.values, cmap="Blues", aspect="auto"); ax.grid(False)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=60, ha="right"); ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M.iloc[i, j] > 0: ax.text(j, i, f"{M.iloc[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if M.iloc[i, j] > M.values.max() * .55 else "black")
    ax.set_title("Optimal allocation plan: units shipped from each warehouse to each customer"); fig.colorbar(im, ax=ax, label="units", shrink=.8); _save(fig, "09_allocation_heatmap.png")

def scenario_chart(sc):
    d = sc[sc.feasible].copy(); fig, ax = plt.subplots(figsize=(8, 3.8))
    col = [C[3] if i == 0 else (C[2] if "Demand" in s else C[1]) for i, s in enumerate(d.scenario)]
    ax.barh(d.scenario[::-1], d.cost_inr[::-1] / 1e3, color=col[::-1])
    for i, (v, p) in enumerate(zip(d.cost_inr[::-1], d.cost_change_pct[::-1])): ax.text(v / 1e3 + 1, i, f"{p:+.1f}%", va="center", fontsize=8)
    ax.set(title="What-if analysis: optimised cost under demand growth and capacity loss", xlabel="INR thousand"); ax.set_xlim(0, d.cost_inr.max() / 1e3 * 1.15)
    _save(fig, "10_scenarios.png")

def route_map(wh, cu, routes):
    fig, ax = plt.subplots(figsize=(6.4, 6.6)); colors = dict(zip(wh.index, C))
    for w, rs in routes.items():
        for r in rs:
            pts = [(wh.loc[n, "lon"], wh.loc[n, "lat"]) if n in wh.index else (cu.loc[n, "lon"], cu.loc[n, "lat"]) for n in r["path"]]
            ax.plot(*zip(*pts), color=colors[w], alpha=.6, lw=1.4)
    ax.scatter(cu.lon, cu.lat, c="#333", s=22, zorder=3)
    for n, r in cu.iterrows(): ax.annotate(n, (r.lon, r.lat), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for w, r in wh.iterrows(): ax.scatter(r.lon, r.lat, marker="s", s=95, c=colors[w], edgecolor="k", zorder=4, label=w)
    ax.legend(fontsize=7, loc="lower left"); ax.set(title="OR-Tools optimised delivery routes", xlabel="Longitude", ylabel="Latitude"); _save(fig, "04_route_map.png")

def routing_chart(vrp):
    x = np.arange(len(vrp)); fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(x - .2, vrp.direct_km, .4, label="Direct trips", color=C[4]); ax.bar(x + .2, vrp.optimised_km, .4, label="OR-Tools VRP", color=C[3])
    ax.set(xticks=x, xticklabels=vrp.warehouse.str.replace("WH-", ""), ylabel="km", title="Distance driven: direct trips vs optimised routes"); ax.legend(); _save(fig, "05_routing_km.png")

def _box(ax, xy, w, h, text, fc="#d6eaf8"):
    ax.add_patch(mp.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=fc, ec="#1f4e79", lw=1.4))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8.5)
def _arrow(ax, a, b): ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#333", lw=1.4))

def diagrams():
    fig, ax = plt.subplots(figsize=(8, 2.8)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    _box(ax, (.2, 1.0), 2, 1, "Planner /\nManager", "#fdebd0"); _box(ax, (3.9, .7), 2.2, 1.6, "Supply Chain Analytics &\nTransport Optimisation\nSystem (0)")
    _box(ax, (7.8, 1.0), 2, 1, "Dashboard, Charts,\nAllocation & Routes", "#d5f5e3")
    _arrow(ax, (2.2, 1.7), (3.9, 1.7)); _arrow(ax, (6.1, 1.5), (7.8, 1.5)); _arrow(ax, (3.9, 1.2), (2.2, 1.2))
    ax.text(3, 1.85, "orders, capacity,\ncosts, scenarios", fontsize=7, ha="center"); ax.text(7, 1.65, "decisions", fontsize=7, ha="center")
    ax.set_title("Data Flow Diagram - Level 0", fontsize=10); _save(fig, "06_dfd_level0.png")
    fig, ax = plt.subplots(figsize=(9, 3.2)); ax.axis("off"); ax.set_xlim(0, 11.4); ax.set_ylim(0, 3.2)
    names = ["CSV data\n(shipments,\nwarehouses,\ncustomers)", "Pandas\ncleaning & EDA", "PuLP (CBC)\ntransport LP", "OR-Tools\nVRP routing", "Scenarios &\ninventory\npolicy", "Streamlit\ndashboard +\nPDF/PNG reports"]
    xs = [.1, 1.95, 3.8, 5.65, 7.5, 9.35]
    for i, (n, x) in enumerate(zip(names, xs)):
        _box(ax, (x, 1.0), 1.55, 1.4, n, "#d6eaf8" if i not in (2, 3) else "#fdebd0")
        if i: _arrow(ax, (xs[i - 1] + 1.55, 1.7), (x, 1.7))
    ax.set_title("System Block Diagram", fontsize=10); _save(fig, "07_block_diagram.png")
