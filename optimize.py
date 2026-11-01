"""Transportation optimisation (PuLP) + vehicle routing (OR-Tools) + inventory policy."""
import math, json
import numpy as np, pandas as pd, pulp
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from generate_data import road_km

COST_PER_UNIT_KM = 0.012    # INR, same rate used in the shipment history
TRUCK_CAP, TRUCK_COST_KM, MAX_TRUCKS = 600, 28.0, 6   # units, INR/km, per warehouse

def load():
    wh = pd.read_csv("data/warehouses.csv").set_index("warehouse")
    cu = pd.read_csv("data/customers.csv").set_index("customer")
    sh = pd.read_csv("data/shipments.csv", parse_dates=["order_date"])
    dm = pd.read_csv("data/distance_matrix.csv").pivot(index="warehouse", columns="customer", values="distance_km")
    return wh, cu, sh, dm

def peak_demand(sh):
    """Peak month of the latest year -> (demand per customer, month, year)."""
    yr = int(sh.order_date.dt.year.max()); d = sh[sh.order_date.dt.year == yr]
    m = d.groupby([d.order_date.dt.month, "customer"]).quantity.sum().unstack()
    mo = int(m.sum(axis=1).idxmax()); return m.loc[mo], mo, yr

def greedy_baseline(wh, dem, dm):
    """Benchmark: serve each customer (largest first) from the nearest warehouse that still has stock."""
    cap = wh.monthly_capacity.astype(float).copy(); plan = {}
    for c in dem.sort_values(ascending=False).index:
        need = dem[c]
        for w in dm[c].sort_values().index:
            q = min(need, cap[w])
            if q > 0: plan[(w, c)] = plan.get((w, c), 0) + q; cap[w] -= q; need -= q
            if need <= 1e-9: break
    return plan

def plan_cost(plan, dm):
    return sum(q * dm.loc[w, c] * COST_PER_UNIT_KM for (w, c), q in plan.items())

def solve_transport(wh, dem, dm):
    """Min-cost transportation LP: min sum(cost*x) s.t. supply <= capacity, demand fully met."""
    prob = pulp.LpProblem("Transportation", pulp.LpMinimize)
    x = {(w, c): pulp.LpVariable(f"x_{i}_{j}", lowBound=0)
         for i, w in enumerate(wh.index) for j, c in enumerate(dem.index)}
    prob += pulp.lpSum(x[w, c] * dm.loc[w, c] * COST_PER_UNIT_KM for (w, c) in x)
    for w in wh.index: prob += pulp.lpSum(x[w, c] for c in dem.index) <= wh.loc[w, "monthly_capacity"]
    for c in dem.index: prob += pulp.lpSum(x[w, c] for w in wh.index) == dem[c]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal": return None      # infeasible: demand exceeds capacity
    return {k: v.value() for k, v in x.items() if v.value() and v.value() > 1e-6}

def solve_vrp(depot, custs, dem, wh, cu, time_limit=3):
    """Capacitated VRP from one warehouse with OR-Tools. Returns (routes, total_km)."""
    nodes = [depot] + custs
    pts = [wh.loc[depot, ["lat","lon"]].tolist()] + [cu.loc[c, ["lat","lon"]].tolist() for c in custs]
    D = [[int(road_km(a, b)) for b in pts] for a in pts]
    # a customer larger than one truck is split into full-truck visits
    demands = [0]; split = [None]
    expanded = [depot]
    for c in custs:
        q = int(round(dem[c]))
        while q > 0:
            take = min(q, TRUCK_CAP); expanded.append(c); demands.append(take); q -= take
    idx = {n: i for i, n in enumerate(nodes)}
    Dx = [[D[idx[a]][idx[b]] for b in expanded] for a in expanded]
    mgr = pywrapcp.RoutingIndexManager(len(expanded), MAX_TRUCKS * 4, 0)
    r = pywrapcp.RoutingModel(mgr)
    t = r.RegisterTransitCallback(lambda i, j: Dx[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
    r.SetArcCostEvaluatorOfAllVehicles(t)
    d = r.RegisterUnaryTransitCallback(lambda i: demands[mgr.IndexToNode(i)])
    r.AddDimensionWithVehicleCapacity(d, 0, [TRUCK_CAP] * (MAX_TRUCKS * 4), True, "Cap")
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = time_limit
    sol = r.SolveWithParameters(p)
    routes, total = [], 0
    for v in range(MAX_TRUCKS * 4):
        i = r.Start(v)
        if r.IsEnd(sol.Value(r.NextVar(i))): continue
        path, km, load = [depot], 0, 0
        while not r.IsEnd(i):
            n = mgr.IndexToNode(i); load += demands[n]
            if n: path.append(expanded[n])
            nxt = sol.Value(r.NextVar(i)); km += r.GetArcCostForVehicle(i, nxt, v); i = nxt
        path.append(depot); routes.append(dict(path=path, km=km, load=load)); total += km
    return routes, total

def direct_trips_km(depot, custs, dem, wh, cu):
    """Naive routing: a separate round trip per truckload to each customer."""
    km = 0
    for c in custs:
        trips = math.ceil(dem[c] / TRUCK_CAP)
        km += trips * 2 * road_km(wh.loc[depot, ["lat","lon"]].tolist(), cu.loc[c, ["lat","lon"]].tolist())
    return km

def inventory_policy(sh, cu, dm, plan):
    """Reorder point = mean lead-time demand + safety stock (z=1.65 -> ~95% service)."""
    m = sh.groupby([sh.order_date.dt.year, sh.order_date.dt.month, "customer"]).quantity.sum().unstack()
    rows = []
    for c in cu.index:
        mu_d, sd_d = m[c].mean() / 30, m[c].std() / 30       # per-day demand
        w = max((k[0] for k in plan if k[1] == c), key=lambda k: plan[(k, c)])
        lt = dm.loc[w, c] / 450 + 1                           # lead time in days
        ss = 1.65 * sd_d * math.sqrt(lt)
        rows.append(dict(customer=c, main_warehouse=w, lead_time_days=round(lt, 1), avg_daily_demand=round(mu_d, 1),
                         safety_stock=round(ss), reorder_point=round(mu_d * lt + ss)))
    return pd.DataFrame(rows)

def scenarios(wh, dem, dm):
    """What-if analysis: demand growth and warehouse capacity loss."""
    rows = []
    def go(name, w2, d2):
        p = solve_transport(w2, d2, dm)
        rows.append(dict(scenario=name, feasible=p is not None, cost_inr=round(plan_cost(p, dm)) if p else None,
                         units=int(d2.sum()), capacity=int(w2.monthly_capacity.sum())))
    go("Base (peak month)", wh, dem)
    for pct in (5, 10, 15):
        go(f"Demand +{pct}%", wh, dem * (1 + pct / 100))
    for w in wh.index:
        w2 = wh.copy(); w2.loc[w, "monthly_capacity"] *= 0.5; go(f"{w} at 50% capacity", w2, dem)
    out = pd.DataFrame(rows); base = out.cost_inr.iloc[0]
    out["cost_change_pct"] = ((out.cost_inr - base) / base * 100).round(2)
    return out

def run():
    wh, cu, sh, dm = load()
    dem, month, year = peak_demand(sh)
    base = greedy_baseline(wh, dem, dm); opt = solve_transport(wh, dem, dm)
    cb, co = plan_cost(base, dm), plan_cost(opt, dm)
    pd.DataFrame([(w, c, round(q)) for (w, c), q in opt.items()], columns=["warehouse","customer","units"]) \
        .to_csv("outputs/optimal_allocation.csv", index=False)
    # routing per warehouse, using the optimised allocation
    vrp_rows, naive_tot, vrp_tot = [], 0, 0
    routes_all = {}
    for w in wh.index:
        custs = sorted({c for (ww, c) in opt if ww == w}); sub = pd.Series({c: opt[(w, c)] for c in custs})
        if not custs: continue
        routes, km = solve_vrp(w, custs, sub, wh, cu); nk = direct_trips_km(w, custs, sub, wh, cu)
        routes_all[w] = routes; naive_tot += nk; vrp_tot += km
        vrp_rows.append(dict(warehouse=w, customers=len(custs), trucks_used=len(routes), direct_km=round(nk), optimised_km=km))
    vrp = pd.DataFrame(vrp_rows); vrp.to_csv("outputs/routing_summary.csv", index=False)
    inv = inventory_policy(sh, cu, dm, opt); inv.to_csv("outputs/inventory_policy.csv", index=False)
    hist = sh[(sh.order_date.dt.month == month) & (sh.order_date.dt.year == year)]
    ch = float((hist.quantity * hist.distance_km * COST_PER_UNIT_KM).sum())   # actual (synthetic) history, same cost rate
    sc = scenarios(wh, dem, dm); sc.to_csv("outputs/scenarios.csv", index=False)
    res = dict(peak_year=year, peak_month=month, historical_cost=round(ch), saving_vs_history_pct=round(100 * (ch - co) / ch, 2), peak_demand_units=int(dem.sum()), baseline_cost=round(cb), optimised_cost=round(co),
               saving_pct=round(100 * (cb - co) / cb, 2), direct_km=round(naive_tot), vrp_km=vrp_tot,
               km_saving_pct=round(100 * (naive_tot - vrp_tot) / naive_tot, 2),
               truck_cost_saving_inr=round((naive_tot - vrp_tot) * TRUCK_COST_KM))
    json.dump(res, open("outputs/results.json", "w"), indent=2)
    return wh, cu, sh, dm, dem, base, opt, vrp, inv, routes_all, res, sc

if __name__ == "__main__":
    *_, vrp, inv, routes, res, sc = run()
    print(json.dumps(res, indent=2)); print(vrp.to_string(index=False)); print(sc.to_string(index=False))
