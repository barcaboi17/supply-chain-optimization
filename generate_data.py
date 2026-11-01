"""Generate a SYNTHETIC supply-chain dataset (warehouses, customers, 12 months of shipments).
Coordinates are approximate city centres. All numbers are simulated for academic use."""
import numpy as np, pandas as pd
from math import radians, sin, cos, asin, sqrt

rng = np.random.default_rng(42)

def haversine_km(a, b):
    la1, lo1, la2, lo2 = map(radians, [a[0], a[1], b[0], b[1]])
    h = sin((la2-la1)/2)**2 + cos(la1)*cos(la2)*sin((lo2-lo1)/2)**2
    return 2*6371*asin(sqrt(h))

def road_km(a, b):            # road distance ~ 1.3 x straight-line
    return 1.3 * haversine_km(a, b)

WAREHOUSES = {  # name: (lat, lon, monthly_capacity_units)
    "WH-Delhi": (28.61, 77.21, 5600), "WH-Mumbai": (19.08, 72.88, 5600),
    "WH-Chennai": (13.08, 80.27, 4300), "WH-Kolkata": (22.57, 88.36, 4300),
    "WH-Nagpur": (21.15, 79.09, 3400), "WH-Bengaluru": (12.97, 77.59, 4000)}
CUSTOMERS = {   # name: (lat, lon, avg monthly demand)
    "Jaipur": (26.91, 75.79, 1100), "Lucknow": (26.85, 80.95, 1000), "Chandigarh": (30.73, 76.78, 800),
    "Ahmedabad": (23.02, 72.57, 1500), "Pune": (18.52, 73.86, 1600), "Hyderabad": (17.38, 78.49, 1900),
    "Kochi": (9.93, 76.27, 700), "Patna": (25.59, 85.14, 900), "Bhopal": (23.26, 77.41, 800),
    "Guwahati": (26.14, 91.74, 600), "Bhubaneswar": (20.30, 85.82, 700), "Coimbatore": (11.02, 76.96, 700),
    "Indore": (22.72, 75.86, 900), "Surat": (21.17, 72.83, 1000), "Visakhapatnam": (17.69, 83.22, 800),
    "Ranchi": (23.34, 85.31, 600), "Dehradun": (30.32, 78.03, 500), "Raipur": (21.25, 81.63, 700)}
CATEGORIES = {"FMCG": .45, "Electronics": .35, "Apparel": .20}

def build():
    wh = pd.DataFrame([(k, *v) for k, v in WAREHOUSES.items()], columns=["warehouse","lat","lon","monthly_capacity"])
    cu = pd.DataFrame([(k, *v) for k, v in CUSTOMERS.items()], columns=["customer","lat","lon","avg_monthly_demand"])
    dist = {(w, c): road_km(WAREHOUSES[w][:2], CUSTOMERS[c][:2]) for w in WAREHOUSES for c in CUSTOMERS}
    pd.DataFrame([(w, c, round(d, 1)) for (w, c), d in dist.items()],
                 columns=["warehouse","customer","distance_km"]).to_csv("data/distance_matrix.csv", index=False)
    rows, oid = [], 1
    season = {1:.95,2:.9,3:.95,4:1,5:1,6:.95,7:.95,8:1,9:1.05,10:1.2,11:1.25,12:1.15}
    growth = {2024: 1.0, 2025: 1.08}
    cats, cp = list(CATEGORIES), list(CATEGORIES.values())
    for yr in (2024, 2025):
      for m in range(1, 13):
        for c, (_, _, base) in CUSTOMERS.items():
            n_orders = max(rng.poisson(10), 1)
            total = base * season[m] * growth[yr] * rng.normal(1, .06)
            qtys = rng.dirichlet(np.ones(n_orders)) * total
            ranked = sorted(WAREHOUSES, key=lambda w: dist[(w, c)])
            for q in qtys:
                w = rng.choice(ranked[:3], p=[.6, .28, .12])   # no central planning: some orders ship from farther sites
                d = dist[(w, c)]
                express = rng.random() < .15
                promised = max(1, round(d/450 + (1 if express else 2)))
                actual = max(1, round(d/450 + 1 + rng.normal(0, 1.1)))
                qi = max(int(round(q)), 1)
                rows.append(dict(order_id=oid, order_date=f"{yr}-{m:02d}-{int(rng.integers(1, 29)):02d}", warehouse=w, customer=c,
                                 product_category=str(rng.choice(cats, p=cp)), priority="Express" if express else "Standard",
                                 quantity=qi, vehicle_type="Small" if qi < 100 else ("Medium" if qi < 300 else "Heavy"),
                                 distance_km=round(d, 1), transport_cost=round(qi*d*0.012 + 800, 2),
                                 promised_days=promised, delivery_days=actual, delayed=int(actual > promised)))
                oid += 1
    pd.DataFrame(rows).to_csv("data/shipments.csv", index=False)
    wh.to_csv("data/warehouses.csv", index=False); cu.to_csv("data/customers.csv", index=False)
    print(f"shipments: {len(rows)} rows")

if __name__ == "__main__":
    build()
