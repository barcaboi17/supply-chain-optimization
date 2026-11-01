"""Run the full pipeline:  python main.py"""
import sys, json; sys.path.insert(0, "src")
import generate_data, analytics, optimize

generate_data.build()
wh, cu, sh, dm, dem, base, opt, vrp, inv, routes, res, sc = optimize.run()
eda = analytics.eda(sh)
analytics.cost_chart(res); analytics.heatmap(opt, wh, cu); analytics.scenario_chart(sc)
analytics.route_map(wh, cu, routes); analytics.routing_chart(vrp); analytics.diagrams()
json.dump(eda, open("outputs/eda_summary.json", "w"), indent=2)
print(json.dumps(res, indent=2)); print(json.dumps(eda, indent=2))
