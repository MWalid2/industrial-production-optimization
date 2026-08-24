from pathlib import Path

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OVERTIME_WAGE_PER_HOUR = 35  # $ per operator-hour, default


def load_base_parameters(overtime_wage=OVERTIME_WAGE_PER_HOUR):
    products = pd.read_csv(DATA_DIR / "optimization/products.csv")
    machines = pd.read_csv(DATA_DIR / "optimization/machines.csv")
    routing = pd.read_csv(DATA_DIR / "optimization/routing.csv")
    materials = pd.read_csv(DATA_DIR / "optimization/materials.csv")
    bills = pd.read_csv(DATA_DIR / "optimization/bill_of_materials.csv")
    availability = pd.read_csv(DATA_DIR / "optimization/machine_availability.csv")
    forecast = pd.read_csv(DATA_DIR / "processed/forecasts/forecast.csv")

    forecast_weeks = sorted(forecast["week"].unique())
    week_to_t = {week: t for t, week in enumerate(forecast_weeks, start=1)}
    forecast["t"] = forecast["week"].map(week_to_t)
    availability["t"] = availability["week"]

    P = products["product_id"].tolist()
    M = machines["machine_id"].tolist()
    T = sorted(forecast["t"].unique().tolist())
    R = materials["material_id"].tolist()
    PM = list(routing[["product_id", "machine_id"]].itertuples(index=False, name=None))
    idx_pmt = [(p, m, t) for (p, m) in PM for t in T]

    overtime_cost_per_hour = {
        row.machine_id: overtime_wage * row.labor_requirement
        for row in machines.itertuples()
    }
    overtime_hours = machines.set_index("machine_id")["overtime_hours"].to_dict()
    availability_fraction = machines.set_index("machine_id")["availability"].to_dict()
    available_hours = {
        (row.machine_id, row.t): row.available_hours * availability_fraction[row.machine_id]
        for row in availability.itertuples()
    }

    production_cost = products.set_index("product_id")["production_cost"].to_dict()
    holding_cost = products.set_index("product_id")["holding_cost"].to_dict()
    shortage_cost = products.set_index("product_id")["shortage_cost"].to_dict()
    max_inventory = products.set_index("product_id")["max_inventory"].to_dict()
    safety_stock = products.set_index("product_id")["safety_stock"].to_dict()
    initial_inventory = products.set_index("product_id")["safety_stock"].to_dict()
    demand = {(row.product_id, row.t): row.forecast for row in forecast.itertuples()}

    processing_time = {(row.product_id, row.machine_id): row.processing_time for row in routing.itertuples()}
    material_availability = materials.set_index("material_id")["availability"].to_dict()
    material_qty = {(row.product_id, row.material_id): row.quantity_required for row in bills.itertuples()}

    setup_time = {(row.product_id, row.machine_id): row.setup_time for row in routing.itertuples()}
    setup_cost = {(row.product_id, row.machine_id): row.setup_cost for row in routing.itertuples()}
    min_batch_size = {(row.product_id, row.machine_id): row.min_batch_size for row in routing.itertuples()}

    available_hours_max = {m: max(available_hours[(m, t)] for t in T) for m in M}
    big_m = {
        (p, m): (available_hours_max[m] + overtime_hours[m]) / processing_time[(p, m)]
        for (p, m) in PM
    }

    return dict(
        P=P, M=M, T=T, R=R, PM=PM, idx_pmt=idx_pmt,
        demand=demand, initial_inventory=initial_inventory, safety_stock=safety_stock,
        max_inventory=max_inventory, production_cost=production_cost, holding_cost=holding_cost,
        shortage_cost=shortage_cost, overtime_cost_per_hour=overtime_cost_per_hour,
        overtime_hours=overtime_hours, available_hours=available_hours,
        processing_time=processing_time, setup_time=setup_time, setup_cost=setup_cost,
        min_batch_size=min_batch_size, material_qty=material_qty,
        material_availability=material_availability, big_m=big_m,
    )


def solve_model_b(params, mip_gap=0.005, time_limit=60, verbose=False):
    """Build and solve Model B (MILP: setup costs, machine allocation, min batch).

    `params` is a dict from load_base_parameters, optionally with some entries
    overridden by the caller. Returns (model, variables).
    """
    p = params
    model = gp.Model("setup_and_allocation")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.MIPGap = mip_gap
    model.Params.TimeLimit = time_limit

    x = model.addVars(p["idx_pmt"], lb=0, name="x")
    y = model.addVars(p["idx_pmt"], vtype=GRB.BINARY, name="y")
    inv = model.addVars(p["P"], p["T"], lb=0, name="inv")
    shortage = model.addVars(p["P"], p["T"], lb=0, name="shortage")
    overtime = model.addVars(p["M"], p["T"], lb=0, name="overtime")
    for m in p["M"]:
        for t in p["T"]:
            overtime[m, t].UB = p["overtime_hours"][m]

    variables = {"x": x, "y": y, "inv": inv, "shortage": shortage, "overtime": overtime}
    _add_constraints(model, p, variables)
    _set_objective(model, p, variables)

    model.optimize()
    return model, variables


def _add_constraints(model, p, v):
    """Same five constraints as Model A plus the two setup-linking ones."""
    P, M, T, R, PM = p["P"], p["M"], p["T"], p["R"], p["PM"]
    x, y, inv, shortage, overtime = v["x"], v["y"], v["inv"], v["shortage"], v["overtime"]

    model.addConstrs(
        (
            (p["initial_inventory"][pr] if t == T[0] else inv[pr, t - 1])
            + gp.quicksum(x[pr, m, t] for m in M if (pr, m) in PM)
            == p["demand"][pr, t] - shortage[pr, t] + inv[pr, t]
            for pr in P for t in T
        ),
        name="inventory_balance",
    )
    model.addConstrs((inv[pr, t] <= p["max_inventory"][pr] for pr in P for t in T), name="max_inventory")
    model.addConstrs((inv[pr, t] >= p["safety_stock"][pr] for pr in P for t in T), name="safety_stock")

    model.addConstrs(
        (
            gp.quicksum(
                p["processing_time"][pr, m] * x[pr, m, t] + p["setup_time"][pr, m] * y[pr, m, t]
                for pr in P if (pr, m) in PM
            )
            <= p["available_hours"][m, t] + overtime[m, t]
            for m in M for t in T
        ),
        name="machine_capacity",
    )
    model.addConstrs(
        (
            gp.quicksum(
                p["material_qty"][pr, r] * gp.quicksum(x[pr, m, t] for m in M if (pr, m) in PM)
                for pr in P if (pr, r) in p["material_qty"]
            )
            <= p["material_availability"][r]
            for r in R for t in T
        ),
        name="material",
    )

    model.addConstrs((x[pr, m, t] <= p["big_m"][pr, m] * y[pr, m, t] for (pr, m, t) in p["idx_pmt"]), name="link_upper")
    model.addConstrs((x[pr, m, t] >= p["min_batch_size"][pr, m] * y[pr, m, t] for (pr, m, t) in p["idx_pmt"]), name="link_lower")


def _set_objective(model, p, v):
    PM, T, P, M = p["PM"], p["T"], p["P"], p["M"]
    x, y, inv, shortage, overtime = v["x"], v["y"], v["inv"], v["shortage"], v["overtime"]

    model.setObjective(
        gp.quicksum(p["production_cost"][pr] * x[pr, m, t] for (pr, m) in PM for t in T)
        + gp.quicksum(p["holding_cost"][pr] * inv[pr, t] for pr in P for t in T)
        + gp.quicksum(p["overtime_cost_per_hour"][m] * overtime[m, t] for m in M for t in T)
        + gp.quicksum(p["shortage_cost"][pr] * shortage[pr, t] for pr in P for t in T)
        + gp.quicksum(p["setup_cost"][pr, m] * y[pr, m, t] for (pr, m) in PM for t in T),
        GRB.MINIMIZE,
    )


def extract_results(model, variables, params):
    """Build result DataFrames from a solved Model B.

    Returns production, inventory, shortage, overtime, setup, cost_breakdown,
    and total_cost.
    """
    x, y, inv, shortage, overtime = (
        variables["x"], variables["y"], variables["inv"], variables["shortage"], variables["overtime"]
    )
    P, M, T, PM = params["P"], params["M"], params["T"], params["PM"]

    production = pd.DataFrame(
        [{"product_id": pr, "machine_id": m, "t": t, "quantity": x[pr, m, t].X} for (pr, m) in PM for t in T]
    )
    production = production[production["quantity"] > 1e-6].round({"quantity": 1})

    inventory = pd.DataFrame(
        [{"product_id": pr, "t": t, "inventory": inv[pr, t].X} for pr in P for t in T]
    ).round({"inventory": 1})

    shortage_df = pd.DataFrame(
        [{"product_id": pr, "t": t, "shortage": shortage[pr, t].X} for pr in P for t in T]
    )
    shortage_df = shortage_df[shortage_df["shortage"] > 1e-6].round({"shortage": 1})

    overtime_df = pd.DataFrame(
        [{"machine_id": m, "t": t, "overtime_hours_used": overtime[m, t].X} for m in M for t in T]
    )
    overtime_df = overtime_df[overtime_df["overtime_hours_used"] > 1e-6].round({"overtime_hours_used": 1})

    setup = pd.DataFrame(
        [{"product_id": pr, "machine_id": m, "t": t, "setup": y[pr, m, t].X} for (pr, m) in PM for t in T]
    )
    setup = setup[setup["setup"] > 0.5].drop(columns="setup").reset_index(drop=True)

    cost = {
        "production": sum(params["production_cost"][pr] * x[pr, m, t].X for (pr, m) in PM for t in T),
        "holding": sum(params["holding_cost"][pr] * inv[pr, t].X for pr in P for t in T),
        "overtime": sum(params["overtime_cost_per_hour"][m] * overtime[m, t].X for m in M for t in T),
        "shortage": sum(params["shortage_cost"][pr] * shortage[pr, t].X for pr in P for t in T),
        "setup": sum(params["setup_cost"][pr, m] * y[pr, m, t].X for (pr, m) in PM for t in T),
    }
    cost_df = pd.Series(cost).rename("cost").to_frame()
    cost_df["share_pct"] = (cost_df["cost"] / cost_df["cost"].sum() * 100).round(1)

    return {
        "production": production,
        "inventory": inventory,
        "shortage": shortage_df,
        "overtime": overtime_df,
        "setup": setup,
        "cost_breakdown": cost_df,
        "total_cost": model.ObjVal,
    }
