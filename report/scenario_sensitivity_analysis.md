# Scenario & Sensitivity Analysis, Mathematical Formulation

This document covers `notebooks/07_scenario_analysis.ipynb` and
`notebooks/08_sensitivity_analysis.ipynb`, and the reusable module behind both,
`src/optimization/model_b.py`.

---

## 1. The reusable module

Notebook 06 each built Model B from scratch. Scenario and
sensitivity analysis need to solve the *same* model dozens of times with different inputs,
repeating 20 constraint cells per run would be unreadable. So the model itself moved to
`src/optimization/model_b.py`, exposing three functions:

```python
load_base_parameters(overtime_wage=35) -> dict     # loads CSVs, builds every parameter dict
solve_model_b(params, mip_gap=0.005, time_limit=60) -> (model, variables)
extract_results(model, variables, params) -> dict of DataFrames
```

`load_base_parameters` returns a plain Python `dict` (`demand`, `available_hours`,
`material_availability`, `holding_cost`, ...), not a bound object, specifically so a notebook
can copy it and override one entry:

```python
params = load_base_parameters()
p2 = dict(params)
p2["demand"] = {k: v * 1.2 for k, v in params["demand"].items()}   # +20% demand scenario
model, variables = solve_model_b(p2)
```

Verified to reproduce notebook 06's result exactly: **$84,286.66**, same MIP gap (0.496%),
same 96/106 setup pattern, confirming the refactor changed nothing about the model itself.

---

## 2. Scenario analysis (`07_scenario_analysis.ipynb`)

Three shocks, each just a different `params` dict passed into the same `solve_model_b`. No new
constraints, no new variables, only the right-hand side of an existing constraint or a demand
value changes.

### 2.1 Demand increase

$$d'_{p,t} = (1 + \alpha)\, d_{p,t} \qquad \alpha \in \{0.10, 0.20, 0.30\}$$

```python
def scale_demand(params, factor):
    new_params = dict(params)
    new_params["demand"] = {k: v * factor for k, v in params["demand"].items()}
    return new_params
```

| Scenario | Total cost | Shortage cost | Shortage rows |
|---|---|---|---|
| base | $84,286.66 | $1,218.10 | 13 |
| +10% | $93,389.36 | $3,508.45 | 49 |
| +20% | $102,710.18 | $7,325.71 | 56 |
| +30% | $115,566.47 | $18,124.30 | 87 |

Shortage cost grows faster than demand itself (10%→3x shortage cost, 30%→~15x), the factory's
slack gets used up by the smaller increases, so each further percent of demand growth has to be
absorbed by the increasingly scarce remaining flexibility rather than spread evenly.

### 2.2 Machine failure

$$H'_{M01,t} = 0 \qquad t \in \{6, 7\}$$

A full two-week outage on M01, the busiest machine (see `model_formulation.md` §5's shadow
price finding, this is also the machine with by far the largest capacity shadow price).

```python
def fail_machine(params, machine_id, weeks, factor=0.0):
    new_params = dict(params)
    new_hours = dict(params["available_hours"])
    for t in weeks:
        new_hours[(machine_id, t)] *= factor
    new_params["available_hours"] = new_hours
    return new_params
```

Result: **$84,558.96** (+0.3% vs base), shortage rises from 13 to 14 rows. The MILP absorbs a
full breakdown on its tightest machine with almost no added cost, by reallocating to M02 (its
slack partner), confirming the routing flexibility built into `routing.csv` is doing real work.
The remaining shortage is concentrated in the outage weeks themselves (t=6, t=7), some demand
simply cannot be rerouted in time even with full flexibility.

### 2.3 Material shortage

$$A'_r = 0.5\, A_r$$

Two materials, same 50% cut, deliberately chosen to contrast: `SHR01` (shared by 6
HOUSEHOLD/HOBBIES products) vs `SHR02` (shared only by P01 and P02, the two highest-value
products).

```python
def cut_material(params, material_id, factor):
    new_params = dict(params)
    new_avail = dict(params["material_availability"])
    new_avail[material_id] *= factor
    new_params["material_availability"] = new_avail
    return new_params
```

| Material | Products affected | Total cost | Shortage rows |
|---|---|---|---|
| `SHR01` -50% | 6 (lower-value) | $84,473.81 | 25 |
| `SHR02` -50% | 2 (P01, P02) | $100,288.21 | 34 |

Same percentage cut, radically different impact ($187 vs $16,000 added cost). `SHR01` losing
half its supply barely moves total cost, `SHR02` losing half its supply adds nearly 19% to it.
The lesson isn't "more products depending on a material means more risk", it's the opposite
here: concentration risk in a *high-value* product matters far more than breadth. A real supply
chain would prioritize securing `SHR02`'s supply over `SHR01`'s, despite `SHR01` nominally
serving three times as many SKUs.

---

## 3. Sensitivity analysis (`08_sensitivity_analysis.ipynb`)

Two different tools for two different questions.

### 3.1 Shadow prices, exact, one solve

Standard LP duality doesn't directly apply to a MILP, once integer variables are fixed at
their optimal values, the remaining problem is a genuine LP, and *that* LP has valid shadow
prices. `Model.fixed()` does exactly this: returns a continuous copy of the model with every
binary $y_{p,m,t}$ fixed at its solved 0/1 value.

```python
fixed_model = base_model.fixed()
fixed_model.optimize()
# fixed_model.ObjVal == base_model.ObjVal, same plan, now solvable as an LP
```

For a `<=` constraint in a minimization problem, the shadow price $\pi$ (`Constr.Pi`) is the
exact marginal value of relaxing that constraint's right-hand side by one unit:

$$\pi_{m,t} = \frac{\partial \, (\text{total cost})}{\partial \, H_{m,t}}$$

Top binding machine capacity constraints, this planning horizon:

| Machine, week | Shadow price ($/hour) |
|---|---|
| M01, t=5 | -60.50 |
| M01, t=4 | -59.30 |
| M01, t=3 | -58.10 |
| M01, t=2 | -56.90 |
| M03, t=4 | -9.48 |

Reading this: one more hour of M01 capacity in week 5 would lower total cost by about $60.50,
*exactly*, no re-solve required. M01 dominates every other machine's shadow price by roughly
6-7x, it's unambiguously the highest-value place to invest in extra capacity, which lines up
directly with §2.2's finding that M01 is also the machine whose failure the plan struggles
hardest to route around.

### 3.2 Objective coefficient ranging

Alongside shadow prices, the fixed LP also gives `Var.SAObjLow` / `Var.SAObjUp`, the range each
product's `production_cost` could move within before the *current* plan would stop being
optimal.

```python
x_vars = [v for v in fixed_model.getVars() if v.VarName.startswith("x[") and v.X > 1e-6]
# v.SAObjLow, v.SAObjUp give the valid range for that variable's cost coefficient
```

| Product | production_cost | valid range |
|---|---|---|
| P01 | $1.19 | [1.190, 1.190], zero slack |
| P05 | $0.73 | [0.727, 0.730] |
| P08 | $0.36 | [-0.188, 0.628] |
| P04 | $7.48 | [-inf, 7.480] |

P01's range is degenerate, zero width, meaning *any* change to its production cost, in either
direction, would change the optimal plan. That's a direct consequence of P01 being routed
across two heavily-constrained machines (M01, M02) simultaneously, there's no slack left in
how it's produced for a small cost change to leave untouched. P08, by contrast, has real room
($-0.19 to $0.63) before its routing would need to change.

### 3.3 Grid sweep, for swings beyond local ranging

Ranging only describes the neighborhood around the current basis, it doesn't say what happens
40% away. For genuine "what if this cost changed by X%" questions, the honest answer is to
re-solve.

```python
def scale_dict(d, factor):
    return {k: v * factor for k, v in d.items()}

p2 = dict(base_params)
p2["setup_cost"] = scale_dict(base_params["setup_cost"], 1.2)
cost = solve_model_b(p2)[0].ObjVal
```

Four parameters, -20% / +20% each:

| Parameter | -20% | +20% | swing |
|---|---|---|---|
| `overtime_wage` | -$44.83 | +$10.39 | $55.22 |
| `holding_cost` | -$235.39 | +$79.07 | $314.46 |
| `shortage_cost` | -$1,126.77 | +$69.30 | $1,196.07 |
| `setup_cost` | -$2,166.46 | +$1,818.82 | **$3,985.28** |

`setup_cost` dominates, by roughly 3x the next largest swing. That's a direct consequence of
§2's finding: the tightened capacity means machine choice, and therefore setups, is doing real
economic work now, so its cost assumption is the one this whole plan is most sensitive to.
`overtime_wage` barely moves total cost at all, because the base solution barely uses overtime
in the first place ($0.00, see `model_formulation.md` §5.1), a parameter can only be sensitive
if the model actually leans on it.

---

## 4. Summary

Both notebooks lean on the same tightened-capacity factory from `model_formulation.md` §5, that
tightness is precisely what makes every result here non-trivial: shadow prices are large and
concentrated on M01, machine failure is absorbable but not free, material shortage risk is
wildly asymmetric by product value, and setup cost is the single parameter this plan is most
exposed to. None of these findings would have shown up under the original, looser capacity
figures, where baseline and Model A produced numerically identical results (see
`model_formulation.md` §5's original note on that).
