# Production Planning Model : Mathematical Formulation

This document covers the two optimization models built in `notebooks/05_production_planning.ipynb` and `notebooks/06_setup_and_allocation.ipynb`: a continuous **LP** (Model A) and its extension into a **MILP** with binary setup/allocation decisions (Model B). Both are implemented with Gurobi's native Python API (`gurobipy`). For each, this documents the math, why it's modeled that way, and the exact `gurobipy` code that implements it.

Both models share the same sets, most of the same parameters, and several constraints verbatim, that overlap is deliberate (see notebook 06's intro): Model B is Model A plus a well-defined set of additions, which is what makes the two directly comparable.

**License note.** `gurobipy` needs a license to solve anything. Nothing was configured for this project, the pip-installed package ships with a bundled *size-limited free* license ("Restricted license, for non-production use only", printed on every run, valid to 2027-11-29), capped at 2000 variables/constraints. Model B is 792 columns / 1068 rows,
comfortably inside that cap, so it runs with zero setup. A real deployment would need a paid or academic license.

---

## 1. Sets and indices

| Symbol | Meaning | Size | Source |
|---|---|---|---|
| $p \in P$ | product | 10 | `products.csv` |
| $m \in M$ | machine | 6 | `machines.csv` |
| $t \in T = \{1, \dots, 12\}$ | planning week (relative index, not calendar week) | 12 | `machine_availability.csv`, reconciled against `forecast.csv`'s absolute weeks 274–285 |
| $r \in R$ | raw material | 13 | `materials.csv` |
| $(p,m) \in PM \subseteq P \times M$ | valid product/machine routing pair | 20 | `routing.csv` |

`PM` is not the full $P \times M$ cross product, each product only has a primary and a secondary machine, most pairs are invalid. Every variable and constraint below that's indexed by $(p,m)$ is restricted to `PM`, not all of $P \times M$.

```python
P = products["product_id"].tolist()
M = machines["machine_id"].tolist()
T = sorted(forecast["t"].unique().tolist())
R = materials["material_id"].tolist()
PM = list(routing[["product_id", "machine_id"]].itertuples(index=False, name=None))
idx_pmt = [(p, m, t) for (p, m) in PM for t in T]   # explicit (p,m,t) index list used for every x/y variable
```

## 2. Parameters

| Symbol | Meaning | Variable name | Real or assumed |
|---|---|---|---|
| $d_{p,t}$ | forecast demand | `demand[p, t]` | derived (ML forecast, notebook 04) |
| $c^{prod}_p$ | production cost / unit | `production_cost[p]` | derived from real `selling_price` |
| $c^{hold}_p$ | holding cost / unit / week | `holding_cost[p]` | assumed formula |
| $c^{short}_p$ | shortage penalty / unit | `shortage_cost[p]` | assumed formula |
| $I^{max}_p$ | max inventory (storage cap) | `max_inventory[p]` | assumed formula |
| $I^{safety}_p$ | safety stock target | `safety_stock[p]` | assumed formula |
| $I^{0}_p$ | initial inventory (week 0) | `initial_inventory[p]` | assumed = `safety_stock[p]` |
| $\tau_{p,m}$ | processing time (h/unit) | `processing_time[p, m]` | assumed (notebook 03) |
| $H_{m,t}$ | **effective** available hours | `available_hours[m, t]` | `regular_hours` × `availability` fraction |
| $O_m$ | max overtime hours | `overtime_hours[m]` | assumed |
| $c^{ot}_m$ | overtime cost / hour | `overtime_cost_per_hour[m]` | assumed = wage × `labor_requirement[m]` |
| $q_{p,r}$ | material use / unit | `material_qty[p, r]` | assumed (bill of materials) |
| $A_r$ | material availability / week | `material_availability[r]` | assumed |
| $\sigma_{p,m}$ | setup time (h) : **MILP only** | `setup_time[p, m]` | assumed (notebook 03), unused in the LP |
| $\kappa_{p,m}$ | setup cost ($) : **MILP only** | `setup_cost[p, m]` | assumed, unused in the LP |
| $b_{p,m}$ | min batch size : **MILP only** | `min_batch_size[p, m]` | assumed, unused in the LP |

$H_{m,t}$ deserves a note: `machine_availability.csv` stores raw `regular_hours`, and
`machines.csv`'s `availability` column (0.90–0.97) is applied at model-build time, not baked into the CSV:

```python
availability_fraction = machines.set_index("machine_id")["availability"].to_dict()
available_hours = {
    (row.machine_id, row.t): row.available_hours * availability_fraction[row.machine_id]
    for row in availability.itertuples()
}
```

---

## 3. Model A : continuous production planning (LP)

### 3.1 Decision variables

| Symbol | Domain | Meaning |
|---|---|---|
| $x_{p,m,t}$ | $\ge 0$, real | production quantity of $p$ on $m$ in week $t$, only for $(p,m)\in PM$ |
| $I_{p,t}$ | $\ge 0$, real | ending inventory of $p$ after week $t$ |
| $s_{p,t}$ | $\ge 0$, real | shortage (unmet demand) of $p$ in week $t$ |
| $OT_{m,t}$ | $0 \le OT_{m,t} \le O_m$ | overtime hours used on $m$ in week $t$ |

Every variable here is continuous, this is what makes Model A a pure **Linear Program**, solvable essentially instantly regardless of size. `gurobipy`'s `Model.addVars` returns a `tupledict` keyed by whatever index tuples you give it; the per-machine overtime bound is set afterward via each variable's `.UB` attribute since it depends on `m`:

```python
model = gp.Model("production_planning_lp")

x = model.addVars(idx_pmt, lb=0, name="x")
inv = model.addVars(P, T, lb=0, name="inv")
shortage = model.addVars(P, T, lb=0, name="shortage")
overtime = model.addVars(M, T, lb=0, name="overtime")
for m in M:
    for t in T:
        overtime[m, t].UB = overtime_hours[m]
```

### 3.2 Objective

$$
\min \; \underbrace{\sum_{(p,m)\in PM}\sum_{t\in T} c^{prod}_p\, x_{p,m,t}}_{\text{production}}
\;+\; \underbrace{\sum_{p\in P}\sum_{t\in T} c^{hold}_p\, I_{p,t}}_{\text{holding}}
\;+\; \underbrace{\sum_{m\in M}\sum_{t\in T} c^{ot}_m\, OT_{m,t}}_{\text{overtime}}
\;+\; \underbrace{\sum_{p\in P}\sum_{t\in T} c^{short}_p\, s_{p,t}}_{\text{shortage}}
$$

`gurobipy` sets the objective directly with `Model.setObjective`, no separate "define, then
attach" step:

```python
model.setObjective(
    gp.quicksum(production_cost[p] * x[p, m, t] for (p, m) in PM for t in T)
    + gp.quicksum(holding_cost[p] * inv[p, t] for p in P for t in T)
    + gp.quicksum(overtime_cost_per_hour[m] * overtime[m, t] for m in M for t in T)
    + gp.quicksum(shortage_cost[p] * shortage[p, t] for p in P for t in T),
    GRB.MINIMIZE,
)
```

No setup cost term, that's the entire point of Model A existing separately from Model B (§4).

### 3.3 Constraints

**Inventory balance.** What came in (starting inventory + production) must equal what went out (demand, net of any shortage) plus what's left over:

$$
I_{p,t-1} + \sum_{m:(p,m)\in PM} x_{p,m,t} \;=\; d_{p,t} - s_{p,t} + I_{p,t}
\qquad \forall p\in P,\; t\in T
$$

with $I_{p,0} := I^{0}_p$ (initial inventory) for the first period. `Model.addConstrs` takes a generator expression directly, no separate rule function needed:

```python
model.addConstrs(
    (
        (initial_inventory[p] if t == T[0] else inv[p, t - 1])
        + gp.quicksum(x[p, m, t] for m in M if (p, m) in PM)
        == demand[p, t] - shortage[p, t] + inv[p, t]
        for p in P for t in T
    ),
    name="inventory_balance",
)
```

This is the constraint whose telescoping sum over all of $T$ is what the notebooks' "accounting identity" sanity check verifies: $I^0_p + \sum_t\sum_m x_{p,m,t} = \sum_t d_{p,t} - \sum_t s_{p,t} + I_{p,|T|}$.

**Storage cap:**
$$I_{p,t} \le I^{max}_p \qquad \forall p,t$$

```python
model.addConstrs((inv[p, t] <= max_inventory[p] for p in P for t in T), name="max_inventory")
```

**Safety stock:**
$$I_{p,t} \ge I^{safety}_p \qquad \forall p,t$$

```python
model.addConstrs((inv[p, t] >= safety_stock[p] for p in P for t in T), name="safety_stock")
```


**Machine capacity**, hours consumed by processing can't exceed regular + overtime hours:
$$
\sum_{p:(p,m)\in PM} \tau_{p,m}\, x_{p,m,t} \;\le\; H_{m,t} + OT_{m,t} \qquad \forall m,t
$$

```python
model.addConstrs(
    (
        gp.quicksum(processing_time[p, m] * x[p, m, t] for p in P if (p, m) in PM)
        <= available_hours[m, t] + overtime[m, t]
        for m in M for t in T
    ),
    name="machine_capacity",
)
```

**Material availability** : no inventory/carryover modeled for raw materials, each week's supply
is independent:
$$
\sum_{p \,:\, (p,r)\ \text{defined}} q_{p,r} \sum_{m:(p,m)\in PM} x_{p,m,t} \;\le\; A_r
\qquad \forall r,t
$$

```python
model.addConstrs(
    (
        gp.quicksum(
            material_qty[p, r] * gp.quicksum(x[p, m, t] for m in M if (p, m) in PM)
            for p in P if (p, r) in material_qty
        )
        <= material_availability[r]
        for r in R for t in T
    ),
    name="material",
)
```

### 3.4 Solve

```python
model.Params.OutputFlag = 0     # suppress Gurobi's solver log
model.optimize()

print("status:", model.Status)          # 2 == GRB.OPTIMAL
print("total cost:", model.ObjVal)
```

LPs of this size solve to certified global optimality essentially instantly, 0.008s in this run, no gap tolerance or time limit needed (unlike Model B, §4.4). Reading a solved value back is a plain attribute on the variable object: `x[p, m, t].X`, not a function call, the one thing easiest to get wrong when translating from other modeling APIs.

### 3.5 Baseline, a non-optimized reference point

Notebook 05 also builds a simple **rule-based baseline** alongside Model A, not a math program at
all, a plain simulation loop: for each product, every week, produce enough on its single fixed
primary machine to cover demand and top back up to `safety_stock`, using overtime only if regular
hours run out, no reallocation to the secondary machine, no cost minimization. It exists purely
as "what a factory following a simple, common-sense rule would do", the point of comparison that
shows what optimization is actually worth (see §5).

```python
for t in T:
    hours_left = {m: available_hours[m, t] for m in M}
    ot_left = {m: overtime_hours[m] for m in M}
    for p in P:
        m = primary_machine[p]          # fixed, lowest-setup-cost machine, no alternative used
        rate = processing_time[p, m]
        target = max(demand[p, t] + safety_stock[p] - baseline_inv[p], 0)
        hours_needed = target * rate

        reg = min(hours_needed, hours_left[m])
        ot = min(hours_needed - reg, ot_left[m])
        produced = (reg + ot) / rate    # whatever's left short of this becomes shortage
```

---

## 4. Model B : setup costs, machine allocation, minimum batch (MILP)

Model B keeps every variable, constraint, and cost term from Model A, and adds exactly four things: one binary variable, one modified constraint, two new constraints, and one added objective term.

### 4.1 New decision variable

$$y_{p,m,t} \in \{0, 1\} \qquad \forall (p,m)\in PM,\; t\in T$$

$y_{p,m,t}=1$ means "product $p$ is run on machine $m$ during week $t$", i.e. a changeover to $p$ happens on $m$ that week, paying its setup time/cost. This single binary is what turns the problem from an LP into a genuine **Mixed-Integer Linear Program**.

```python
y = model.addVars(idx_pmt, vtype=GRB.BINARY, name="y")
```

### 4.2 Modified objective : add the setup cost term

$$
\min \;\Big(\text{production} + \text{holding} + \text{overtime} + \text{shortage}\Big)
\;+\; \underbrace{\sum_{(p,m)\in PM}\sum_{t\in T} \kappa_{p,m}\, y_{p,m,t}}_{\text{setup}}
$$

```python
model.setObjective(
    gp.quicksum(production_cost[p] * x[p, m, t] for (p, m) in PM for t in T)
    + gp.quicksum(holding_cost[p] * inv[p, t] for p in P for t in T)
    + gp.quicksum(overtime_cost_per_hour[m] * overtime[m, t] for m in M for t in T)
    + gp.quicksum(shortage_cost[p] * shortage[p, t] for p in P for t in T)
    + gp.quicksum(setup_cost[p, m] * y[p, m, t] for (p, m) in PM for t in T),
    GRB.MINIMIZE,
)
```

This is the term that finally makes primary vs. secondary machine choice *matter* economically, in Model A, `production_cost` doesn't vary by machine and nothing else in the objective distinguishes routing options, so machine assignment was cost-neutral (see §5's LP-degeneracy note for a concrete demonstration of exactly this).

### 4.3 Modified/new constraints

**Machine capacity : now setup time competes for the same hours as processing time:**

$$
\sum_{p:(p,m)\in PM} \Big[\tau_{p,m}\, x_{p,m,t} + \sigma_{p,m}\, y_{p,m,t}\Big] \;\le\; H_{m,t} + OT_{m,t}
\qquad \forall m,t
$$

```python
model.addConstrs(
    (
        gp.quicksum(
            processing_time[p, m] * x[p, m, t] + setup_time[p, m] * y[p, m, t]
            for p in P if (p, m) in PM
        )
        <= available_hours[m, t] + overtime[m, t]
        for m in M for t in T
    ),
    name="machine_capacity",
)
```

**Linking constraints, the "Big-M" fixed-charge pattern.** $x$ and $y$ need to be tied together: producing anything ($x>0$) should force the setup indicator on ($y=1$), and if a setup happens at all it should be for a real batch, not a token unit. Two inequalities do this jointly:

$$
x_{p,m,t} \le BigM_{p,m}\, y_{p,m,t} \qquad \text{(can't produce without a setup)}
$$
$$
x_{p,m,t} \ge b_{p,m}\, y_{p,m,t} \qquad \text{(if set up, must hit the minimum batch)}
$$

Together: if $y_{p,m,t}=0$, both force $x_{p,m,t}=0$. If $y_{p,m,t}=1$, $x_{p,m,t}$ is free to be anywhere in $[\,b_{p,m},\, BigM_{p,m}\,]$, the optimizer decides how much, but not whether to switch it on. $BigM_{p,m}$ must be large enough to never bind when $y=1$, but as *tight* as possible, a looser BigM weakens the LP relaxation, which slows branch-and-bound convergence (see §4.4). It's computed per pair as the most that machine could physically produce in its best week, fully loaded with overtime:

$$
BigM_{p,m} = \frac{\max_t H_{m,t} + O_m}{\tau_{p,m}}
$$

```python
available_hours_max = {m: max(available_hours[(m, t)] for t in T) for m in M}
BigM = {
    (p, m): (available_hours_max[m] + overtime_hours[m]) / processing_time[(p, m)]
    for (p, m) in PM
}

model.addConstrs((x[p, m, t] <= BigM[p, m] * y[p, m, t] for (p, m, t) in idx_pmt), name="link_upper")
model.addConstrs((x[p, m, t] >= min_batch_size[p, m] * y[p, m, t] for (p, m, t) in idx_pmt), name="link_lower")
```

Every other constraint (inventory balance, max inventory, safety stock, material) is inherited unchanged from Model A.

### 4.4 Solve : MILPs need a gap tolerance

```python
model.Params.MIPGap = 0.005      # accept within 0.5% of certified optimal
model.Params.TimeLimit = 60      # hard safety net
model.optimize()
```

With 240 binaries this is a real combinatorial problem, not a trivial LP. Gurobi typically finds a solution within a fraction of a percent of optimal in well under a second, but with default (near-zero) gap tolerance can spend much longer chasing the last fraction of a percent of *certified* optimality, because the BigM linking constraints create a comparatively weak LP relaxation for branch-and-bound to prune against. Setting a **0.5% relative gap tolerance** plus a **60s time limit** as a hard safety net is standard MILP practice, not a shortcut being hidden, the achieved gap and solve time are themselves meaningful results (they're exactly the "solver
time" / "optimality gap" computational-performance metrics the project spec asks to report). In this run: `optimal` (within tolerance) in **1.94s**, at a **0.496%** gap.

---

## 5. What changes between baseline, Model A, and Model B

`machines.csv` deliberately cuts `regular_hours` on M01, M03, and M04 (see notebook 03) so
capacity actually binds at peak demand instead of sitting comfortably slack everywhere. That's
what makes the three approaches below actually diverge, on the original, looser capacity figures
the baseline and Model A produced numerically identical costs, there was no hard tradeoff for
the LP to solve that a simple heuristic didn't already get right.

| | Baseline (heuristic) | Model A (LP) | Model B (MILP) |
|---|---|---|---|
| Machine assignment | fixed primary only | free across `PM` | free across `PM` |
| Setup cost modeled | no | no | yes |
| Total cost | \$100,978.64 | \$72,878.47 | \$84,286.66 |
| Solve time | instant (simulation) | 0.008s (exact) | 1.94s (0.496% gap) |

### 5.1 Cost breakdown, all three

| | Baseline | Model A | Model B |
|---|---|---|---|
| Production | \$72,488.50 (71.8%) | \$72,488.50 (99.5%) | \$71,884.92 (85.3%) |
| Holding | \$389.96 (0.4%) | \$389.96 (0.5%) | \$713.64 (0.8%) |
| Overtime | \$28,100.17 (27.8%) | \$0.00 (0%) | \$0.00 (0%) |
| Shortage | \$0.00 (0%) | \$0.00 (0%) | \$1,218.10 (1.4%) |
| Setup | n/a | \$0.00 (0%) | \$10,470.00 (12.4%) |
| **Total** | **\$100,978.64** | **\$72,878.47** | **\$84,286.66** |

### 5.2 Headline finding: optimization is worth 27.8% here, and it comes entirely from flexibility

Baseline and Model A produce and hold the exact same amount, they differ on exactly one thing:
the baseline is stuck on each product's single fixed machine, so once M01/M03/M04 run short on
regular hours it has no choice but to pay for \$28,100 of overtime. Model A can shift the same
work onto the slack M02/M05/M06 instead, and pays **zero** overtime for it, a **27.8% total cost
reduction** for making the exact same products in the exact same quantities, purely from where
they're allowed to be made.

Model B costing more than Model A (\$84,287 vs \$72,878) is not a regression, it's Model B
being charged for two real costs Model A's objective doesn't include at all: \$10,470 in setup
costs and \$1,218 in shortage. Under this tighter capacity, machine choice also stopped being
perfectly clean: only **87 of 106 setups (82%)** land on a product's primary (cheapest-setup)
machine, down from 100% under the looser capacity figures, sometimes the primary machine is
simply full, and paying a secondary machine's higher setup cost is cheaper than the alternative.
That's a more realistic result than a clean 100%, and it's a direct, visible consequence of
tightening capacity rather than an artifact of the model.

---

## 6. Quick reference : building a gurobipy model, in order

1. `gp.Model(name)`
2. `model.addVars(<indices>, lb=..., ub=..., vtype=...)` for every decision variable, index with an explicit list of tuples (e.g. `idx_pmt`) when mixing a routing-pair set with a time set, rather than relying on the API to flatten a nested cross product. Set any bound that depends on the index (like the per-machine overtime cap) afterward via `.UB`/`.LB` on each variable.
3. Constraints directly as `model.addConstrs(<generator expression>, name=...)`, no separate rule function needed; the constraint is defined and attached in one call
4. `model.setObjective(<expression>, GRB.MINIMIZE)`
5. Set any solver parameters needed (`model.Params.MIPGap`, `model.Params.TimeLimit`,
   `model.Params.OutputFlag`, …) before solving
6. `model.optimize()`
7. Extract results via `.X` on each variable object (e.g. `x[p, m, t].X`), and `.ObjVal` / `.MIPGap` / `.Status` on the model itself (`2` == `GRB.OPTIMAL`).

A pitfall worth flagging explicitly, since it's exactly what broke this project's first pass at
converting from Pyomo: variables created with `model.addVars(...)` are **plain Python names** (e.g. `x`, `inv`), not attributes of the model object, there is no `model.x`. Every downstream results-extraction and plotting cell must reference the variable object directly (`x[p, m, t].X`), not `model.x[p, m, t]`.
