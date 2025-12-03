# flex-depot

## Asymmetric Horizon Model Predictive Control for HDV-Flexibility Trading at Logistics Depot

This document explains how the MPC system jointly optimizes **Day-Ahead
(DA)** and **Intraday (ID)** markets using a single optimization model
with different effective horizons. It is intended as a clear onboarding
reference for developers and operators.

------------------------------------------------------------------------

## 1. Motivation

Battery trading must simultaneously respect two different electricity
markets:

  Market               Horizon         Gate Closure (GC)   Products
  -------------------- --------------- ------------------- ----------
  **Day-Ahead (DA)**   Full next day   D-1 12:00           15-min
  **Intraday (ID)**    Short-term      T -- 30 min         15-min

Running two separate optimizations (DA first, then ID) leads to
inconsistencies and duplicated logic.

**Our MPC handles both markets simultaneously in one unified model.**

------------------------------------------------------------------------

## 2. Concept: Asymmetric Horizon MPC

The MPC uses **one common 15-minute time index** but applies different
effective horizons:

    ┌─────────────────────────────┬─────────────────────────────┐
    │     DA Optimization Window  │   Entire DA Delivery Day     │
    └─────────────────────────────┴─────────────────────────────┘
    <---------------------- ~48 hours -------------------------->

    ┌─────────┬─────────────────────────┬────────────────────────┐
    │  now    │   ID Optimization       │   ID Delivery Window   │
    └─────────┴─────────────────────────┴────────────────────────┘
              <------ ~12 hours ------->

Key points:

-   Same Pyomo model for battery + markets\
-   Same 15-minute timestep\
-   DA visible & optimized across full \~48h\
-   ID only visible in next \~12h\
-   Gate closure rules remain independent\
-   Physical battery must fulfill committed DA trades next day\
-   ID cannot overwrite DA after its gate closure

------------------------------------------------------------------------

## 3. Architecture Overview

### One Model

-   Single Pyomo model instance
-   Shared variables:
    -   Battery: `SoC`, `p_batt_pos`, `p_batt_neg`
    -   Markets: `p_market_opt`, `p_market_committed`

### One Unified Time Index

-   15-minute resolution\
-   Length ≈ DA horizon (36--48 hours)

### Two Market Masks

-   `mask_DA_active[t]`
-   `mask_ID_active[t]` (includes ID horizon limit)

### Prices

-   DA prices are loaded for full horizon\
-   ID prices limited to short horizon\
-   Both aligned to unified timestep

------------------------------------------------------------------------

## 4. Gate Closure & Commit Logic

### Day-Ahead (DA)

-   Market open until **D-1 12:00**
-   When GC is crossed between `current_time` and `next_time`:


    p_market_committed_DA[t] = p_market_opt[t]

-   After commit, DA positions are **fixed**\
-   ID is never allowed to override DA commitments

### Intraday (ID)

-   GC: **T -- 30 minutes**
-   Slot-wise commit when GC is reached:


    p_market_committed_ID[t] = p_market_opt[t]

-   Each 15-minute product becomes frozen independently

------------------------------------------------------------------------

## 5. What the MPC Does in Each Loop

Every 15 minutes:

1.  **Build the 48h DA-scope window**\
2.  Apply DA and ID masks\
3.  Insert committed positions (DA + ID)\
4.  Optimize all remaining free variables\
5.  Commit new DA or ID slots whose GC has passed\
6.  Export:
    -   `dispatch_mpc.csv` (realized SoC, flows, activities)
    -   `commit_mpc.csv` (all market commitments)

------------------------------------------------------------------------

## 6. Key Properties of the System

-   DA commitments propagate into all future MPC iterations\
-   ID trades are short-term and cannot modify DA products\
-   Battery SoC trajectory covers the entire 48h horizon\
-   Resulting dispatch is physically and market-accurate\
-   No double models, no DA-first-then-ID structure\
-   Fully aligned with real European market mechanisms

------------------------------------------------------------------------

## 7. Files of Interest

-   `model.py` --- Pyomo model definition\
-   `mpc_workflow.py` --- Rolling MPC loop, window construction, commit
    logic\
-   `trading_rules.py` --- Gate closure and market masks\
-   `settings.yaml` --- Configuration (DA/ID horizons, fees, prices)

------------------------------------------------------------------------

## 8. Summary

This asymmetric MPC architecture mirrors real-world trading for battery/EV assets:

-   **Unified model**
-   **Multiple markets**
-   **Independent gate closures**
-   **Different visibility horizons**
-   **Single rolling 48h optimization window**

