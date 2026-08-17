| # | Feature | Group | Source / rationale |
|---:|---|---|---|
| 1 | `queue_length` | queue | Standard congestion state; de Koster et al. (2007). |
| 2 | `mean_queue_age` | queue | Waiting-time proxy; distinguishes fresh from stale backlog. |
| 3 | `max_queue_age` | queue | Tail of the waiting-time distribution — starvation detector. |
| 4 | `pct_critical` | queue | Share of queue within 30 min of its due time. |
| 5 | `pct_perishable` | queue | Gates the FEFO mask; required for the expiry-aware rule. |
| 6 | `n_arrivals_last_interval` | queue | Short-run demand shock; wave arrivals, Boysen et al. (2019). |
| 7 | `labor_utilization` | resources | Classical queueing load indicator rho. |
| 8 | `n_pickers_busy` | resources | Absolute capacity remaining this epoch. |
| 9 | `mean_pickup_time_recent` | resources | Realised service-rate estimate; drifts with order mix. |
| 10 | `n_orders_late_so_far` | deadline | Realised failures to date; regime indicator. |
| 11 | `n_orders_at_risk_30min` | deadline | Count with negative slack inside 30 min — the actionable set. |
| 12 | `mean_slack_minutes` | deadline | Mean d - t - p; the ATC/MS/COVERT decision variable. |
| 13 | `std_slack_minutes` | deadline | Slack dispersion; separates uniform from bimodal pressure. |
| 14 | `mean_processing_time_remaining` | deadline | Expected work content of the queue. |
| 15 | `pct_high_priority` | deadline | Share of the economically weighted tail. |
| 16 | `pct_expiring_30min` | expiry | Product-clock analogue of pct_critical (revision). |
| 17 | `mean_expiry_slack` | expiry | Mean x - t - p over perishables; FEFO's decision variable (revision). |
| 18 | `n_spoiled_so_far` | expiry | Realised spoilage to date (revision). |
| 19 | `arrival_rate_recent_60min` | arrivals | Non-stationarity detector; empirical lambda-hat. |
| 20 | `queue_length_lag_1` | history | Congestion trend; standard lag structure. |
| 21 | `queue_length_lag_2` | history | Congestion trend. |
| 22 | `queue_length_lag_3` | history | Congestion trend. |
| 23 | `failure_rate_lag_1` | history | Failure trend; replaces breach_rate lag under the new metric. |
| 24 | `failure_rate_lag_2` | history | Failure trend. |
| 25 | `failure_rate_lag_3` | history | Failure trend. |
| 26 | `interval_index_in_shift` | temporal | Position in the finite horizon; end-of-shift effects. |