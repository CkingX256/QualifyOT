# Theory-validation experiment report

Retention experiment: two-state simplex MAE, grid {0,0.1,...,1}, population Monte Carlo n=1,000,000. The population grid optimum was lambda*=0.6; reference MAE=0.174287, candidate MAE=0.121933, retained optimum MAE=0.058545, nearest-grid risk gap=0.005149. Selection used 10,000 independent development datasets per sample size.

Retention selection results:

|   n_development |   replicates |   p_select_population_grid_optimum |   mean_selected_lambda |   p_select_nonzero_lambda |   mean_population_excess_mae |   q95_population_excess_mae |
|----------------:|-------------:|-----------------------------------:|-----------------------:|--------------------------:|-----------------------------:|----------------------------:|
|              10 |        10000 |                             0.4417 |                0.6029  |                         1 |                  0.00478035  |                  0.0198913  |
|              20 |        10000 |                             0.5942 |                0.60443 |                         1 |                  0.00251716  |                  0.00618874 |
|              40 |        10000 |                             0.7659 |                0.60408 |                         1 |                  0.00132225  |                  0.00618874 |
|              80 |        10000 |                             0.9037 |                0.60267 |                         1 |                  0.000532016 |                  0.00514875 |
|             160 |        10000 |                             0.9806 |                0.60108 |                         1 |                  0.000104358 |                  0          |
|             320 |        10000 |                             0.9992 |                0.60006 |                         1 |                  4.223e-06   |                  0          |

Honest-confirmation planning experiment: alpha=0.05, target overall power >= 0.80, three necessary axes each with mean theta=0.15 and score width w=0.4. The sufficient Hoeffding/union-bound formula gives n >= 41. At n=41, empirical qualification power was 0.98249. Under a least-favourable one-boundary-axis null, the empirical false-Qualification rate at n=41 was 0.00561.

Power results:

|   n_confirmation |   replicates |   effect_each_axis |   interval_width |   empirical_qualification_power |   planned_n_bound |
|-----------------:|-------------:|-------------------:|-----------------:|--------------------------------:|------------------:|
|               10 |       100000 |               0.15 |              0.4 |                         0.0536  |                41 |
|               15 |       100000 |               0.15 |              0.4 |                         0.3377  |                41 |
|               20 |       100000 |               0.15 |              0.4 |                         0.65603 |                41 |
|               25 |       100000 |               0.15 |              0.4 |                         0.69324 |                41 |
|               30 |       100000 |               0.15 |              0.4 |                         0.85839 |                41 |
|               35 |       100000 |               0.15 |              0.4 |                         0.94114 |                41 |
|               41 |       100000 |               0.15 |              0.4 |                         0.98249 |                41 |
|               50 |       100000 |               0.15 |              0.4 |                         0.99635 |                41 |
|               60 |       100000 |               0.15 |              0.4 |                         0.99946 |                41 |
|               80 |       100000 |               0.15 |              0.4 |                         0.99999 |                41 |
|              100 |       100000 |               0.15 |              0.4 |                         1       |                41 |

Boundary-null results:

|   n_confirmation |   replicates |   null_axis_mean |   other_axis_mean |   interval_width |   false_qualification_rate |
|-----------------:|-------------:|-----------------:|------------------:|-----------------:|---------------------------:|
|               10 |       200000 |                0 |              0.15 |              0.4 |                   0.0015   |
|               20 |       200000 |                0 |              0.15 |              0.4 |                   0.00442  |
|               30 |       200000 |                0 |              0.15 |              0.4 |                   0.007075 |
|               41 |       200000 |                0 |              0.15 |              0.4 |                   0.00561  |
|               60 |       200000 |                0 |              0.15 |              0.4 |                   0.00669  |
|              100 |       200000 |                0 |              0.15 |              0.4 |                   0.00618  |

Independent contract-shopping identity check: for K=27 and alpha=0.05, analytic 1-(1-alpha)^K=0.749656; Monte Carlo with 1,000,000 replicates gave 0.748725. The previously frozen manuscript simulation reported 0.7479; this new check is not used to replace the frozen value.
