__version__ = "0.1.1"
__author__ = "Gisell"

from .objectives import (
    objective_func_dim_penalty_param,
    objective_func_trust_constr,
    objective_penalty_norm_relaxation,
)
from .optimizers import (
    optimizer_grad_desc_penalty,
    optimizer_basinhop_constrained,
    optimizer_grad_desc_penalty_norm_ramp,
    optimizer_box_grad_desc_penalty_norm_ramp,
)

# from .multivariate_normal_pdf import (
#     multivariate_normal_pdf_func,
#     build_grid_and_pdf,
#     build_jsa_grid,
# )

from .gaussian_jsa import (
    multivariate_normal_pdf_func,
    build_grid_and_pdf,
    build_jsa_grid,
    scaled_gaussian_jsa_vectorized_rsvd,
)

from .waveguide_spdc_jsa import (
    ldelk,
    scaled_spdc_jsa_vectorized,
    scaled_spdc_jsa_chirped,
    scaled_spdc_jsa_vectorized_rsvd,
)

from .run_optimization import (
    penalty_continuation_grad_desc,
    run_comparison,
    run_penalty_only,
    run_trust_only,
    penalty_norm_relaxation_grad_desc,
    run_penalty_norm_relaxation,
    penalty_norm_relaxation_box_grad_desc,
    run_penalty_norm_relaxation_box,
)

from .penalty_trust_comparison_plots import (
    plot_2d_projection,
    plot_modes,
    plot_init_mode,
    plot_outer_iterations,
    plot_final_objective,
    plot_constraint_violation,
    plot_outer_objective_both_seeds,
    plot_all_amplitudes,
    is_totally_symmetric,
)

from .penalty_trust_data_tables import (
    compute_penalty_vs_trust_similarity,
    summarize_outer_results,
)

from .utils import (
    generate_seeds,
    save_results,
    load_results,
    to_complex,
)

from .bounds import (
    tensor_knapsack_bound,
    tensor_knapsack_bound_4D,
    tensor_knapsack_bound_6d,
    tensor_knapsack_bound_5D,
)
