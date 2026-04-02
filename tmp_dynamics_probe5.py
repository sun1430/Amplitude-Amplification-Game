from dataclasses import replace
from interference_game.experiments.common import load_cases
from interference_game.dynamics.simulate import simulate_dynamics

case = {c.scenario_name: c for c in load_cases(r'C:\Users\szAsh\OneDrive\Desktop\UW\work\ECE752project\configs\full\dynamics.yaml')}['high_conflict_exact_vs_approx']
models = {'exact': case.exact_game, **case.baselines}
for model_name in ['exact','aggregate','mean_field','no_mixing','sampling']:
    for method in ['projected_gradient','extra_gradient','best_response']:
        cfg = replace(case.dynamics_config, method=method, step_size=0.02, tolerance=0.1, stable_window=4, max_steps=150, cycle_tolerance=0.12, cycle_window=30)
        result = simulate_dynamics(models[model_name], cfg, seed=109)
        print(model_name, method, result.converged, result.cycle_detected, result.time_to_stability, round(result.trajectory_variance,6), round(float(result.final_utilities.mean().item()),6), len(result.action_history))
