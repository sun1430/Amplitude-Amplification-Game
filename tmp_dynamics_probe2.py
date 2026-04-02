from dataclasses import replace
from interference_game.experiments.common import load_cases
from interference_game.dynamics.simulate import simulate_dynamics

cases = {c.scenario_name: c for c in load_cases(r'C:\Users\szAsh\OneDrive\Desktop\UW\work\ECE752project\configs\full\dynamics.yaml')}
for scenario_name in ['high_conflict_exact_vs_approx','deep_competitive_high_conflict','no_mixing_low_conflict']:
    case = cases[scenario_name]
    models = {'exact': case.exact_game, **case.baselines}
    print('SCENARIO', scenario_name)
    for model_name in ['exact','aggregate','mean_field','no_mixing','sampling']:
        for method in ['projected_gradient','extra_gradient','best_response']:
            cfg = replace(case.dynamics_config, method=method, step_size=0.01 if method!='best_response' else 0.02, tolerance=0.1, stable_window=4, max_steps=200, cycle_tolerance=0.12, cycle_window=40)
            result = simulate_dynamics(models[model_name], cfg, seed=109)
            print(model_name, method, 'conv', result.converged, 'cycle', result.cycle_detected, 'tts', result.time_to_stability, 'var', round(result.trajectory_variance,6), 'u', round(float(result.final_utilities.mean().item()),6), 'len', len(result.action_history))
    print()
