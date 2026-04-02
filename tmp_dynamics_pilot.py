from dataclasses import replace
from pathlib import Path

from interference_game.dynamics.simulate import simulate_dynamics
from interference_game.experiments.common import load_cases


def summarize(case_name: str, method: str, step_size: float, max_steps: int, tolerance: float, stable_window: int) -> None:
    config_path = Path("configs/full/dynamics.yaml")
    cases = {case.scenario_name: case for case in load_cases(config_path)}
    case = cases[case_name]
    model = case.exact_game
    stats = []
    for init_idx in range(3):
        config = replace(
            case.dynamics_config,
            method=method,
            step_size=step_size,
            max_steps=max_steps,
            tolerance=tolerance,
            stable_window=stable_window,
        )
        result = simulate_dynamics(model, config, seed=700 + init_idx)
        stats.append(
            (
                int(result.converged),
                int(result.cycle_detected),
                result.time_to_stability if result.time_to_stability is not None else -1,
                result.trajectory_variance,
                float(result.final_utilities.mean().item()),
                len(result.action_history) - 1,
            )
        )

    converged = sum(item[0] for item in stats) / len(stats)
    cycled = sum(item[1] for item in stats) / len(stats)
    avg_time = sum(item[2] for item in stats if item[2] >= 0) / (sum(1 for item in stats if item[2] >= 0) or 1)
    avg_variance = sum(item[3] for item in stats) / len(stats)
    avg_utility = sum(item[4] for item in stats) / len(stats)
    avg_steps = sum(item[5] for item in stats) / len(stats)
    print(
        case_name,
        method,
        {
            "step_size": step_size,
            "max_steps": max_steps,
            "tolerance": tolerance,
            "stable_window": stable_window,
            "converged_rate": converged,
            "cycle_rate": cycled,
            "avg_time": avg_time,
            "avg_variance": round(avg_variance, 6),
            "avg_utility": round(avg_utility, 6),
            "avg_steps": avg_steps,
        },
    )


if __name__ == "__main__":
    trials = [
        ("shared_target_variant", "projected_gradient", 0.05, 160, 0.02, 5),
        ("shared_target_variant", "projected_gradient", 0.02, 220, 0.015, 6),
        ("shared_target_variant", "extra_gradient", 0.05, 160, 0.02, 5),
        ("shared_target_variant", "extra_gradient", 0.02, 220, 0.015, 6),
        ("shared_target_variant", "best_response", 0.08, 80, 0.03, 4),
        ("high_conflict_exact_vs_approx", "projected_gradient", 0.05, 160, 0.02, 5),
        ("high_conflict_exact_vs_approx", "projected_gradient", 0.02, 220, 0.015, 6),
        ("high_conflict_exact_vs_approx", "extra_gradient", 0.05, 160, 0.02, 5),
        ("high_conflict_exact_vs_approx", "extra_gradient", 0.02, 220, 0.015, 6),
        ("high_conflict_exact_vs_approx", "best_response", 0.08, 80, 0.03, 4),
    ]
    for trial in trials:
        summarize(*trial)

    print("UNIFIED-CONFIG CHECK")
    cases = {case.scenario_name: case for case in load_cases(Path("configs/full/dynamics.yaml"))}
    unified = replace(cases["shared_target_variant"].dynamics_config, step_size=0.05, max_steps=160, tolerance=0.02, stable_window=5)
    for scenario_name in ["shared_target_variant", "high_conflict_exact_vs_approx"]:
        case = cases[scenario_name]
        for model_name, model in {"exact": case.exact_game, **case.baselines}.items():
            for method in ["best_response", "projected_gradient", "extra_gradient"]:
                stats = []
                config = replace(unified, method=method)
                for init_idx in range(2):
                    result = simulate_dynamics(model, config, seed=900 + init_idx)
                    stats.append((int(result.converged), int(result.cycle_detected), result.trajectory_variance, float(result.final_utilities.mean().item())))
                print(
                    scenario_name,
                    model_name,
                    method,
                    {
                        "converged_rate": sum(item[0] for item in stats) / len(stats),
                        "cycle_rate": sum(item[1] for item in stats) / len(stats),
                        "avg_variance": round(sum(item[2] for item in stats) / len(stats), 6),
                        "avg_utility": round(sum(item[3] for item in stats) / len(stats), 6),
                    },
                )
