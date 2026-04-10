from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"


FORMULAS = [
    (
        "formula_state_update.png",
        r"$p_{t+1}=\mathrm{entmax}_{1.5}\!\left(M p_t + \sum_i B_i a_i\right)$",
        (11, 1.4),
        30,
    ),
    (
        "formula_payoff.png",
        r"$u_i(a)=\mathbb{E}_{z\sim p_T(a)}\left[o_i(z)\right]-\lambda_i\|a_i\|_2^2-\mathrm{penalty}$",
        (13, 1.6),
        30,
    ),
    (
        "formula_mapping.png",
        r"$|\psi_t\rangle,\ |\psi_t|^2 = p_t$",
        (6, 1.0),
        24,
    ),
]


def main() -> None:
    for name, formula, size, font_size in FORMULAS:
        figure = plt.figure(figsize=size, dpi=240)
        axis = figure.add_axes([0, 0, 1, 1])
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            formula,
            fontsize=font_size,
            ha="center",
            va="center",
            color="black",
        )
        output = ASSET_DIR / name
        figure.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)
        print(output)


if __name__ == "__main__":
    main()
