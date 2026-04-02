from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "full"
DYNAMICS_RESULTS = RESULTS / "dynamics_reportable" if (RESULTS / "dynamics_reportable").exists() else RESULTS / "dynamics"
OUT_DIR = ROOT / "output" / "pdf"
OUT_PATH = OUT_DIR / "preliminary_report_bilingual.pdf"
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
FONT_NAME = "MSYH"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def average_by_baseline(rows: Iterable[dict[str, str]], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["baseline"], []).append(float(row[key]))
    return {baseline: statistics.mean(values) for baseline, values in grouped.items()}


def find_row(rows: Iterable[dict[str, str]], scenario: str, baseline: str) -> dict[str, str]:
    for row in rows:
        if row["scenario"] == scenario and row["baseline"] == baseline:
            return row
    raise KeyError(f"Missing row for scenario={scenario}, baseline={baseline}")


def filter_rows(rows: Iterable[dict[str, str]], **conditions: str) -> list[dict[str, str]]:
    result = []
    for row in rows:
        if all(row[key] == value for key, value in conditions.items()):
            result.append(row)
    return result


def register_font() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCNEN",
            parent=styles["Title"],
            fontName=FONT_NAME,
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCNEN",
            parent=styles["Normal"],
            fontName=FONT_NAME,
            fontSize=10.5,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "Heading1CNEN",
            parent=styles["Heading1"],
            fontName=FONT_NAME,
            fontSize=15,
            leading=18,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "Heading2CNEN",
            parent=styles["Heading2"],
            fontName=FONT_NAME,
            fontSize=12.5,
            leading=15,
            spaceBefore=6,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyCNEN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.8,
            leading=13.5,
            wordWrap="CJK",
            spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "CaptionCNEN",
            parent=styles["Italic"],
            fontName=FONT_NAME,
            fontSize=8.8,
            leading=11,
            wordWrap="CJK",
            textColor=colors.HexColor("#444444"),
            spaceBefore=3,
            spaceAfter=8,
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text), style)


def cn_en(cn: str, en: str, style: ParagraphStyle) -> list[Paragraph]:
    return [p(f"中文：{cn}", style), p(f"English: {en}", style)]


def fig(path: Path, caption_cn: str, caption_en: str, max_width: float, max_height: float, styles: dict[str, ParagraphStyle]):
    image = Image(str(path))
    width, height = image.imageWidth, image.imageHeight
    scale = min(max_width / width, max_height / height)
    image.drawWidth = width * scale
    image.drawHeight = height * scale
    caption = p(f"图 / Figure: {caption_cn} {caption_en}", styles["caption"])
    return [image, caption]


def make_table(data: list[list[str]], col_widths: list[float]) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 10.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#777777")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT_NAME, 8.5)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(doc.pagesize[0] - 40, 20, f"Page {doc.page}")
    canvas.restoreState()


def build_story() -> list:
    styles = build_styles()
    story: list = []

    payoff_rows = read_csv(RESULTS / "payoff" / "summary.csv")
    equilibrium_rows = read_csv(RESULTS / "equilibrium" / "summary.csv")
    dynamics_rows = read_csv(DYNAMICS_RESULTS / "summary.csv")
    ablation_rows = read_csv(RESULTS / "ablation" / "summary.csv")

    payoff_error = average_by_baseline(payoff_rows, "mean_abs_utility_error_mean")
    payoff_br = average_by_baseline(payoff_rows, "br_match_rate_mean")
    payoff_rank = average_by_baseline(payoff_rows, "spearman_rank_correlation_mean")

    cycle_rate = statistics.mean(float(row["cycle_detected"]) for row in dynamics_rows)
    convergence_rate = statistics.mean(float(row["converged"]) for row in dynamics_rows)

    br_rates: dict[str, tuple[float, float]] = {}
    for model_name in ["exact", "aggregate", "mean_field", "no_mixing", "sampling"]:
        model_rows = filter_rows(dynamics_rows, model_name=model_name, method="best_response")
        br_rates[model_name] = (
            statistics.mean(float(row["converged"]) for row in model_rows),
            statistics.mean(float(row["cycle_detected"]) for row in model_rows),
        )

    high_exact_br = filter_rows(dynamics_rows, scenario="high_conflict_exact_vs_approx", model_name="exact", method="best_response")
    high_nomix_br = filter_rows(dynamics_rows, scenario="high_conflict_exact_vs_approx", model_name="no_mixing", method="best_response")
    high_sampling_br = filter_rows(dynamics_rows, scenario="high_conflict_exact_vs_approx", model_name="sampling", method="best_response")
    shared_sampling_br = filter_rows(dynamics_rows, scenario="shared_target_variant", model_name="sampling", method="best_response")

    rep_l1_agg = find_row(payoff_rows, "N2_n4_L1_medium_B6", "aggregate")
    rep_l2_agg = find_row(payoff_rows, "N2_n4_L2_high_B10", "aggregate")
    rep_l2_nomix = find_row(payoff_rows, "N2_n4_L2_high_B10", "no_mixing")

    eq_nomix_shared_agg = find_row(equilibrium_rows, "N2_n2_nomixing_shared", "aggregate")
    eq_n2_mix_agg = find_row(equilibrium_rows, "N2_n4_mixing_high", "aggregate")
    eq_n2_mix_mf = find_row(equilibrium_rows, "N2_n4_mixing_high", "mean_field")
    eq_n2_mix_nomix = find_row(equilibrium_rows, "N2_n4_mixing_high", "no_mixing")
    eq_n3_mix_agg = find_row(equilibrium_rows, "N3_n2_mixing_high", "aggregate")
    eq_n3_mix_mf = find_row(equilibrium_rows, "N3_n2_mixing_high", "mean_field")

    ab_no_mixing = next(row for row in ablation_rows if row["scenario"] == "A_no_mixing")
    ab_with_mixing = next(row for row in ablation_rows if row["scenario"] == "B_with_mixing")
    ab_low_conflict = next(row for row in ablation_rows if row["scenario"] == "D_low_conflict")
    ab_high_conflict = next(row for row in ablation_rows if row["scenario"] == "E_high_conflict")

    story.append(Spacer(1, 0.35 * inch))
    story.append(p("Preliminary Bilingual Report: Mixing-Enhanced Interference Games / 双语初步报告：Mixing-Enhanced Interference Games", styles["title"]))
    story.append(
        p(
            "ECE 752 project progress report generated from the current repository implementation and results/full artifacts. "
            "This version is intended to be shareable as a standalone PDF with embedded figures.",
            styles["subtitle"],
        )
    )
    story.extend(
        cn_en(
            "本报告基于当前仓库实现与 results/full 中保存的实验结果整理而成，目标是形成一份可以直接发送的、带插图的初步报告。当前最有说服力的证据来自 payoff distortion 与 equilibrium distortion；sanity checks 已完成；在新跑完的 reportable dynamics 配置下，动态部分也已经出现了可报告的模型差异。",
            "This report is compiled from the current repository implementation and the saved results/full artifacts. It is intended as a directly shareable preliminary report with embedded figures. The strongest evidence currently comes from payoff distortion and equilibrium distortion. Sanity checks are complete, and the newly completed reportable dynamics configuration now also shows reportable model differences.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 0.18 * inch))

    story.append(p("1. Objective and Current Status / 研究目标与当前状态", styles["h1"]))
    story.extend(
        cn_en(
            "项目研究一个 mixing-enhanced interference game，并测试 aggregate、mean-field、sampling 与 no-mixing 等近似是否会破坏 exact game 的战略结构。与只比较单个 payoff 数值不同，本项目更强调 best response、equilibrium 和 dynamics 是否被保留。",
            "The project studies a mixing-enhanced interference game and tests whether aggregate, mean-field, sampling, and no-mixing approximations alter the strategic structure of the exact game. Rather than focusing only on individual payoff error, the project emphasizes preservation of best responses, equilibria, and dynamic behavior.",
            styles["body"],
        )
    )
    story.extend(
        cn_en(
            "当前仓库已经实现了 exact simulator、approximation baselines、离散均衡枚举、动态仿真和 config-driven 实验脚本。在目标环境 ece752-route2 中，项目测试已通过，因此本报告引用的结果具备基本可复现性。",
            "The current repository already implements the exact simulator, approximation baselines, discrete equilibrium enumeration, dynamic simulation, and config-driven experiment scripts. Tests pass in the intended environment ece752-route2, so the results referenced here have a basic level of reproducibility.",
            styles["body"],
        )
    )

    story.append(p("2. Sanity Checks / 合理性检查", styles["h1"]))
    story.extend(
        cn_en(
            "基础正确性检查已经全部通过，包括 state normalization、cost monotonicity、L=0 与 no-mixing 的一致性、shared target 下的对称性，以及受控 phase perturbation 对 utility 的影响。这些检查说明核心实现没有明显逻辑错误，后续结果具有解释基础。",
            "All foundational sanity checks passed, including state normalization, cost monotonicity, consistency between L=0 and no-mixing, symmetry under shared targets, and the effect of a controlled phase perturbation on utility. These checks suggest that the core implementation is logically sound enough to support interpretation of later experiments.",
            styles["body"],
        )
    )
    story.extend(fig(RESULTS / "sanity" / "sanity_summary.png", "Sanity checks summary.", "Summary of normalization and controlled perturbation checks.", 6.6 * inch, 3.2 * inch, styles))

    story.append(p("3. Payoff Distortion / 收益失真", styles["h1"]))
    story.extend(
        cn_en(
            "按 full payoff 实验中的所有 scenario 汇总后，aggregate、mean-field 与 no-mixing baseline 都出现了明显的 utility distortion 与 best-response mismatch。sampling baseline 当前表现很强，但这部分结论需要谨慎，因为其实现仍保留了较多 exact amplitude 信息。",
            "Aggregating over all full payoff scenarios, the aggregate, mean-field, and no-mixing baselines all show visible utility distortion and best-response mismatch. The current sampling baseline performs unusually well, but this conclusion should be treated cautiously because its implementation still retains more exact amplitude information than a pure outcome-only approximation.",
            styles["body"],
        )
    )
    payoff_table = make_table(
        [
            ["Baseline", "Mean abs. utility error", "BR match rate", "Rank correlation"],
            ["aggregate", f"{payoff_error['aggregate']:.3f}", f"{payoff_br['aggregate']:.3f}", f"{payoff_rank['aggregate']:.3f}"],
            ["mean_field", f"{payoff_error['mean_field']:.3f}", f"{payoff_br['mean_field']:.3f}", f"{payoff_rank['mean_field']:.3f}"],
            ["no_mixing", f"{payoff_error['no_mixing']:.3f}", f"{payoff_br['no_mixing']:.3f}", f"{payoff_rank['no_mixing']:.3f}"],
            ["sampling", f"{payoff_error['sampling']:.3f}", f"{payoff_br['sampling']:.3f}", f"{payoff_rank['sampling']:.3f}"],
        ],
        [1.5 * inch, 1.55 * inch, 1.2 * inch, 1.2 * inch],
    )
    story.append(payoff_table)
    story.append(Spacer(1, 0.12 * inch))
    story.extend(
        cn_en(
            f"代表性地看，在 N2_n4_L1_medium_B6 中，aggregate 的平均 utility error 为 {float(rep_l1_agg['mean_abs_utility_error_mean']):.3f}，best-response match rate 仅为 {float(rep_l1_agg['br_match_rate_mean']):.3f}。在 N2_n4_L2_high_B10 中，aggregate error 上升到 {float(rep_l2_agg['mean_abs_utility_error_mean']):.3f}，而 no-mixing 也达到 {float(rep_l2_nomix['mean_abs_utility_error_mean']):.3f}。这说明 mixing 与 conflict 的增强会显著削弱压缩近似对战略结构的保真度。",
            f"As representative examples, in N2_n4_L1_medium_B6 the aggregate baseline reaches a mean utility error of {float(rep_l1_agg['mean_abs_utility_error_mean']):.3f} and a best-response match rate of only {float(rep_l1_agg['br_match_rate_mean']):.3f}. In N2_n4_L2_high_B10, the aggregate error rises to {float(rep_l2_agg['mean_abs_utility_error_mean']):.3f}, while no-mixing also reaches {float(rep_l2_nomix['mean_abs_utility_error_mean']):.3f}. This indicates that stronger mixing and conflict significantly reduce the strategic faithfulness of compressed approximations.",
            styles["body"],
        )
    )
    story.extend(fig(RESULTS / "payoff" / "payoff_distortion.png", "Payoff distortion, best-response preservation, and candidate rank correlation across mixing depth.", "Payoff distortion, best-response preservation, and rank correlation across mixing depth.", 6.6 * inch, 2.8 * inch, styles))

    story.append(p("4. Equilibrium Distortion / 均衡失真", styles["h1"]))
    story.extend(
        cn_en(
            "均衡结果比单纯 payoff error 更有说服力，因为它直接说明近似是否会改变游戏的战略结构。在共享目标、无 mixing 的控制组中，aggregate 已经会改变 pure equilibrium 集合；进入 mixing/high-conflict 场景后，多种 baseline 会完全丢失 exact equilibrium overlap，并在重新放回 exact game 时产生较高 regret。",
            "The equilibrium results are more compelling than payoff error alone because they directly show whether an approximation changes the strategic structure of the game. Even in the shared-target, no-mixing control case, aggregate already alters the pure equilibrium set. Once the game enters mixing and high-conflict settings, several baselines lose exact equilibrium overlap entirely and incur substantial regret when re-evaluated in the exact game.",
            styles["body"],
        )
    )
    equilibrium_table = make_table(
        [
            ["Scenario", "Baseline", "Pure overlap", "Exact regret"],
            ["N2_n2_nomixing_shared", "aggregate", f"{float(eq_nomix_shared_agg['pure_jaccard']):.3f}", f"{float(eq_nomix_shared_agg['mean_exact_regret_of_approx_epsilon']):.3f}"],
            ["N2_n4_mixing_high", "no_mixing", f"{float(eq_n2_mix_nomix['pure_jaccard']):.3f}", f"{float(eq_n2_mix_nomix['mean_exact_regret_of_approx_epsilon']):.3f}"],
            ["N2_n4_mixing_high", "aggregate", f"{float(eq_n2_mix_agg['pure_jaccard']):.3f}", f"{float(eq_n2_mix_agg['mean_exact_regret_of_approx_epsilon']):.3f}"],
            ["N2_n4_mixing_high", "mean_field", f"{float(eq_n2_mix_mf['pure_jaccard']):.3f}", f"{float(eq_n2_mix_mf['mean_exact_regret_of_approx_epsilon']):.3f}"],
            ["N3_n2_mixing_high", "aggregate", f"{float(eq_n3_mix_agg['pure_jaccard']):.3f}", f"{float(eq_n3_mix_agg['mean_exact_regret_of_approx_epsilon']):.3f}"],
            ["N3_n2_mixing_high", "mean_field", f"{float(eq_n3_mix_mf['pure_jaccard']):.3f}", f"{float(eq_n3_mix_mf['mean_exact_regret_of_approx_epsilon']):.3f}"],
        ],
        [2.2 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch],
    )
    story.append(equilibrium_table)
    story.append(Spacer(1, 0.12 * inch))
    story.extend(fig(RESULTS / "equilibrium" / "equilibrium_distortion.png", "Equilibrium overlap and approximate-equilibrium regret.", "Equilibrium overlap and regret of approximate equilibria under the exact game.", 6.6 * inch, 2.8 * inch, styles))

    story.append(PageBreak())
    story.append(p("5. Dynamics and Ablation / 动态行为与消融分析", styles["h1"]))
    story.extend(
        cn_en(
            f"新跑完的 reportable dynamics 配置明显比旧版更有信息量：整体平均 convergence rate 为 {convergence_rate:.3f}，整体平均 cycle rate 为 {cycle_rate:.3f}。更具体地说，projected gradient 与 extra-gradient 在当前场景中基本全部收敛且没有检测到 cycle；主要差异集中在 best-response dynamics 上。",
            f"The newly completed reportable dynamics configuration is substantially more informative than the previous one: the overall average convergence rate is {convergence_rate:.3f}, and the overall average cycle rate is {cycle_rate:.3f}. More specifically, projected-gradient and extra-gradient runs converge almost everywhere under the current scenarios with no detected cycles; the main separation appears under best-response dynamics.",
            styles["body"],
        )
    )
    dynamic_table = make_table(
        [
            ["Model", "BR convergence rate", "BR cycle rate"],
            ["exact", f"{br_rates['exact'][0]:.3f}", f"{br_rates['exact'][1]:.3f}"],
            ["aggregate", f"{br_rates['aggregate'][0]:.3f}", f"{br_rates['aggregate'][1]:.3f}"],
            ["mean_field", f"{br_rates['mean_field'][0]:.3f}", f"{br_rates['mean_field'][1]:.3f}"],
            ["no_mixing", f"{br_rates['no_mixing'][0]:.3f}", f"{br_rates['no_mixing'][1]:.3f}"],
            ["sampling", f"{br_rates['sampling'][0]:.3f}", f"{br_rates['sampling'][1]:.3f}"],
        ],
        [1.5 * inch, 1.6 * inch, 1.35 * inch],
    )
    story.append(dynamic_table)
    story.append(Spacer(1, 0.12 * inch))
    story.extend(
        cn_en(
            f"这组结果给出了一个比原 proposal 更细的图景。在 `high_conflict_exact_vs_approx` 场景下，exact best-response 的收敛率为 {statistics.mean(float(row['converged']) for row in high_exact_br):.3f}、cycle rate 为 {statistics.mean(float(row['cycle_detected']) for row in high_exact_br):.3f}；而 no-mixing 的 cycle rate 为 {statistics.mean(float(row['cycle_detected']) for row in high_nomix_br):.3f}，sampling 为 {statistics.mean(float(row['cycle_detected']) for row in high_sampling_br):.3f}。在 `shared_target_variant` 中，sampling 的 best-response cycle rate 仍达到 {statistics.mean(float(row['cycle_detected']) for row in shared_sampling_br):.3f}。因此，当前动态证据支持“不同近似会改变动态行为”，但并不支持最初版本里“exact 更容易 cycling”的强假设。",
            f"These results provide a more nuanced picture than the original proposal. In the `high_conflict_exact_vs_approx` scenario, the exact best-response dynamics has convergence rate {statistics.mean(float(row['converged']) for row in high_exact_br):.3f} and cycle rate {statistics.mean(float(row['cycle_detected']) for row in high_exact_br):.3f}; by contrast, the no-mixing cycle rate is {statistics.mean(float(row['cycle_detected']) for row in high_nomix_br):.3f}, and the sampling cycle rate is {statistics.mean(float(row['cycle_detected']) for row in high_sampling_br):.3f}. In `shared_target_variant`, the sampling best-response cycle rate remains {statistics.mean(float(row['cycle_detected']) for row in shared_sampling_br):.3f}. Therefore, the current dynamics evidence supports the claim that approximations can change dynamic behavior, but it does not support the stronger original hypothesis that the exact model is systematically more prone to cycling.",
            styles["body"],
        )
    )
    story.extend(
        cn_en(
            f"消融实验提供了方向性证据。在 A_no_mixing 中，mean-field error 近似为 {float(ab_no_mixing['mean_field_error']):.3f}；引入 mixing 后，在 B_with_mixing 中上升到 {float(ab_with_mixing['mean_field_error']):.3f}。同时，aggregate error 在 D_low_conflict 中约为 {float(ab_low_conflict['aggregate_error']):.3f}，而在 E_high_conflict 中上升到 {float(ab_high_conflict['aggregate_error']):.3f}。这表明 mixing 与目标冲突都可能加剧 approximation failure。",
            f"The ablation study provides directional evidence. In A_no_mixing, the mean-field error is approximately {float(ab_no_mixing['mean_field_error']):.3f}; after introducing mixing in B_with_mixing, it increases to {float(ab_with_mixing['mean_field_error']):.3f}. Meanwhile, the aggregate error is about {float(ab_low_conflict['aggregate_error']):.3f} in D_low_conflict and rises to {float(ab_high_conflict['aggregate_error']):.3f} in E_high_conflict. This suggests that both mixing and target conflict can worsen approximation failure.",
            styles["body"],
        )
    )
    story.extend(fig(DYNAMICS_RESULTS / "dynamics_summary.png", "Dynamics outcome rates and a representative trajectory.", "Dynamics outcome rates and a representative utility trajectory.", 6.6 * inch, 3.0 * inch, styles))
    story.extend(fig(RESULTS / "ablation" / "ablation_summary.png", "Ablation summary for approximation error and exact dynamics cycle rate.", "Ablation summary for approximation error and exact dynamics cycle rate.", 6.6 * inch, 3.0 * inch, styles))

    story.append(p("6. Preliminary Conclusion / 初步结论", styles["h1"]))
    story.extend(
        cn_en(
            "基于当前结果，本项目已经足以形成一份可信的初步报告。最扎实的结论是：在 mixing-enhanced exact game 中，aggregate、no-mixing 以及部分 mean-field 近似会明显扭曲 payoff ranking、best response 与 equilibrium structure；这种 distortion 在若干 high-conflict 场景下相当显著。更新后的 dynamics 结果进一步表明，不同近似确实会改变动态行为，但其方向并不完全符合最初的 H3 假设。",
            "Based on the current results, the project already supports a credible preliminary report. The strongest conclusion is that in the mixing-enhanced exact game, aggregate, no-mixing, and in some cases mean-field approximations visibly distort payoff ranking, best responses, and equilibrium structure; this distortion becomes substantial in several high-conflict settings. The updated dynamics results further show that approximations do change dynamic behavior, but the direction of the effect does not fully align with the original H3 hypothesis.",
            styles["body"],
        )
    )
    story.extend(
        cn_en(
            "因此，当前最合适的外发版本应把 narrative 聚焦在 strategic preservation failure 上，并明确写出两点限制：第一，sampling baseline 需要进一步弱化；第二，dynamics 仍属 ongoing investigation。这样的表述既能展示已有成果，也不会因为结论过度而失去可信度。",
            "Therefore, the most appropriate external-facing narrative at this stage is to focus on strategic preservation failure and to state two limitations explicitly: first, the sampling baseline still needs to be weakened; second, the dynamics study remains an ongoing investigation. This framing presents the current accomplishments clearly without overstating the evidence.",
            styles["body"],
        )
    )

    return story


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    register_font()
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=42,
        bottomMargin=34,
        title="Preliminary Bilingual Report",
        author="OpenAI Codex",
    )
    doc.build(build_story(), onFirstPage=page_number, onLaterPages=page_number)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
