# Preliminary Report / 初步报告

Date / 日期: 2026-04-01

## Status / 当前判断

中文：
基于仓库中的实现、保存下来的 `full` 实验结果，以及在目标环境中的通过测试结果，本项目已经具备撰写一版初步报告的条件。当前最有说服力的证据来自 payoff distortion 和 equilibrium distortion；sanity checks 已完成且全部通过；dynamics 部分已经有初步数据，但暂时不足以支持关于 cycling 或 convergence 的强结论。因此，本报告将其定位为中期进展报告，而不是最终版结题报告。

English:
Based on the implemented code, the saved `full` experiment outputs, and passing tests in the intended environment, the project is ready for an initial written report. The strongest evidence currently comes from payoff distortion and equilibrium distortion. Sanity checks are complete and passed. The dynamics experiments already produce usable observations, but they do not yet support strong claims about cycling or convergence. Accordingly, this document should be treated as a mid-project preliminary report rather than a final report.

## 1. Project Goal / 项目目标

中文：
本项目研究一个 mixing-enhanced interference game，并检验常见压缩近似是否会破坏精确博弈中的战略结构。核心问题不是近似是否会改变某个单独 payoff，而是它们是否会改变 best response、equilibrium 以及动态行为。与 proposal 一致，项目关注四类近似或对照：`aggregate`、`mean_field`、`sampling` 和 `no_mixing`。

English:
This project studies a mixing-enhanced interference game and tests whether common compressed approximations fail to preserve the strategic structure of the exact game. The main question is not only whether an approximation changes a payoff value, but whether it changes best responses, equilibria, and dynamic behavior. Consistent with the proposal, the main baselines are `aggregate`, `mean_field`, `sampling`, and `no_mixing`.

## 2. Implementation Status / 实现完成度

中文：
仓库已经实现了 proposal 中的大部分模块，包括 exact simulator、baseline models、离散均衡枚举、动态仿真、config-driven experiments，以及从结果文件自动生成图表的脚本。主要代码位于：

- `interference_game/models`
- `interference_game/equilibrium`
- `interference_game/dynamics`
- `interference_game/experiments`

English:
Most modules proposed in the project proposal have already been implemented, including the exact simulator, approximation baselines, discrete equilibrium enumeration, dynamic simulation, config-driven experiments, and plotting scripts that generate figures from saved results. The main code lives in:

- `interference_game/models`
- `interference_game/equilibrium`
- `interference_game/dynamics`
- `interference_game/experiments`

中文：
项目还具有基本可复现性。使用目标 Conda 环境 `ece752-route2` 运行测试时，`pytest` 结果为 `10 passed`。需要注意，若直接在当前机器的 `base` 环境中运行，可能因为 `numpy`/`pandas` 版本不匹配而失败；因此结果应当以 [environment.yml](environment.yml) 指定的环境为准。

English:
The project also has a basic level of reproducibility. Running tests inside the intended Conda environment `ece752-route2` gives `10 passed`. A practical caveat is that running in the machine's `base` environment may fail because of a `numpy`/`pandas` version mismatch, so the intended reference environment is the one declared in [environment.yml](environment.yml).

## 3. Experimental Artifacts / 现有实验产物

中文：
当前已经存在完整的 `full` 结果目录，可直接作为报告素材：

- [results/full/sanity](results/full/sanity)
- [results/full/payoff](results/full/payoff)
- [results/full/equilibrium](results/full/equilibrium)
- [results/full/dynamics](results/full/dynamics)
- [results/full/ablation](results/full/ablation)

English:
The repository already contains a complete `full` results directory that can be used directly as report material:

- [results/full/sanity](results/full/sanity)
- [results/full/payoff](results/full/payoff)
- [results/full/equilibrium](results/full/equilibrium)
- [results/full/dynamics](results/full/dynamics)
- [results/full/ablation](results/full/ablation)

## 4. Preliminary Results / 初步结果

### 4.1 Sanity Checks / 合理性检查

中文：
Sanity checks 已全部通过，见 [results/full/sanity/sanity_note.txt](results/full/sanity/sanity_note.txt)。检查内容包括：state normalization、cost monotonicity、`L=0` 与 `no_mixing` 的一致性、shared target 下的对称性，以及受控 phase perturbation 对 utility 的影响。这说明基础实现没有明显逻辑错误，足以支持后续实验解释。

English:
All sanity checks passed, as recorded in [results/full/sanity/sanity_note.txt](results/full/sanity/sanity_note.txt). The checks cover state normalization, cost monotonicity, consistency between `L=0` and `no_mixing`, symmetry under shared targets, and the effect of a controlled phase perturbation on utility. This suggests that the core implementation is logically consistent enough to support downstream interpretation.

Relevant figure / 对应图表:

- [results/full/sanity/sanity_summary.png](results/full/sanity/sanity_summary.png)

### 4.2 Payoff Distortion / 收益失真

中文：
Payoff distortion 是当前最强的一组结果。按全部 `full` payoff scenarios 做平均，四类 baseline 的表现如下：

| Baseline | Mean absolute utility error | Best-response match rate | Spearman rank correlation |
| --- | ---: | ---: | ---: |
| aggregate | 0.143 | 0.433 | 0.577 |
| mean_field | 0.074 | 0.594 | 0.718 |
| no_mixing | 0.118 | 0.399 | 0.480 |
| sampling | 0.010 | 0.933 | 0.982 |

English:
Payoff distortion is currently the strongest part of the empirical evidence. Averaged over all `full` payoff scenarios, the four baselines behave as follows:

| Baseline | Mean absolute utility error | Best-response match rate | Spearman rank correlation |
| --- | ---: | ---: | ---: |
| aggregate | 0.143 | 0.433 | 0.577 |
| mean_field | 0.074 | 0.594 | 0.718 |
| no_mixing | 0.118 | 0.399 | 0.480 |
| sampling | 0.010 | 0.933 | 0.982 |

中文：
这些结果表明，`aggregate` 和 `no_mixing` 往往无法可靠保留 exact game 的 action ranking 和 best response。`mean_field` 相对更稳健，但仍会在 mixing/high-conflict 情况下出现明显误差。当前实现中的 `sampling` baseline 表现异常强，这一点需要在解释中保持谨慎，后文会单独说明。

English:
These numbers indicate that `aggregate` and `no_mixing` often fail to preserve action ranking and best-response structure in the exact game. `mean_field` is relatively more stable, but still exhibits substantial error in mixing and high-conflict settings. The current `sampling` baseline performs unusually well, which should be interpreted cautiously; this caveat is discussed later.

中文：
代表性案例进一步支持这一趋势。对于低冲突、无 mixing 的控制组 `N2_n4_L0_low_B2`，`mean_field` 与 exact 基本一致，而 `aggregate` 已经产生了 `0.072` 的平均 utility error。进入有 mixing 的情形后，误差迅速增大。例如在 `N2_n4_L1_medium_B6` 中，`aggregate` 的平均 utility error 达到 `0.226`，best-response match rate 下降到 `0.292`；在 `N2_n4_L2_high_B10` 中，`aggregate` error 进一步上升到 `0.310`，`no_mixing` 也达到 `0.247`。这与 proposal 中“mixing 可能破坏压缩近似的战略保真度”的主叙事是一致的，虽然误差随 mixing depth 单调增长并不是在所有 baseline 上都严格成立。

English:
Representative cases reinforce the same pattern. In the low-conflict, no-mixing control case `N2_n4_L0_low_B2`, `mean_field` is essentially identical to the exact model, while `aggregate` already incurs a mean utility error of `0.072`. Once mixing is introduced, the distortion grows quickly. For instance, in `N2_n4_L1_medium_B6`, `aggregate` reaches a mean utility error of `0.226` and the best-response match rate drops to `0.292`; in `N2_n4_L2_high_B10`, the `aggregate` error rises further to `0.310`, and `no_mixing` also reaches `0.247`. This is broadly consistent with the proposal's narrative that mixing can break the strategic faithfulness of compressed approximations, although monotonic growth with depth does not hold uniformly for every baseline.

Relevant figure / 对应图表:

- [results/full/payoff/payoff_distortion.png](results/full/payoff/payoff_distortion.png)
- [results/full/payoff/summary.csv](results/full/payoff/summary.csv)

### 4.3 Equilibrium Distortion / 均衡失真

中文：
均衡结果同样支持“近似会改变战略结构”的核心论点。对于 `N2_n2_nomixing_shared` 这一共享目标、无 mixing 的控制案例，`mean_field` 与 `sampling` 完全保留 pure equilibrium 集合，而 `aggregate` 的 pure-equilibrium Jaccard overlap 只有 `0.474`。这说明即使在较简单的 setting 下，粗糙聚合也可能改变均衡集合。

English:
The equilibrium results also support the main claim that approximations can change strategic structure. In the shared-target, no-mixing control case `N2_n2_nomixing_shared`, `mean_field` and `sampling` fully preserve the pure equilibrium set, whereas `aggregate` achieves only a pure-equilibrium Jaccard overlap of `0.474`. This shows that even in a comparatively simple setting, coarse aggregation can alter equilibrium structure.

中文：
在 mixing/high-conflict 场景下，这种失真更加明显。对于 `N2_n4_mixing_high`，`no_mixing`、`aggregate` 和 `mean_field` 的 pure-equilibrium overlap 全部为 `0.0`，且把近似模型找到的 epsilon-equilibria 重新放回 exact game 评估时，平均 regret 分别约为 `0.323`、`0.304` 和 `0.262`。对于 `N3_n2_mixing_high`，`aggregate` 仍然为 `0.0` overlap 且 regret 约 `0.320`；`mean_field` 部分恢复了结构，overlap 为 `0.667`，但仍不是完全保真。总体上，这已经足以支撑报告中关于“近似会错判均衡数量和均衡位置”的初步结论。

English:
The distortion becomes much more pronounced in mixing and high-conflict settings. For `N2_n4_mixing_high`, the pure-equilibrium overlap is `0.0` for `no_mixing`, `aggregate`, and `mean_field`, and when epsilon-equilibria found by the approximate models are re-evaluated in the exact game, their mean regrets are about `0.323`, `0.304`, and `0.262`, respectively. For `N3_n2_mixing_high`, `aggregate` still has `0.0` overlap and a regret of about `0.320`; `mean_field` partially recovers the structure with overlap `0.667`, but is still not fully faithful. Overall, this is already enough to support an initial conclusion that approximations can misidentify both the number and the location of equilibria.

Representative equilibrium metrics / 代表性均衡指标:

| Scenario | Baseline | Pure-equilibrium overlap | Mean exact regret of approx. epsilon equilibria |
| --- | --- | ---: | ---: |
| N2_n2_nomixing_shared | aggregate | 0.474 | 0.013 |
| N2_n2_nomixing_shared | mean_field | 1.000 | 0.000 |
| N2_n4_mixing_high | no_mixing | 0.000 | 0.323 |
| N2_n4_mixing_high | aggregate | 0.000 | 0.304 |
| N2_n4_mixing_high | mean_field | 0.000 | 0.262 |
| N3_n2_mixing_high | no_mixing | 0.000 | 0.170 |
| N3_n2_mixing_high | aggregate | 0.000 | 0.320 |
| N3_n2_mixing_high | mean_field | 0.667 | 0.033 |

Relevant figure / 对应图表:

- [results/full/equilibrium/equilibrium_distortion.png](results/full/equilibrium/equilibrium_distortion.png)
- [results/full/equilibrium/summary.csv](results/full/equilibrium/summary.csv)

### 4.4 Dynamics / 动态行为

中文：
动态部分目前已经有结果，但不适合写成强结论。按照当前 `full` dynamics 配置，所有记录到的 convergence rate 和 cycle rate 都是 `0.0`。这意味着当前实验至少说明两件事：第一，proposal 中关于 cycling/oscillation 的假设还没有被当前配置验证出来；第二，当前 step size、tolerance、cycle detector 或运行步数可能还不足以把差异显式放大。

English:
The dynamics section already produces data, but it is not yet suitable for strong claims. Under the current `full` dynamics configuration, all recorded convergence rates and cycle rates are `0.0`. This means at least two things: first, the proposal's cycling and oscillation hypothesis has not yet been validated under the current setup; second, the current step sizes, tolerances, cycle detector, or horizon length may not be sufficient to expose the intended differences.

中文：
尽管如此，轨迹仍然显示 exact 与近似模型在 utility 水平和轨迹方差上存在差异。例如在 `high_conflict_exact_vs_approx` 场景下，`best_response` 动态中 exact 的最终平均 utility 约为 `0.164`，而 `aggregate` 为 `0.225`、`mean_field` 为 `0.112`、`no_mixing` 为 `0.056`。这说明动态更新下的行为结果并不一致，只是目前还无法据此稳健宣称“exact 更容易 cycling”或“approx 更容易收敛”。

English:
That said, the trajectories still show differences in utility level and trajectory variance between the exact and approximate models. For example, in the `high_conflict_exact_vs_approx` scenario under `best_response` dynamics, the final mean utility is about `0.164` for the exact model, compared with `0.225` for `aggregate`, `0.112` for `mean_field`, and `0.056` for `no_mixing`. This suggests that dynamic behavior is not identical across models, but the current evidence is still insufficient to claim robustly that the exact game cycles more often or that approximate games converge more readily.

Relevant figure / 对应图表:

- [results/full/dynamics/dynamics_summary.png](results/full/dynamics/dynamics_summary.png)
- [results/full/dynamics/summary.csv](results/full/dynamics/summary.csv)

### 4.5 Ablation / 消融实验

中文：
消融结果目前给出的是方向性证据，而不是最终定论。最清楚的现象是：在 `A_no_mixing` 中，`mean_field` error 基本为 `0`；而引入 mixing 后，在 `B_with_mixing` 中 `mean_field` error 上升到 `0.059`。此外，冲突程度对误差也有明显影响：`D_low_conflict` 下 `aggregate` error 约为 `0.060`，而 `E_high_conflict` 上升到 `0.115`。这说明 heterogeneous objectives 和 target conflict 确实会加重 preservation failure。

English:
At this stage, the ablation results provide directional evidence rather than a final conclusion. The clearest pattern is that in `A_no_mixing`, the `mean_field` error is essentially `0`, while after mixing is introduced in `B_with_mixing`, the `mean_field` error increases to `0.059`. In addition, conflict level has a visible effect: under `D_low_conflict`, the `aggregate` error is about `0.060`, whereas it rises to `0.115` under `E_high_conflict`. This suggests that heterogeneous objectives and target conflict do contribute to preservation failure.

中文：
不过，当前 ablation 中的 dynamics 指标仍然全部为 `0.0`，因此这一部分还不能支撑 proposal 里关于“mixing 导致更高 cycle rate”的叙述。更稳妥的写法是：当前消融已经表明 conflict 和 mixing 会影响 payoff-level approximation quality，但对动态稳定性的影响仍需要更强实验设计。

English:
However, the dynamics-related ablation metrics are still all `0.0`, so this part does not yet support the proposal's narrative that mixing increases cycle rates. A more defensible statement is that the current ablations already show that conflict and mixing affect payoff-level approximation quality, while their effect on dynamic stability still requires stronger experimental design.

Relevant figure / 对应图表:

- [results/full/ablation/ablation_summary.png](results/full/ablation/ablation_summary.png)
- [results/full/ablation/summary.csv](results/full/ablation/summary.csv)

## 5. Main Takeaways / 当前核心结论

中文：
基于现有结果，本项目已经可以提出以下初步结论：

1. 在多个 mixing/high-conflict 配置下，`aggregate`、`no_mixing` 以及部分 `mean_field` 近似都无法稳定保留 exact game 的 payoff ranking 与 best response。
2. 均衡失真比单纯 payoff error 更能支持 proposal 的核心论点。多个场景中，近似模型会丢失 exact equilibria，或者引入在 exact game 中 regret 很高的伪均衡。
3. `mean_field` 在无 mixing 或较弱结构耦合时较为可靠，但在更复杂 setting 下并不总能保持战略结构。
4. 当前实现中的 `sampling` baseline 表现过强，暂时不适合直接作为“measurement-level approximation 失败”的证据。
5. dynamics 部分已有初步差异，但尚不能支撑有关 cycling、oscillation 或 convergence advantage 的强论断。

English:
Based on the current results, the project can already support the following preliminary conclusions:

1. Under several mixing and high-conflict configurations, `aggregate`, `no_mixing`, and sometimes `mean_field` fail to reliably preserve payoff ranking and best responses from the exact game.
2. Equilibrium distortion is more convincing than payoff error alone for the proposal's central claim. In several scenarios, approximate models either miss exact equilibria or introduce false equilibria with high regret in the exact game.
3. `mean_field` is reliable in no-mixing or weakly coupled settings, but it does not preserve strategic structure universally once the interaction becomes richer.
4. The current `sampling` baseline is unusually strong and should not yet be used as direct evidence that measurement-level approximations fail.
5. The dynamics section already shows differences, but it still does not support strong claims about cycling, oscillation, or convergence advantages.

## 6. Limitations / 当前局限

中文：
本报告必须明确以下局限，否则会过度陈述：

1. `sampling` baseline 的定义目前使用了 exact state 的振幅信息来估计 target overlap，因此它比纯 measurement-only baseline 更强。
2. dynamics 图表目前信息量偏低，既没有出现显式 cycle，也没有出现稳定收敛率差异。
3. 当前绘图脚本更偏“实验调试图”，而不是最终论文图，尤其是 equilibrium 与 dynamics 图的标签布局还不够好。
4. 有些趋势是明显的，但并非所有 baseline、所有维度、所有 agent 数下都严格单调，因此最终写作需要避免绝对化表述。

English:
The following limitations should be stated explicitly to avoid overstating the results:

1. The current `sampling` baseline uses amplitude information from the exact state to estimate target overlap, which makes it stronger than a purely measurement-only baseline.
2. The current dynamics figures have limited evidential value: they show neither explicit cycles nor meaningful differences in convergence rates.
3. The plotting scripts are closer to experiment-debugging plots than final-paper figures, especially for the equilibrium and dynamics visualizations.
4. Some trends are clear, but they are not strictly monotonic across every baseline, dimension, and agent count, so the final writeup should avoid absolute claims.

## 7. Recommended Next Steps / 下一步建议

中文：
若要把这份初步报告推进到最终版，优先级建议如下：

1. 重新定义或弱化 `sampling` baseline，使其真正只依赖 outcome distribution，而不是 exact amplitudes。
2. 调整 dynamics 实验设计，包括更长 horizon、更激进的 step size sweep、不同 cycle detector，以及更能诱发不稳定性的 scenario。
3. 重画 payoff、equilibrium 和 dynamics 图，使其更适合报告或答辩展示。
4. 在最终报告中把核心叙事聚焦到“strategic preservation failure”，并把 dynamics 结论明确降级为 ongoing investigation，除非新实验能提供更强证据。

English:
To turn this preliminary report into a final version, the recommended priorities are:

1. Redefine or weaken the `sampling` baseline so that it truly depends only on the outcome distribution rather than exact amplitudes.
2. Redesign the dynamics experiments, including longer horizons, stronger step-size sweeps, alternative cycle detectors, and scenarios more likely to induce instability.
3. Rework the payoff, equilibrium, and dynamics figures so they are suitable for a report or presentation.
4. Keep the final narrative centered on strategic preservation failure, and explicitly downgrade the dynamics claim to an ongoing investigation unless stronger evidence is produced.

## 8. Bottom Line / 总结

中文：
结论是：这个项目已经可以写一份可信的中期或初步报告。现阶段最扎实的主线是，mixing-enhanced exact game 与若干压缩近似之间确实存在 payoff distortion 与 equilibrium distortion，而且这种失真在若干 high-conflict 场景下相当明显。需要谨慎的是，dynamics 证据还弱，`sampling` baseline 的定义也还需要进一步校准。因此，当前最合适的写法是“初步支持 proposal 的核心命题，但仍需要补充动态实验与 baseline 修正来完成最终论证”。

English:
The bottom line is that this project is already ready for a credible mid-project or preliminary report. The strongest current storyline is that the mixing-enhanced exact game does exhibit payoff distortion and equilibrium distortion relative to several compressed approximations, and that this distortion becomes substantial in a number of high-conflict settings. The main cautions are that the dynamics evidence is still weak and the `sampling` baseline still needs calibration. Therefore, the most defensible current framing is that the evidence provides preliminary support for the proposal's central claim, while further dynamics experiments and baseline refinement are still needed for a final argument.
