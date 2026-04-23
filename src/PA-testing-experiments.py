#!/usr/bin/env python3
"""Reproduce the figures for Principal-Agent Hypothesis Testing."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats


# Plotting palette.
BLUE = (40 / 255, 103 / 255, 178 / 255)
RED = (174 / 255, 4 / 255, 14 / 255)
GOLD = (218 / 255, 165 / 255, 32 / 255)

# Single-period welfare design.
ALPHA = 0.05
STATUS_QUO_POWER = 0.80
FALSE_APPROVAL_COST = 0.07
SEVERITY_CASES = (
    ("Low", 0.04),
    ("High", 0.71),
)
R_OVER_C_VALUES = {
    "small": 5,
    "large": 50,
}
SEVERITY_COLOR = {"Low": RED, "High": BLUE}

# Multi-round dynamic-programming design.
MAX_ROUNDS = 5
STAGE_COST = 0.1
TOP_THETA = 1.645
ALTERNATIVES = np.array([0.5, 1.0, 1.645, 2.49])
PROFIT_CAPS = (1.0, 5.0)


@dataclass
class DiscreteUtility:
    """Piecewise-linear utility on a fixed grid."""

    knots: np.ndarray
    heights: np.ndarray

    def __post_init__(self) -> None:
        self.knots = np.asarray(self.knots, dtype=float)
        self.heights = least_concave_majorant(self.knots, np.asarray(self.heights, dtype=float))
        self.slopes = np.r_[np.diff(self.heights) / np.diff(self.knots), 0.0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.knots, self.heights)


@dataclass
class StepLicense:
    """Step-function license update in the Gaussian statistic."""

    values: np.ndarray
    thresholds: np.ndarray

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        self.thresholds = np.asarray(self.thresholds, dtype=float)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        ind = np.searchsorted(self.thresholds, y)
        ind = np.minimum(ind, len(self.values) - 1)
        return self.values[ind]


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Reproduce the PA testing experiment figures.")
    parser.add_argument(
        "--use-tex",
        action="store_true",
        help="Use LaTeX for Matplotlib text rendering when available.",
    )
    return parser.parse_args()


def configure_matplotlib(use_tex: bool = False) -> None:
    """Set common plot styling."""
    if use_tex and shutil.which("latex") is None:
        print("LaTeX was requested but no `latex` executable was found; using mathtext.")
        use_tex = False

    params = {
        "axes.linewidth": 2,
        "font.size": 22,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "text.usetex": use_tex,
    }
    if use_tex:
        params.update(
            {
                "font.serif": ["Computer Modern"],
                "text.latex.preamble": r"\usepackage{amsmath,amsfonts}",
            }
        )
    plt.rcParams.update(params)


def clean_axes(ax: plt.Axes, *, bottom_at_zero: bool = True, left_at_zero: bool = False) -> None:
    """Remove top/right spines and optionally anchor axes at zero."""
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    if bottom_at_zero:
        ax.spines["bottom"].set_position("zero")
    if left_at_zero:
        ax.spines["left"].set_position("zero")


def save_pdf(fig: plt.Figure, filename: str) -> None:
    """Save and close a PDF figure."""
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Place a panel label just outside the axes."""
    ax.text(-0.16, 1.12, label, transform=ax.transAxes, ha="left", va="top", clip_on=False)


# -----------------------------------------------------------------------------
# Figure 1: welfare under status quo and incentive alignment
# -----------------------------------------------------------------------------


def calibrated_theta1(alpha: float = ALPHA, power: float = STATUS_QUO_POWER) -> float:
    """Calibrate theta_1 to match the status-quo power."""
    return stats.norm.isf(alpha) - stats.norm.isf(power)


def incentive_aligned_power(r_over_c: float, theta1: float) -> float:
    """Power of the incentive-aligned license."""
    cutoff = 1 / r_over_c
    return stats.norm.sf(stats.norm.isf(cutoff) - theta1)


def status_quo_welfare(pi0: np.ndarray, c2: float, r_over_c: float) -> np.ndarray:
    """Principal utility under the status quo."""
    null_agents_enter = float(r_over_c * ALPHA > 1)
    true_approval_benefit = (1 - pi0) * STATUS_QUO_POWER * c2
    false_approval_cost = pi0 * ALPHA * FALSE_APPROVAL_COST * null_agents_enter
    return true_approval_benefit - false_approval_cost


def incentive_aligned_welfare(pi0: np.ndarray, c2: float, r_over_c: float, theta1: float) -> np.ndarray:
    """Principal utility under incentive alignment."""
    return (1 - pi0) * incentive_aligned_power(r_over_c, theta1) * c2


def gen_figure_1() -> None:
    """Generate the two welfare PDFs."""
    theta1 = calibrated_theta1()
    pi0 = np.linspace(0, 1, 1000)

    for market, r_over_c in R_OVER_C_VALUES.items():
        fig, ax = plt.subplots(figsize=(8, 8))

        for severity, c2 in SEVERITY_CASES:
            color = SEVERITY_COLOR[severity]
            ax.plot(pi0, status_quo_welfare(pi0, c2, r_over_c), color=color, linestyle="--")
            ax.plot(pi0, incentive_aligned_welfare(pi0, c2, r_over_c, theta1), color=color, linestyle="-")

        is_small_market = market == "small"
        ax.set_title("Low profit" if is_small_market else "High profit")
        ax.text(0.93, 0.048, "(a)" if is_small_market else "(b)")
        ax.set_xlabel(r"Null proportion $\pi_0$")
        ax.set_ylabel("Principal's utility")
        ax.set_xticks([0.9, 0.95, 1.0])
        ax.set_xticklabels([".9", ".95", "1"])
        ax.set_xlim(0.945, 1)
        ax.set_ylim(-0.01, 0.05)
        ax.grid(alpha=0.2)
        clean_axes(ax, bottom_at_zero=True)

        if is_small_market:
            mechanism_handles = [
                Line2D([0], [0], color="black", linestyle="--", label="Status quo"),
                Line2D([0], [0], color="black", linestyle="-", label="Incentive-aligned"),
            ]
            severity_handles = [
                Line2D([0], [0], color=SEVERITY_COLOR["High"], linestyle="-", label="High"),
                Line2D([0], [0], color=SEVERITY_COLOR["Low"], linestyle="-", label="Low"),
            ]
            mechanism_legend = ax.legend(handles=mechanism_handles, loc="upper right", title="System")
            severity_legend = ax.legend(handles=severity_handles, loc="center right", title="Severity")
            ax.add_artist(mechanism_legend)
            ax.add_artist(severity_legend)

        filename = "../output/1a.pdf" if is_small_market else "../output/1b.pdf"
        save_pdf(fig, filename)



# -----------------------------------------------------------------------------
# Figure 2: capped e-value payoff
# -----------------------------------------------------------------------------


def capped_likelihood_ratio_quantiles(
    theta: float,
    n_max: int,
    cap: float,
    probabilities: tuple[float, float, float] = (0.25, 0.50, 0.75),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantiles of min(E_n, cap) under the alternative."""
    n = np.arange(1, n_max + 1)
    mean_log_e = 0.5 * theta**2 * n
    sd_log_e = theta * np.sqrt(n)
    log_cap = np.log(cap)

    quantiles = []
    for p in probabilities:
        log_q = mean_log_e + sd_log_e * stats.norm.ppf(p)
        quantiles.append(np.where(log_q >= log_cap, cap, np.exp(log_q)))

    return n, quantiles[0], quantiles[1], quantiles[2]


def gen_figure_2() -> None:
    """Generate 2.pdf."""
    theta = 0.20
    n_max = 1000
    cap = 50

    n, q25, median, q75 = capped_likelihood_ratio_quantiles(theta, n_max, cap)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(n, median, color=BLUE, linestyle="-", label="Median")
    ax.fill_between(n, q25 - 0.1, q75 + 0.2, color=BLUE, alpha=0.2, label="IQR")
    ax.legend(loc="center right")

    clean_axes(ax, bottom_at_zero=True, left_at_zero=True)
    ax.set_xlabel(r"Sample size $n$")
    ax.set_ylabel(r"$\min(E, 50)$")
    ax.set_ylim(0, 51)
    save_pdf(fig, "../output/2.pdf")


# -----------------------------------------------------------------------------
# Figure 3: dynamic multi-round profit license
# -----------------------------------------------------------------------------


def weighted_isotonic_increasing(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Fit increasing block means by PAVA."""
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)

    levels: list[float] = []
    block_weights: list[float] = []
    block_lengths: list[int] = []

    for value, weight in zip(y, weights):
        levels.append(float(value))
        block_weights.append(float(weight))
        block_lengths.append(1)

        while len(levels) >= 2 and levels[-2] > levels[-1]:
            new_weight = block_weights[-2] + block_weights[-1]
            new_level = (levels[-2] * block_weights[-2] + levels[-1] * block_weights[-1]) / new_weight
            new_length = block_lengths[-2] + block_lengths[-1]
            levels[-2:] = [new_level]
            block_weights[-2:] = [new_weight]
            block_lengths[-2:] = [new_length]

    return np.repeat(np.asarray(levels), np.asarray(block_lengths))


def greatest_convex_minorant(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate the greatest convex minorant at x."""
    widths = np.diff(x)
    slopes = np.diff(y) / widths
    fitted_slopes = weighted_isotonic_increasing(slopes, widths)
    return np.r_[0.0, np.cumsum(fitted_slopes * widths)]


def least_concave_majorant(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate the least concave majorant at x."""
    return -greatest_convex_minorant(x, -y) + y[0]


def optimal_e_value(
    utility: DiscreteUtility,
    theta: float,
    variance: float = 1.0,
    tol: float = 1e-10,
) -> tuple[StepLicense, float]:
    """Solve the one-period Gaussian e-value problem."""
    knots = utility.knots.copy()
    heights = least_concave_majorant(knots, utility.heights.copy())
    slopes = np.r_[np.diff(heights) / np.diff(knots), 0.0]

    flat = np.flatnonzero(slopes <= 0)
    first_flat = int(flat[0]) if len(flat) else len(slopes) - 1
    slopes = slopes[:first_flat]
    heights = heights[: first_flat + 1]
    knots = knots[: first_flat + 1]

    unique_slopes, locs = np.unique(np.round(slopes, decimals=8), return_index=True)
    slopes = unique_slopes[::-1]
    locs = locs[::-1]
    locs = np.r_[locs, first_flat]
    heights = heights[locs]
    knots = knots[locs]

    sd = np.sqrt(variance)

    def null_mean(lam: float) -> float:
        cdf = stats.norm.cdf(theta / 2 - (variance / theta) * np.log(slopes / lam), scale=sd)
        cdf = np.r_[cdf, 1.0]
        return float(np.dot(knots[1:], np.diff(cdf)))

    left, right = 1e-10, 1.0
    while null_mean(left) < 1.0:
        right = left
        left /= 2
    while null_mean(right) > 1.0:
        left = right
        right *= 2

    while right - left > tol:
        mid = (left + right) / 2
        if null_mean(mid) > 1.0:
            left = mid
        else:
            right = mid

    lam = (left + right) / 2
    thresholds = theta / 2 - (variance / theta) * np.log(slopes / lam)
    e_value = StepLicense(knots, thresholds)

    alt_cdf = stats.norm.cdf(-theta / 2 - (variance / theta) * np.log(slopes / lam), scale=sd)
    alt_cdf = np.r_[alt_cdf, 1.0]
    expected_utility = float(np.dot(heights[1:], np.diff(alt_cdf)))

    return e_value, expected_utility


def solve_dynamic_program(
    theta: float,
    profit_cap: float,
    license_grid: np.ndarray,
    max_rounds: int = MAX_ROUNDS,
    stage_cost: float = STAGE_COST,
) -> tuple[dict[int, list[StepLicense]], dict[int, np.ndarray]]:
    """Compute the optimal finite-grid multi-round policy."""
    stop_value = {
        t: np.minimum(license_grid, profit_cap) - stage_cost * t
        for t in range(max_rounds + 1)
    }

    value = {max_rounds: stop_value[max_rounds]}
    stop = {max_rounds: np.ones(len(license_grid), dtype=bool)}
    strategy: dict[int, list[StepLicense]] = defaultdict(list)

    for t in range(max_rounds - 1, -1, -1):
        continue_value = np.zeros(len(license_grid))

        for state_index, current_license in enumerate(license_grid):
            if current_license >= profit_cap or np.isclose(current_license, profit_cap):
                break

            e_grid = license_grid / (current_license + stage_cost)
            utility = DiscreteUtility(e_grid, value[t + 1])
            e_value, expected_value = optimal_e_value(utility, theta)
            continue_value[state_index] = expected_value
            strategy[t].append(StepLicense(e_value.values * (current_license + stage_cost), e_value.thresholds))

        value[t] = np.maximum(continue_value, stop_value[t])
        stop[t] = stop_value[t] >= continue_value

    return strategy, stop


def step_license_probabilities(
    step_license: StepLicense,
    theta: float,
    variance: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact mass for each step-license value."""
    values = step_license.values
    thresholds = step_license.thresholds

    if len(values) == 1:
        return values, np.array([1.0])

    cdf = stats.norm.cdf(thresholds, loc=theta, scale=np.sqrt(variance))
    probs = np.empty(len(values))
    probs[0] = cdf[0]
    probs[1:-1] = np.diff(cdf)
    probs[-1] = 1 - cdf[-1]
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return values, probs


def nearest_grid_index(grid: np.ndarray, value: float) -> int:
    """Find the nearest grid index."""
    return int(np.argmin(np.abs(grid - value)))


def forward_distribution(
    strategy: dict[int, list[StepLicense]],
    stop: dict[int, np.ndarray],
    license_grid: np.ndarray,
    theta: float,
    profit_cap: float,
    max_rounds: int = MAX_ROUNDS,
) -> list[tuple[float, int, float]]:
    """Propagate the terminal distribution exactly."""
    running: dict[int, float] = {0: 1.0}
    terminal: list[tuple[float, int, float]] = []

    for t in range(max_rounds):
        next_running: DefaultDict[int, float] = defaultdict(float)

        for state_index, state_probability in running.items():
            current_license = license_grid[state_index]

            if stop[t][state_index]:
                terminal.append((min(current_license, profit_cap), t + 1, state_probability))
                continue

            values, probs = step_license_probabilities(strategy[t][state_index], theta)
            for next_license, transition_probability in zip(values, probs):
                probability = state_probability * transition_probability
                next_index = nearest_grid_index(license_grid, next_license)
                terminal_license = min(float(next_license), profit_cap)

                if next_license >= profit_cap or t + 1 == max_rounds:
                    terminal.append((terminal_license, t + 1, probability))
                else:
                    next_running[next_index] += probability

        running = dict(next_running)

    for state_index, state_probability in running.items():
        terminal.append((min(license_grid[state_index], profit_cap), max_rounds, state_probability))

    return terminal


def one_period_success_probability(theta: float, total_cost: float, profit_cap: float, variance: float) -> float:
    """Compute success probability for the capped one-period license."""
    threshold = stats.norm.isf(total_cost / profit_cap, scale=np.sqrt(variance))
    return float(stats.norm.sf(threshold, loc=theta, scale=np.sqrt(variance)))


def one_period_profit(theta: float, total_cost: float, profit_cap: float, variance: float) -> float:
    """Compute expected one-period profit."""
    return profit_cap * one_period_success_probability(theta, total_cost, profit_cap, variance) - total_cost


def terminal_license_probabilities(terminal: list[tuple[float, int, float]], cap: float) -> np.ndarray:
    """Aggregate terminal license mass at 0 and R."""
    return np.array(
        [
            sum(prob for license_value, _, prob in terminal if np.isclose(license_value, 0.0)),
            sum(prob for license_value, _, prob in terminal if np.isclose(license_value, cap)),
        ]
    )


def terminal_stage_probabilities(terminal: list[tuple[float, int, float]]) -> np.ndarray:
    """Aggregate terminal mass by stopping round."""
    return np.array([sum(prob for _, stage, prob in terminal if stage == k) for k in range(1, MAX_ROUNDS + 1)])


def expected_profit(terminal: list[tuple[float, int, float]]) -> float:
    """Compute expected terminal license minus trial costs."""
    return float(sum((license_value - STAGE_COST * stage) * prob for license_value, stage, prob in terminal))


def check_probabilities(name: str, probs: np.ndarray, tol: float = 1e-8) -> None:
    """Ensure a probability vector sums to one."""
    total = float(np.sum(probs))
    if not np.isclose(total, 1.0, atol=tol):
        raise RuntimeError(f"{name} probabilities sum to {total:.12f}, not 1.")


def save_panel_3a(output_path: Path, terminal: list[tuple[float, int, float]]) -> None:
    """Save panel 3a."""
    cap = 5.0
    x = np.array([0.0, cap])

    prob_big = one_period_success_probability(TOP_THETA, MAX_ROUNDS * STAGE_COST, cap, variance=1 / MAX_ROUNDS)
    prob_small = one_period_success_probability(TOP_THETA, STAGE_COST, cap, variance=1.0)
    probs_big = np.array([1 - prob_big, prob_big])
    probs_small = np.array([1 - prob_small, prob_small])
    probs_dp = terminal_license_probabilities(terminal, cap)

    check_probabilities("Panel 3a one-period, 5x data", probs_big)
    check_probabilities("Panel 3a one-period", probs_small)
    check_probabilities("Panel 3a five periods", probs_dp)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x - 0.5, probs_big, color=RED, width=0.5, zorder=4, label=r"One period, $5\times$ data")
    ax.bar(x, probs_small, color=GOLD, width=0.5, zorder=5, label="One period")
    ax.bar(x + 0.5, probs_dp, color=BLUE, width=0.5, zorder=3, label="Five periods")

    ax.grid(zorder=0, alpha=0.5)
    ax.set_xticks([0, cap], ["0", r"$R=5$"])
    ax.set_xlabel("License value")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.05)
    add_panel_label(ax, "(a)")
    ax.legend(loc="upper left", fontsize=18)
    save_pdf(fig, output_path.as_posix())


def save_panel_3b(output_path: Path, terminal: list[tuple[float, int, float]]) -> None:
    """Save panel 3b."""
    probs = terminal_stage_probabilities(terminal)
    check_probabilities("Panel 3b", probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(np.arange(1, MAX_ROUNDS + 1), probs, color=BLUE, zorder=3)
    ax.set_xticks(np.arange(1, MAX_ROUNDS + 1))
    ax.set_xlabel("Number of rounds ran")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.05)
    ax.grid(zorder=0, alpha=0.5)
    add_panel_label(ax, "(b)")
    save_pdf(fig, output_path.as_posix())


def compute_profit_curves() -> dict[float, dict[str, np.ndarray]]:
    """Compute profit curves for the three strategies."""
    curves: dict[float, dict[str, np.ndarray]] = {}

    for cap in PROFIT_CAPS:
        grid = np.linspace(0, 10, 101)
        five_periods = []
        one_period = []
        one_period_big = []

        for theta in ALTERNATIVES:
            strategy, stop = solve_dynamic_program(theta, cap, grid)
            terminal = forward_distribution(strategy, stop, grid, theta, cap)
            check_probabilities(
                f"terminal distribution R={cap}, theta={theta}",
                np.array([p for _, _, p in terminal]),
            )

            five_periods.append(expected_profit(terminal))
            one_period.append(one_period_profit(theta, STAGE_COST, cap, variance=1.0))
            one_period_big.append(one_period_profit(theta, MAX_ROUNDS * STAGE_COST, cap, variance=1 / MAX_ROUNDS))

        curves[cap] = {
            "Five periods": np.array(five_periods),
            "One period": np.array(one_period),
            r"One period, $5\times$ data": np.array(one_period_big),
        }

    return curves


def save_profit_panel(output_path: Path, cap: float, curves: dict[str, np.ndarray], panel_label: str) -> None:
    """Save a Figure 3 profit panel."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {
        "Five periods": BLUE,
        "One period": GOLD,
        r"One period, $5\times$ data": RED,
    }

    for label in ["Five periods", "One period", r"One period, $5\times$ data"]:
        ax.plot(ALTERNATIVES, curves[label], ".-", color=colors[label], label=label)

    ax.set_title(r"Agent's profit ($R = %.1f$)" % cap)
    ax.set_xlabel(r"Alternative $\theta_1$")
    ax.set_ylabel(r"$\mathbb{E}[f(Z) \wedge R - C]$")
    ax.set_yticks([i * cap / 5 for i in range(6)])
    ax.grid()
    if cap == 1.0:
        ax.legend(loc="lower right", fontsize=18)
    add_panel_label(ax, panel_label)
    save_pdf(fig, output_path.as_posix())


def gen_figure_3() -> None:
    """Generate panels 3a.pdf through 3d.pdf."""
    top_grid = np.linspace(0, 10, 99)
    top_strategy, top_stop = solve_dynamic_program(TOP_THETA, 5.0, top_grid)
    top_terminal = forward_distribution(top_strategy, top_stop, top_grid, TOP_THETA, 5.0)
    check_probabilities("top terminal distribution", np.array([p for _, _, p in top_terminal]))

    save_panel_3a(Path("../output/3a.pdf"), top_terminal)
    save_panel_3b(Path("../output/3b.pdf"), top_terminal)

    curves = compute_profit_curves()
    save_profit_panel(Path("../output/3c.pdf"), 1.0, curves[1.0], "(c)")
    save_profit_panel(Path("../output/3d.pdf"), 5.0, curves[5.0], "(d)")


if __name__ == "__main__":
    args = parse_args()
    configure_matplotlib(use_tex=args.use_tex)
    gen_figure_1()
    gen_figure_2()
    gen_figure_3()
