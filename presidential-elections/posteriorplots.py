from typing import List

import arviz
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def retrodictive_plot(
    trace: arviz.InferenceData,
    parties_complete: List[str],
    polls_train: pd.DataFrame,
    group: str = "posterior",
):
    if len(parties_complete) % 2 == 0:
        fig, axes = plt.subplots(
            len(parties_complete) // 2, 2, figsize=(12, 12), sharey=True
        )
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(
            len(parties_complete) // 2 + 1, 2, figsize=(12, 15), sharey=True
        )
        axes = axes.ravel()
        axes[-1].remove()

    colors = sns.color_palette("rocket", n_colors=len(parties_complete), as_cmap=False)

    N = trace.constant_data["observed_N"]
    if group == "posterior":
        var = trace.posterior_predictive
    elif group == "prior":
        var = trace.prior_predictive
    POST_MEANS = (var["N_approve"] / N).mean(("chain", "draw"))
    HDI = arviz.hdi(var)["N_approve"] / N

    for i, p in enumerate(parties_complete):
        if group == "posterior":
            axes[i].plot(
                polls_train["date"],
                polls_train[p] / N,
                "o",
                color=colors[i],
                alpha=0.4,
            )
        axes[i].fill_between(
            polls_train["date"],
            HDI.sel(parties_complete=p, hdi="lower"),
            HDI.sel(parties_complete=p, hdi="higher"),
            color=colors[i],
            alpha=0.4,
        )
        axes[i].plot(
            polls_train["date"],
            POST_MEANS.sel(parties_complete=p),
            color=colors[i],
        )
        axes[i].tick_params(axis="x", labelrotation=45, labelsize=10)
        axes[i].set(title=p.title())
    plt.suptitle(f"{group.title()} Predictive Check", fontsize=16, fontweight="bold")
    
    
def predictive_plot(
    predictions: arviz.InferenceData,
    parties_complete: List[str],
    polls_train: pd.DataFrame,
    polls_test: pd.DataFrame,
    results: pd.DataFrame,
    election_date: str,
    test_cutoff: pd.Timedelta = None,
):
    election_date = pd.to_datetime(election_date)
    if test_cutoff is None:
        test_cutoff = election_date - pd.Timedelta(2, "D")
    else:
        test_cutoff = election_date - test_cutoff

    if len(parties_complete) % 2 == 0:
        fig, axes = plt.subplots(
            len(parties_complete) // 2, 2, figsize=(12, 12), sharey=True
        )
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(
            len(parties_complete) // 2 + 1, 2, figsize=(12, 15), sharey=True
        )
        axes = axes.ravel()
        axes[-1].remove()

    colors = sns.color_palette("rocket", n_colors=len(parties_complete), as_cmap=False)

    post_N = predictions.predictions_constant_data["observed_N"]
    POST_MEANS = predictions.predictions["popularity"].mean(("chain", "draw"))
    HDI_POP = arviz.hdi(predictions.predictions)["popularity"]
    HDI_MULT = arviz.hdi(predictions.predictions)["N_approve"] / post_N

    for i, p in enumerate(parties_complete):
        axes[i].plot(
            polls_train["date"],
            polls_train[p] / polls_train["samplesize"],
            "o",
            color=colors[i],
            alpha=0.4,
            label="Observed polls",
        )
        if not polls_test.empty:
            axes[i].plot(
                polls_test["date"],
                polls_test[p] / polls_test["samplesize"],
                "x",
                color="red",
                alpha=0.4,
                label="Unobserved polls",
            )
        axes[i].plot(
            election_date,
            results[p] / 100,
            "s",
            color=colors[i],
            alpha=0.8,
            label="Result",
        )
        axes[i].fill_between(
            predictions.predictions["estimated_days"],
            HDI_MULT.sel(parties_complete=p, hdi="lower"),
            HDI_MULT.sel(parties_complete=p, hdi="higher"),
            color=colors[i],
            alpha=0.2,
            label="HDI Polls",
        )
        axes[i].fill_between(
            predictions.predictions["estimated_days"],
            HDI_POP.sel(parties_complete=p, hdi="lower"),
            HDI_POP.sel(parties_complete=p, hdi="higher"),
            color=colors[i],
            alpha=0.5,
            label="HDI Popularity",
        )
        axes[i].plot(
            predictions.predictions["estimated_days"],
            POST_MEANS.sel(parties_complete=p),
            color=colors[i],
            label="Mean Popularity",
        )
        axes[i].axvline(
            x=test_cutoff,
            ymin=-0.01,
            ymax=1.0,
            ls="--",
            c="k",
            alpha=0.4,
            label="Test cutoff",
        )
        axes[i].axvline(
            x=election_date,
            ymin=-0.01,
            ymax=1.0,
            ls=":",
            c="k",
            alpha=0.4,
            label="Election Day",
        )
        axes[i].tick_params(axis="x", labelrotation=45, labelsize=10)
        axes[i].set(title=p.title(), ylim=(-0.01, 0.4))
        axes[i].legend(fontsize=10, ncol=3)