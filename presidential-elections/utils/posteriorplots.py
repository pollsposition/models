from typing import List

import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax

colors = sns.color_palette(as_cmap=True)


def retrodictive_plot(
    trace: arviz.InferenceData,
    parties_complete: List[str],
    polls_train: pd.DataFrame,
    group: str = "posterior",
):
    if len(parties_complete) % 2 == 0:
        fig, axes = plt.subplots(
            len(parties_complete) // 2, 2, figsize=(12, 15), sharey=True
        )
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(
            len(parties_complete) // 2 + 1, 2, figsize=(12, 15), sharey=True
        )
        axes = axes.ravel()
        axes[-1].remove()

    N = trace.constant_data["observed_N"]
    if group == "posterior":
        pp = trace.posterior_predictive
        POST_MEDIANS = pp["latent_popularity"].median(("chain", "draw"))
        STACKED_POP = pp["latent_popularity"].stack(sample=("chain", "draw"))

    elif group == "prior":
        prior = trace.prior
        pp = trace.prior_predictive
        POST_MEDIANS = prior["latent_popularity"].median(("chain", "draw"))
        STACKED_POP = prior["latent_popularity"].stack(sample=("chain", "draw"))

    POST_MEDIANS_MULT = (pp["N_approve"] / N).median(("chain", "draw"))
    HDI = arviz.hdi(pp)["N_approve"] / N
    SAMPLES = np.random.choice(range(len(STACKED_POP.sample)), size=1000)

    for i, p in enumerate(parties_complete):
        if group == "posterior":
            axes[i].plot(
                polls_train["date"],
                polls_train[p] / N,
                "o",
                color=colors[i],
                alpha=0.4,
            )
        for sample in SAMPLES:
            axes[i].plot(
                polls_train["date"],
                STACKED_POP.sel(parties_complete=p).isel(sample=sample),
                color=colors[i],
                alpha=0.05,
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
            POST_MEDIANS_MULT.sel(parties_complete=p),
            color="black",
            ls="--",
            lw=3,
            label="Noisy Popularity",
        )
        axes[i].plot(
            polls_train["date"],
            POST_MEDIANS.sel(parties_complete=p),
            color="grey",
            lw=3,
            label="Latent Popularity",
        )
        axes[i].tick_params(axis="x", labelrotation=45, labelsize=10)
        axes[i].set(title=p.title())
        axes[i].legend(fontsize=9, ncol=2)
    plt.suptitle(f"{group.title()} Predictive Check", fontsize=16, fontweight="bold")


def predictive_plot(
    idata: arviz.InferenceData,
    parties_complete: List[str],
    election_date: str,
    polls_train: pd.DataFrame,
    polls_test: pd.DataFrame,
    results: pd.DataFrame = None,
    # test_cutoff: pd.Timedelta = None,
    hdi: bool = False,
):
    election_date = pd.to_datetime(election_date)
    # results = results[results.dateelection == election_date]
    new_dates = idata.predictions_constant_data["observations"].to_index()
    predictions = idata.predictions.sel(
        observations=new_dates[new_dates.year == int(f"{election_date.year}")]
    )
    # constant_data = idata.predictions_constant_data.sel(
    #     observations=new_dates[new_dates.year == int(f"{election_date.year}")]
    # )

    # if test_cutoff is None:
    #     test_cutoff = election_date - pd.Timedelta(2, "D")
    # else:
    #     test_cutoff = election_date - test_cutoff

    if len(parties_complete) % 2 == 0:
        fig, axes = plt.subplots(len(parties_complete) // 2, 2, figsize=(12, 15))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(len(parties_complete) // 2 + 1, 2, figsize=(12, 15))
        axes = axes.ravel()
        axes[-1].remove()

    # post_N = constant_data["observed_N"]
    POST_MEDIANS = predictions["latent_popularity"].median(("chain", "draw"))
    STACKED_POP = predictions["latent_popularity"].stack(sample=("chain", "draw"))
    HDI_POP_83 = arviz.hdi(predictions, hdi_prob=0.83)["latent_popularity"]
    SAMPLES = np.random.choice(range(len(STACKED_POP.sample)), size=1000)
    POST_MEDIANS_MULT = predictions["noisy_popularity"].median(("chain", "draw"))
    # HDI_MULT = arviz.hdi(predictions, hdi_prob=0.83)["N_approve"] / post_N

    if election_date.year == 2022:
        TITLES = dict(zip(
                parties_complete,
                [
                    "Mélenchon",
                    "Hidalgo",
                    "Jadot",
                    "Macron",
                    "Pécresse",
                    "Le Pen",
                    "Zemmour",
                    "Autre",
                ],
            ))

    for i, p in enumerate(parties_complete):
        # axes[i].fill_between(
        #     predictions["observations"],
        #     HDI_MULT.sel(parties_complete=p, hdi="lower"),
        #     HDI_MULT.sel(parties_complete=p, hdi="higher"),
        #     color=colors[i],
        #     alpha=0.2,
        #     label="5 in 6 chance Polls",
        # )
        if hdi:
            axes[i].fill_between(
                predictions["observations"],
                HDI_POP_83.sel(parties_complete=p, hdi="lower"),
                HDI_POP_83.sel(parties_complete=p, hdi="higher"),
                color=colors[i],
                alpha=0.5,
                label="5 in 6 chance",
            )
        else:
            for sample in SAMPLES:
                axes[i].plot(
                    predictions["observations"],
                    STACKED_POP.sel(parties_complete=p).isel(sample=sample),
                    color=colors[i],
                    alpha=0.05,
                )
        axes[i].plot(
            predictions["observations"],
            POST_MEDIANS.sel(parties_complete=p),
            lw=3,
            color="black",
            label="Latent Popularity",
        )
        axes[i].plot(
            predictions["observations"],
            POST_MEDIANS_MULT.sel(parties_complete=p),
            ls="--",
            color="grey",
            label="Noisy Popularity",
        )
        axes[i].plot(
            polls_train["date"],
            polls_train[p] / polls_train["samplesize"],
            "o",
            color="black",
            alpha=0.4,
            label="Observed polls",
        )
        if not polls_test.empty:
            axes[i].plot(
                polls_test["date"],
                polls_test[p] / polls_test["samplesize"],
                "x",
                color="black",
                alpha=0.4,
                label="Unobserved polls",
            )
        # axes[i].axvline(
        #     x=test_cutoff,
        #     ymin=-0.01,
        #     ymax=1.0,
        #     ls="--",
        #     c="k",
        #     alpha=0.6,
        #     label="Test cutoff",
        # )
        axes[i].axvline(
            x=election_date,
            ymin=-0.01,
            ymax=1.0,
            ls=":",
            c="k",
            alpha=0.6,
            label="Election Day",
        )
        # axes[i].axhline(
        #     y=(results[p] / 100).to_numpy(),
        #     xmin=-0.01,
        #     xmax=1.0,
        #     ls="-.",
        #     c="k",
        #     alpha=0.6,
        #     label="Result",
        # )
        axes[i].axhline(
            y=softmax(predictions["party_intercept"].mean(("chain", "draw"))).sel(
                parties_complete=p
            ),
            xmin=-0.01,
            xmax=1.0,
            ls="-.",
            c=colors[i],
            label="Historical Average",
        )
        axes[i].tick_params(axis="x", labelrotation=45, labelsize=10)
        if TITLES:
            axes[i].set(title=TITLES[p], ylim=(-0.01, 0.4))
        else:
            axes[i].set(title=p.title(), ylim=(-0.01, 0.4))
        axes[i].legend(fontsize=9, ncol=3)
