from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_data_and_coords(
    election_date: str,
    parties_complete: List[str] = [
        "farleft",
        "left",
        "green",
        "center",
        "right",
        "farright",
        "other",
    ],
    test_cutoff: pd.Timedelta = None,
):
    election_date = pd.to_datetime(election_date)
    polls = load_data(election_date)
    results, polls = format_data(polls, parties_complete)
    (
        polls_train,
        polls_test,
        observed_days_id,
        estimated_days,
        whole_timeline,
    ) = train_split_and_idx_vars(polls, election_date, test_cutoff)
    pollster_id, COORDS = dims_and_coords(polls_train, parties_complete, whole_timeline)
    plot_check(polls, parties_complete)

    return (
        polls_train,
        polls_test,
        results,
        observed_days_id,
        estimated_days,
        pollster_id,
        COORDS,
    )


def load_data(election_date: str):
    polls = pd.read_csv(
        "../../data/polls_1st_round/tour1_complet_unitedfl.csv",
        index_col=0,
        parse_dates=["dateelection", "date"],
    )

    return (
        polls[
            (polls.dateelection == election_date)
            & (polls.date >= f"{election_date.year}-01")
        ]
        .drop(
            [
                "type",
                "dateelection",
                "abstention",
                "undecided",
            ],
            axis=1,
        )
        .set_index(["date", "sondage", "samplesize"])
        .sort_index()
    )


def format_data(polls: pd.DataFrame, parties_complete: List[str]):
    polls = polls.rename(
        columns={col: col.split("nb")[1] for col in polls if col.startswith("nb")}
    )[parties_complete[:-1]]

    # compute other category
    polls["other"] = 100 - polls.sum(1)
    np.testing.assert_allclose(polls.sum(1).values, 100)

    # isolate results
    polls = polls.reset_index()
    results = polls[polls.sondage == "result"]
    polls = polls[polls.sondage != "result"].set_index(["date", "sondage"])

    # cast as multinomial obs
    polls[parties_complete] = (
        (polls[parties_complete] / 100)
        .mul(polls["samplesize"], axis=0)
        .round()
        .fillna(0)
        .astype(int)
    )
    polls["samplesize"] = polls[parties_complete].sum(1)

    return results, polls.reset_index()


def train_split_and_idx_vars(polls: pd.DataFrame, election_date: pd.Timestamp, test_cutoff: pd.Timedelta = None):
    if test_cutoff is None:
        test_cutoff = election_date - pd.Timedelta(2, "D")
    else:
        test_cutoff = election_date - test_cutoff
    
    whole_timeline = pd.date_range(polls.date[0], election_date, freq="D")

    polls_train = polls[polls.date <= test_cutoff]
    polls_test = polls[polls.date > test_cutoff]

    observed_days_idx = dates_to_idx(polls_train["date"]).astype(int)
    estimated_days = dates_to_idx(whole_timeline).astype(int)
    # observed_days_distances = dates_to_idx(polls_train.date.unique()).astype(int)
    # observed_days_idx, COORDS["observed_days"] = polls_train.date.factorize()

    return polls_train, polls_test, observed_days_idx, estimated_days, whole_timeline


def dates_to_idx(timelist):
    """Convert datetimes to numbers in reference to a given date"""

    reference_time = timelist[0]
    t = (timelist - reference_time) / np.timedelta64(1, "D")

    return np.asarray(t)


def dims_and_coords(
    polls_train: pd.DataFrame,
    parties_complete: List[str],
    whole_timeline: pd.DatetimeIndex,
):
    COORDS = {
        "estimated_days": whole_timeline,
        "observed_days": polls_train["date"],
        "observations": polls_train.set_index(["date", "sondage", "samplesize"]).index,
        "parties": parties_complete[:-1],
        "parties_complete": parties_complete,
    }
    pollster_id, COORDS["pollsters"] = polls_train["sondage"].factorize(sort=True)

    return pollster_id, COORDS


def plot_check(polls: pd.DataFrame, parties_complete: List[str]):
    fig, ax = plt.subplots(figsize=(12, 5))
    for p in parties_complete:
        ax.plot(polls["date"], polls[p] / polls["samplesize"], "o", label=p, alpha=0.4)
    ax.legend(ncol=4, frameon=True, loc="upper right");