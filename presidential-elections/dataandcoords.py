from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_data_and_coords(
    test_cutoff: pd.Timedelta = None,
):
    parties_complete = [
        "farleft",
        "left",
        "green",
        "center",
        "right",
        "farright",
        "other",
    ]
    polls = load_data()
    results, polls = format_data(polls, parties_complete)
    (
        polls_train,
        polls_test,
    ) = train_split_and_idx_vars(polls, test_cutoff)
    pollster_id, countdown_id, election_id, COORDS = dims_and_coords(polls_train, parties_complete)
  #  plot_check(polls, parties_complete)

    return (
        polls_train,
        polls_test,
        results,
        pollster_id,
        countdown_id, 
        election_id,
        COORDS,
    )


def load_data():
    polls = pd.read_csv(
        "../../data/polls_1st_round/tour1_complet_unitedfl.csv",
        index_col=0,
        parse_dates=["dateelection", "date"],
    )
    
    # only president elections after 2002
    polls = polls[(polls.date >= "2002-01") & (polls.type == "president")].drop(
        [
            "type",
            "abstention",
            "undecided",
        ],
        axis=1,
    )
    
    # no green party candidate in 2017
    polls.loc[polls["dateelection"] == "2017-04-23", "nbgreen"] = 0

    return polls.sort_values(
        ["dateelection", "date", "sondage", "samplesize"]
    ).reset_index(drop=True)


def format_data(polls: pd.DataFrame, parties_complete: List[str]):
    
    # start all elections on Jan 1st
    dfs = []
    for date in polls.dateelection.unique():
        date = pd.to_datetime(date)
        df = polls[(polls.dateelection == date) & (polls.date >= f"{date.year}-01")]
        df["countdown"] = dates_to_idx(df["date"], date).astype(int)
        dfs.append(df)
    
    # compute "other" category
    polls = (
        pd.concat(dfs)
        .set_index(["dateelection", "date", "countdown", "sondage", "samplesize"])
        .rename(columns={col: col.split("nb")[1] for col in polls if col.startswith("nb")})[
            parties_complete[:-1]
        ]
    )
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


def train_split_and_idx_vars(
    polls: pd.DataFrame, 
    test_cutoff: pd.Timedelta = None
):
    
    last_election = polls.dateelection.unique()[-1]
    polls_train = polls[polls.dateelection != last_election]
    polls_test = polls[polls.dateelection == last_election]
    
    if test_cutoff:
        test_cutoff_ = last_election - test_cutoff
    else:
        test_cutoff_ = last_election - pd.Timedelta(2, "D")
    
    polls_train = pd.concat([polls_train, polls_test[polls_test.date <= test_cutoff_]])
    polls_test = polls_test[polls_test.date > test_cutoff_]
    

    return polls_train, polls_test


def dates_to_idx(timelist, reference_date):
    """Convert datetimes to numbers in reference to reference_date"""

    t = (reference_date - timelist) / np.timedelta64(1, "D")

    return np.asarray(t)


def dims_and_coords(
    polls_train: pd.DataFrame,
    parties_complete: List[str],
):
    COORDS = {
        "observations": polls_train.index,
        "parties": parties_complete[:-1],
        "parties_complete": parties_complete,
    }
    pollster_id, COORDS["pollsters"] = polls_train["sondage"].factorize(sort=True)
    countdown_id, COORDS["countdown"] = polls_train["countdown"].values, np.arange(
        polls_train["countdown"].max() + 1
    )
    election_id, COORDS["elections"] = polls_train["dateelection"].factorize()

    return pollster_id, countdown_id, election_id, COORDS


def plot_check(polls: pd.DataFrame, parties_complete: List[str]):
    fig, ax = plt.subplots(figsize=(12, 5))
    for p in parties_complete:
        ax.plot(polls["date"], polls[p] / polls["samplesize"], "o", label=p, alpha=0.4)
    ax.legend(ncol=4, frameon=True, loc="upper right")
    
