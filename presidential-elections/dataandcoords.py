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
    #(
     #   polls_train,
      #  polls_test,
       # observed_days_id,
        #estimated_days,
        #whole_timeline,
#    ) = train_split_and_idx_vars(polls, election_date, test_cutoff)
 #   pollster_id, COORDS = dims_and_coords(polls_train, parties_complete, whole_timeline)
  #  plot_check(polls, parties_complete)

#    return (
 #       polls_train,
  #      polls_test,
   #     results,
    #    observed_days_id,
     #   estimated_days,
      #  pollster_id,
       # COORDS,
    #)
    return results, polls


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

    return polls_train, polls_test, observed_days_idx, estimated_days, whole_timeline


def dates_to_idx(timelist, reference_date):
    """Convert datetimes to numbers in reference to reference_date"""

    t = (reference_date - timelist) / np.timedelta64(1, "D")

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

#COORDS = {
 #   "parties": ["farleft", "left", "green", "center", "right", "farright"],
  #  "parties_complete": [
   #     "farleft",
    #    "left",
     #   "green",
      #  "center",
       # "right",
        #"farright",
        #"other",
#    ],
#}

#pollster_id, COORDS["pollsters"] = polls["sondage"].factorize(sort=True)
#countdown_id, COORDS["countdown"] = polls.countdown.values, np.arange(
 #   polls.countdown.max() + 1
#)
#election_id, COORDS["elections"] = polls["dateelection"].factorize()
#COORDS[
 #   "observations"
#] = polls.index


def plot_check(polls: pd.DataFrame, parties_complete: List[str]):
    fig, ax = plt.subplots(figsize=(12, 5))
    for p in parties_complete:
        ax.plot(polls["date"], polls[p] / polls["samplesize"], "o", label=p, alpha=0.4)
    ax.legend(ncol=4, frameon=True, loc="upper right");