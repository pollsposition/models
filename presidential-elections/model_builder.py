from typing import Dict, Tuple, List

import arviz
import numpy as np
import pandas as pd
import pymc3 as pm

from gpapproximation import make_gp_basis
from zerosumnormal import ZeroSumNormal

# Aesara will replace Theano in PyMC 4.0
if pm.math.erf.__module__.split(".")[0] == "theano":
    import theano.tensor as aet
else:
    import aesara.tensor as aet


def dates_to_idx(timelist, reference_date):
    """Convert datetimes to numbers in reference to reference_date"""
    t = (reference_date - timelist) / np.timedelta64(1, "D")
    return np.asarray(t)


def standardize(series):
    """Standardize a pandas series"""
    return (series - series.mean()) / series.std()


class ModelBuilder:
    def __init__(
        self,
        election_to_predict: str,
        test_cutoff: pd.Timedelta = None,
    ):
        """
        Initialize the model builder.
        Parameters
        ----------
        data: pd.DataFrame
            Dataframe containing the raw, uncleaned field data.
        """
        self.parties_complete = [
            "farleft",
            "left",
            "green",
            "center",
            "right",
            "farright",
            "other",
        ]
        self.gp_config = {
            "lengthscale": [5, 14, 28],
            "kernel": "gaussian",
            "zerosum": True,
            "variance_limit": 0.95,
            "variance_weight": [0.1, 0.3, 0.6],
        }

        (
            self.polls_train,
            self.polls_test,
            self.results_raw,
            self.results_mult,
        ) = self._clean_polls(test_cutoff)

        _, self.unique_elections = self.polls_train["dateelection"].factorize()
        _, self.unique_pollsters = self.polls_train["sondage"].factorize()
        self.results_oos = self.results_mult[
            self.results_mult.dateelection != election_to_predict
        ].copy()

        self._load_predictors()
        (
            self.results_preds,
            self.campaign_preds,
        ) = self._standardize_continuous_predictors()

    def _clean_polls(
        self,
        test_cutoff: pd.Timedelta = None,
    ):
        polls = self._load_polls()
        results_raw, results_mult, polls = self._format_polls(
            polls, self.parties_complete
        )
        (
            polls_train,
            polls_test,
        ) = self._train_split(polls, test_cutoff)

        return polls_train, polls_test, results_raw, results_mult

    @staticmethod
    def _load_polls():
        polls = pd.read_csv(
            "https://raw.githubusercontent.com/pollsposition/data/main/sondages"
            "/tour1_complet_unitedfl.csv",
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

    def _format_polls(self, polls: pd.DataFrame, parties_complete: List[str]):
        # start all elections on Jan 1st
        dfs = []
        for date in polls.dateelection.unique():
            date = pd.to_datetime(date)
            df = polls[(polls.dateelection == date) & (polls.date >= f"{date.year}-01")]
            df["countdown"] = dates_to_idx(df["date"], reference_date=date).astype(int)
            dfs.append(df)

        # compute "other" category
        polls = (
            pd.concat(dfs)
            .set_index(["dateelection", "date", "countdown", "sondage", "samplesize"])
            .rename(
                columns={
                    col: col.split("nb")[1] for col in polls if col.startswith("nb")
                }
            )[parties_complete[:-1]]
        )
        polls["other"] = 100 - polls.sum(1)
        np.testing.assert_allclose(polls.sum(1).values, 100)

        # isolate results
        polls = polls.reset_index()
        results_raw = polls[polls.sondage == "result"]
        polls = polls[polls.sondage != "result"].set_index(["date", "sondage"])

        # cast polls as multinomial obs
        polls = self.cast_as_multinomial(polls)

        # cast results as multinomial
        results_mult = self.results_as_multinomial(results_raw)

        return results_raw, results_mult, polls.reset_index()

    def results_as_multinomial(self, results_raw: pd.DataFrame) -> pd.DataFrame:
        # need number of people who voted
        raw_json = pd.read_json(
            "https://raw.githubusercontent.com/pollsposition/data/main/resultats/presidentielles"
            ".json",
        )
        raw_json = raw_json.loc["premier_tour"].to_dict()

        jsons = []
        for year, dateelection in zip(
            results_raw.dateelection.dt.year.unique(), results_raw.dateelection.unique()
        ):
            df = pd.json_normalize(raw_json[year])[["exprimes"]]
            df["dateelection"] = dateelection
            jsons.append(df)
        jsons = pd.concat(jsons)

        results_mult = (
            results_raw.join(jsons.set_index("dateelection"), on="dateelection")
            .drop("samplesize", axis="columns")
            .rename(columns={"exprimes": "samplesize"})
        )
        results_mult["samplesize"] = (
            results_mult["samplesize"] // 100
        )  # to prevent overflow in Multinomial

        return self.cast_as_multinomial(results_mult)

    def cast_as_multinomial(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.parties_complete] = (
            (df[self.parties_complete] / 100)
            .mul(df["samplesize"], axis=0)
            .round()
            .fillna(0)
            .astype(int)
        )
        df["samplesize"] = df[self.parties_complete].sum(1)

        return df

    @staticmethod
    def _train_split(polls: pd.DataFrame, test_cutoff: pd.Timedelta = None):
        last_election = polls.dateelection.unique()[-1]
        polls_train = polls[polls.dateelection != last_election]
        polls_test = polls[polls.dateelection == last_election]

        if test_cutoff:
            test_cutoff_ = last_election - test_cutoff
        else:
            test_cutoff_ = last_election - pd.Timedelta(2, "D")

        polls_train = pd.concat(
            [polls_train, polls_test[polls_test.date <= test_cutoff_]]
        )
        polls_test = polls_test[polls_test.date > test_cutoff_]

        return polls_train, polls_test

    def _load_predictors(self):
        self.unemployment_data = self._load_generic_predictor(
            "https://raw.githubusercontent.com/pollsposition/data/main/predicteurs"
            "/chomage_national_trim.csv",
            name="unemployment",
            freq="Q",
            skiprows=2,
        )
        self.polls_train, self.polls_test, self.results_mult = self._merge_with_data(
            self.unemployment_data, freq="Q"
        )

        self.gaz_data = self._load_generic_predictor(
            "https://raw.githubusercontent.com/pollsposition/data/main/predicteurs"
            "/gazole_nat_mois.csv",
            name="gazole",
            freq="M",
            skiprows=3,
        )
        self.polls_train, self.polls_test, self.results_mult = self._merge_with_data(
            self.gaz_data, freq="M"
        )

        self.popularity_data = self._load_popularity()
        self.polls_train, self.polls_test, self.results_mult = self._merge_with_data(
            self.popularity_data, freq="M"
        )
        self.incumbency_index, self.election_incumbent = self._load_incumbents()
        return

    def _merge_with_data(
        self, predictor: pd.DataFrame, freq: str
    ) -> List[pd.DataFrame]:
        polls_train = self.polls_train.copy()
        polls_test = self.polls_test.copy()
        results_mult = self.results_mult.copy()
        dfs = []

        for data in [polls_train, polls_test, results_mult]:
            # add freq to data
            data.index = data["date"].dt.to_period(freq)
            # data.index = pd.DatetimeIndex(data["date"].values).to_period(freq)
            # merge with data
            dfs.append(data.join(predictor).reset_index(drop=True))

        return dfs

    @staticmethod
    def _load_generic_predictor(
        file: str, name: str, freq: str, skiprows: int, sep: str = ";"
    ) -> pd.DataFrame:
        data = pd.read_csv(
            file,
            sep=sep,
            skiprows=skiprows,
        ).iloc[:, [0, 1]]
        data.columns = ["date", name]
        data = data.sort_values("date")

        # as timestamps variables:
        data.index = pd.period_range(
            start=data.date.iloc[0], periods=len(data), freq=freq
        )

        return data.drop("date", axis=1)

    @staticmethod
    def _load_popularity() -> pd.DataFrame:
        popularity = pd.read_csv(
            "https://raw.githubusercontent.com/pollsposition/dashboards/main/exports"
            "/popularity_all_presidents.csv",
            parse_dates=["month"],
        )
        popularity["month"] = popularity["month"].dt.to_period("M")
        popularity = popularity.set_index("month")
        popularity.index = popularity.index.shift(-1)

        # popularity model starts from chirac2
        raw_popularity = pd.read_csv(
            "https://raw.githubusercontent.com/pollsposition/data/main/sondages/popularite.csv",
            index_col=0,
        )
        raw_popularity = raw_popularity[raw_popularity.president == "chirac1"]
        raw_popularity.index = pd.to_datetime(raw_popularity.index)

        raw_popularity = raw_popularity.resample("M").mean()
        raw_popularity.index = raw_popularity.index.to_period("M")

        raw_popularity["president"] = "chirac1"
        raw_popularity["mean_pop"] = raw_popularity["approve_pr"] / 100
        raw_popularity = raw_popularity.drop(
            ["samplesize", "approve_pr", "disapprove_pr"], axis="columns"
        )

        return pd.concat([raw_popularity, popularity])

    def _load_incumbents(self):
        incumbents = {
            "chirac1": "right",
            "chirac2": "right",
            "sarkozy": "right",
            "hollande": "left",
        }
        never_incumbents = list(set(self.parties_complete) - set(incumbents.values()))

        incumbency_index = pd.get_dummies(
            self.polls_train["president"].replace(incumbents)
        )
        election_incumbent = pd.get_dummies(
            self.results_mult["president"].replace(incumbents)
        )
        for p in never_incumbents:
            incumbency_index[p] = 0
            election_incumbent[p] = 0

        return (
            incumbency_index[self.parties_complete],
            election_incumbent[self.parties_complete],
        )

    def _standardize_continuous_predictors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        continuous_predictors = [
            "date",
            "unemployment",
            "gazole",
            "mean_pop",
        ]
        self.continuous_predictors = (
            pd.concat(
                [
                    self.polls_train[continuous_predictors],
                    self.results_mult[continuous_predictors],
                ]
            )
            .set_index("date")
            .sort_index()
        )
        cont_preds_stdz = standardize(self.continuous_predictors)

        return (
            cont_preds_stdz.loc[self.unique_elections],
            cont_preds_stdz.loc[
                self.continuous_predictors.index.difference(self.unique_elections)
            ],
        )

    def build_model(
        self, polls: pd.DataFrame = None, predictors: pd.DataFrame = None
    ) -> pm.Model:
        """Build and return a pymc3 model."""
        (
            self.pollster_id,
            self.countdown_id,
            self.election_id,
            self.coords,
        ) = self._build_coords(polls)

        with pm.Model(coords=self.coords) as model:

            data_containers = self._build_data_containers(polls, predictors)
            party_intercept, election_party_intercept = self._build_intercepts()
            house_effects, house_election_effects = self._build_house_effects()
            (
                unemployment_effect,
                gas_effect,
                popularity_effect,
                incumbency_effect,
            ) = self._build_predictors()

            party_time_weight = self._build_party_amplitude()
            election_party_time_weight = self._build_election_party_amplitude()
            gp_basis_funcs, gp_basis_dim = make_gp_basis(
                time=self.coords["countdown"], gp_config=self.gp_config, key="parties"
            )
            party_time_effect = self._build_party_gp(
                gp_basis_funcs, gp_basis_dim, party_time_weight
            )
            election_party_time_effect = self._build_election_party_gp(
                gp_basis_funcs, gp_basis_dim, election_party_time_weight
            )

            noisy_popularity, latent_pop_t0 = self._build_regressions(
                party_intercept,
                election_party_intercept,
                party_time_effect,
                election_party_time_effect,
                unemployment_effect,
                gas_effect,
                popularity_effect,
                incumbency_effect,
                house_effects,
                house_election_effects,
                data_containers,
            )

            concentration = pm.InverseGamma("concentration", mu=300, sigma=100)
            pm.DirichletMultinomial(
                "N_approve",
                a=concentration * noisy_popularity,
                n=data_containers["observed_N"],
                observed=data_containers["observed_polls"],
                dims=("observations", "parties_complete"),
            )
            pm.DirichletMultinomial(
                "R",
                a=concentration * latent_pop_t0[:-1],
                n=data_containers["results_N"],
                observed=data_containers["observed_results"],
                dims=("elections_observed", "parties_complete"),
            )

        return model

    def _build_coords(self, polls: pd.DataFrame = None):
        data = polls if polls is not None else self.polls_train

        COORDS = {
            "observations": data.index,
            "parties_complete": self.parties_complete,
        }
        pollster_id, COORDS["pollsters"] = data["sondage"].factorize(sort=True)
        countdown_id, COORDS["countdown"] = data["countdown"].values, np.arange(
            data["countdown"].max() + 1
        )
        election_id, COORDS["elections"] = data["dateelection"].factorize()
        COORDS["elections_observed"] = COORDS["elections"][:-1]

        return pollster_id, countdown_id, election_id, COORDS

    def _build_data_containers(
        self, polls: pd.DataFrame = None, campaign_predictors: pd.DataFrame = None
    ) -> Dict[str, pm.Data]:

        if polls is None:
            polls = self.polls_train
        if campaign_predictors is None:
            campaign_predictors = self.campaign_preds

        return dict(
            election_idx=pm.Data("election_idx", self.election_id, dims="observations"),
            pollster_idx=pm.Data("pollster_idx", self.pollster_id, dims="observations"),
            countdown_idx=pm.Data(
                "countdown_idx", self.countdown_id, dims="observations"
            ),
            stdz_unemp=pm.Data(
                "stdz_unemp",
                campaign_predictors["unemployment"].to_numpy(),
                dims="observations",
            ),
            stdz_gas=pm.Data(
                "stdz_gas",
                campaign_predictors["gazole"].to_numpy(),
                dims="observations",
            ),
            stdz_pop=pm.Data(
                "stdz_pop",
                campaign_predictors["mean_pop"].to_numpy(),
                dims="observations",
            ),
            incumbency_index=pm.Data(
                "incumbency_index",
                self.incumbency_index.to_numpy(),
                dims=("observations", "parties_complete"),
            ),
            election_unemp=pm.Data(
                "election_unemp",
                self.results_preds["unemployment"].to_numpy(),
                dims="elections",
            ),
            election_gas=pm.Data(
                "election_gas",
                self.results_preds["gazole"].to_numpy(),
                dims="elections",
            ),
            election_pop=pm.Data(
                "election_pop",
                self.results_preds["mean_pop"].to_numpy(),
                dims="elections",
            ),
            election_incumbent=pm.Data(
                "election_incumbent",
                self.election_incumbent.to_numpy(),
                dims=("elections", "parties_complete"),
            ),
            observed_N=pm.Data(
                "observed_N",
                polls["samplesize"].to_numpy(),
                dims="observations",
            ),
            observed_polls=pm.Data(
                "observed_polls",
                polls[self.parties_complete].to_numpy(),
                dims=("observations", "parties_complete"),
            ),
            results_N=pm.Data(
                "results_N",
                self.results_oos["samplesize"].to_numpy(),
                dims="elections_observed",
            ),
            observed_results=pm.Data(
                "observed_results",
                self.results_oos[self.parties_complete].to_numpy(),
                dims=("elections_observed", "parties_complete"),
            ),
        )

    @staticmethod
    def _build_intercepts() -> Tuple[pm.Distribution, pm.Distribution]:
        party_intercept = ZeroSumNormal(
            "party_intercept", sigma=0.5, dims="parties_complete"
        )
        election_party_intercept_sd = pm.HalfNormal(
            "election_party_intercept_sd", 0.15, dims="parties_complete"
        )
        election_party_intercept = ZeroSumNormal(
            "election_party_intercept",
            sigma=election_party_intercept_sd[None, ...],
            dims=("elections", "parties_complete"),
            zerosum_axes=(0, 1),
        )

        return party_intercept, election_party_intercept

    @staticmethod
    def _build_house_effects() -> Tuple[pm.Distribution, pm.Distribution]:
        house_effects = ZeroSumNormal(
            "house_effects",
            sigma=0.1,
            dims=("pollsters", "parties_complete"),
            zerosum_axes=(0, 1),  # try no ZeroSum on pollsters
        )
        house_election_effect_sd = pm.HalfNormal(
            "house_election_effect_sd", 0.15, dims=("pollsters", "parties_complete")
        )
        house_election_effects_raw = ZeroSumNormal(
            "house_election_effects_raw",
            dims=("pollsters", "parties_complete", "elections"),
            zerosum_axes=(0, 1, 2),
        )
        house_election_effects = pm.Deterministic(
            "house_election_effects",
            house_election_effect_sd[..., None] * house_election_effects_raw,
            dims=("pollsters", "parties_complete", "elections"),
        )

        return house_effects, house_election_effects

    @staticmethod
    def _build_predictors() -> Tuple[
        pm.Distribution,
        pm.Distribution,
        pm.Distribution,
        pm.Distribution,
        pm.Distribution,
        pm.Distribution,
    ]:
        unemployment_effect = ZeroSumNormal(
            "unemployment_effect",
            sigma=0.1,
            dims="parties_complete",
        )
        gas_effect = ZeroSumNormal(
            "gas_effect",
            sigma=0.1,
            dims="parties_complete",
        )
        popularity_effect = ZeroSumNormal(
            "popularity_effect",
            sigma=0.1,
            dims="parties_complete",
        )
        incumbency_effect = pm.Normal("incumbency_effect", sigma=0.1)
        return (
            unemployment_effect,
            gas_effect,
            popularity_effect,
            incumbency_effect,
        )

    @staticmethod
    def _build_party_amplitude() -> pm.Distribution:
        lsd_intercept = pm.Normal("lsd_intercept", sigma=0.5)
        lsd_party_effect = ZeroSumNormal(
            "lsd_party_effect_1", sigma=0.1, dims="parties_complete"
        )
        return pm.Deterministic(
            "party_time_weight",
            aet.exp(lsd_intercept + lsd_party_effect),
            dims="parties_complete",
        )

    @staticmethod
    def _build_election_party_amplitude() -> pm.Distribution:
        lsd_party_effect = ZeroSumNormal(
            "lsd_party_effect_2", sigma=0.5, dims="parties_complete"
        )
        lsd_election_effect = ZeroSumNormal(
            "lsd_election_effect", sigma=0.5, dims="elections"
        )
        lsd_election_party_sd = pm.HalfNormal("lsd_election_party_sd", 0.15)
        lsd_election_party_raw = ZeroSumNormal(
            "lsd_election_party_raw",
            dims=("parties_complete", "elections"),
            zerosum_axes=(0, 1),
        )
        lsd_election_party_effect = pm.Deterministic(
            "lsd_election_party_effect",
            lsd_election_party_sd * lsd_election_party_raw,
            dims=("parties_complete", "elections"),
        )
        return pm.Deterministic(
            "election_party_time_weight",
            aet.exp(
                lsd_party_effect[:, None]
                + lsd_election_effect[None, :]
                + lsd_election_party_effect
            ),
            dims=("parties_complete", "elections"),
        )

    @staticmethod
    def _build_party_gp(
        gp_basis_funcs: np.ndarray,
        gp_basis_dim: str,
        party_time_weight: pm.Distribution,
    ) -> pm.Distribution:

        party_time_coefs_raw = ZeroSumNormal(
            "party_time_coefs_raw",
            dims=(gp_basis_dim, "parties_complete"),
            zerosum_axes=-1,
        )
        return pm.Deterministic(
            "party_time_effect",
            aet.tensordot(
                gp_basis_funcs,
                party_time_weight[None, ...] * party_time_coefs_raw,
                axes=(1, 0),
            ),
            dims=("countdown", "parties_complete"),
        )

    @staticmethod
    def _build_election_party_gp(
        gp_basis_funcs: np.ndarray,
        gp_basis_dim: str,
        election_party_time_weight: pm.Distribution,
    ) -> pm.Distribution:

        election_party_time_coefs = ZeroSumNormal(
            "election_party_time_coefs",
            sigma=election_party_time_weight[None, ...],
            dims=(gp_basis_dim, "parties_complete", "elections"),
            zerosum_axes=(1, 2),
        )
        return pm.Deterministic(
            "election_party_time_effect",
            aet.tensordot(gp_basis_funcs, election_party_time_coefs, axes=(1, 0)),
            dims=("countdown", "parties_complete", "elections"),
        )

    @staticmethod
    def _build_regressions(
        party_intercept: pm.Distribution,
        election_party_intercept: pm.Distribution,
        party_time_effect: pm.Distribution,
        election_party_time_effect: pm.Distribution,
        unemployment_effect: pm.Distribution,
        gas_effect: pm.Distribution,
        popularity_effect: pm.Distribution,
        incumbency_effect: pm.Distribution,
        house_effects: pm.Distribution,
        house_election_effects: pm.Distribution,
        data_containers: Dict[str, pm.Data],
    ) -> Tuple[pm.Distribution, pm.Distribution]:
        # regression for polls
        latent_mu = (
            party_intercept
            + election_party_intercept[data_containers["election_idx"]]
            + party_time_effect[data_containers["countdown_idx"]]
            + election_party_time_effect[
                data_containers["countdown_idx"], :, data_containers["election_idx"]
            ]
            + aet.dot(
                data_containers["stdz_unemp"][:, None], unemployment_effect[None, :]
            )
            + aet.dot(data_containers["stdz_gas"][:, None], gas_effect[None, :])
            + aet.dot(data_containers["stdz_pop"][:, None], popularity_effect[None, :])
            + data_containers["incumbency_index"] * incumbency_effect
        )
        pm.Deterministic(
            "latent_popularity",
            aet.nnet.softmax(latent_mu),
            dims=("observations", "parties_complete"),
        )
        noisy_mu = (
            latent_mu
            + house_effects[data_containers["pollster_idx"]]
            + house_election_effects[
                data_containers["pollster_idx"], :, data_containers["election_idx"]
            ]
        )

        # regression for results
        latent_mu_t0 = (
            party_intercept
            + election_party_intercept
            + party_time_effect[0]
            + election_party_time_effect[0].T
            + aet.dot(
                data_containers["election_unemp"][:, None], unemployment_effect[None, :]
            )
            + aet.dot(data_containers["election_gas"][:, None], gas_effect[None, :])
            + aet.dot(
                data_containers["election_pop"][:, None], popularity_effect[None, :]
            )
            + data_containers["election_incumbent"] * incumbency_effect
        )

        return (
            pm.Deterministic(
                "noisy_popularity",
                aet.nnet.softmax(noisy_mu),
                dims=("observations", "parties_complete"),
            ),
            pm.Deterministic(
                "latent_pop_t0",
                aet.nnet.softmax(latent_mu_t0),
                dims=("elections", "parties_complete"),
            ),
        )

    def sample_all(
        self, *, model: pm.Model = None, var_names: List[str], **sampler_kwargs
    ) -> arviz.InferenceData:
        """
        Sample the model and return the trace.
        Parameters
        ----------
        model : optional
            A model previously created using `self.build_model()`.
            Build a new model if None (default)
        **sampler_kwargs : dict
            Additional arguments to `pm.sample`
        """
        if model is None:
            model = self.build_model()

        with model:
            prior_checks = pm.sample_prior_predictive()
            trace = pm.sample(return_inferencedata=False, **sampler_kwargs)
            post_checks = pm.sample_posterior_predictive(trace, var_names=var_names)

        return arviz.from_pymc3(
            trace=trace,
            prior=prior_checks,
            posterior_predictive=post_checks,
            model=model,
        )

    def forecast_election(
        self,
        idata: arviz.InferenceData,
    ) -> arviz.InferenceData:
        new_dates, oos_data = self._generate_oos_data(idata)
        oos_data = self._join_with_predictors(oos_data)
        forecast_data_index = pd.DataFrame(
            data=0,  # just a placeholder
            index=pd.MultiIndex.from_frame(oos_data),
            columns=self.parties_complete,
        )
        forecast_data = forecast_data_index.reset_index()

        PREDICTION_COORDS = {"observations": new_dates}
        PREDICTION_DIMS = {
            "latent_popularity": ["observations", "parties_complete"],
            "noisy_popularity": ["observations", "parties_complete"],
            "N_approve": ["observations", "parties_complete"],
        }

        forecast_model = self.build_model(polls=forecast_data, predictors=forecast_data)
        with forecast_model:
            ppc = pm.sample_posterior_predictive(
                idata,
                var_names=[
                    "latent_popularity",
                    "noisy_popularity",
                    "N_approve",
                    "latent_pop_t0",
                    "R",
                ],
            )
            ppc = arviz.from_pymc3_predictions(
                ppc,
                idata_orig=idata,
                inplace=False,
                coords=PREDICTION_COORDS,
                dims=PREDICTION_DIMS,
            )

        return ppc

    def _generate_oos_data(
        self, idata: arviz.InferenceData
    ) -> Tuple[pd.Index, pd.DataFrame]:
        countdown = idata.posterior["countdown"]
        elections = idata.posterior["elections"]

        estimated_days = np.tile(countdown[::-1], reps=len(elections))
        N_estimated_days = len(estimated_days)

        new_dates = [
            pd.date_range(
                periods=max(countdown.data) + 1,
                end=date,
                freq="D",
            ).to_series()
            for date in elections.data
        ]
        new_dates = pd.concat(new_dates).index

        oos_data = pd.DataFrame.from_dict(
            {
                "countdown": estimated_days,
                "dateelection": np.repeat(
                    self.unique_elections, repeats=len(countdown)
                ),
                "sondage": np.random.choice(
                    self.unique_pollsters, size=N_estimated_days
                ),
                "samplesize": np.random.choice(
                    self.polls_train["samplesize"].values, size=N_estimated_days
                ),
            }
        )
        oos_data["date"] = new_dates

        return new_dates, oos_data.set_index("date")

    def _join_with_predictors(self, oos_data: pd.DataFrame) -> pd.DataFrame:
        oos_data["quarter"] = oos_data.index.to_period("Q")
        oos_data["month"] = oos_data.index.to_period("M")

        oos_data = oos_data.join(self.unemployment_data, on="quarter").join(
            self.popularity_data, on="month"
        )
        # check no missing values
        np.testing.assert_allclose(0, oos_data.isna().any().mean())

        # stdz predictors based on observed values
        oos_data["unemployment"] = (
            oos_data["unemployment"] - self.continuous_predictors["unemployment"].mean()
        ) / self.continuous_predictors["unemployment"].std()
        oos_data["mean_pop"] = (
            oos_data["mean_pop"] - self.continuous_predictors["mean_pop"].mean()
        ) / self.continuous_predictors["mean_pop"].std()

        return oos_data.reset_index()
