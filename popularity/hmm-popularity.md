---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.0
  kernelspec:
    display_name: pollposition
    language: python
    name: pollposition
---

# French presidents' popularity

```python
import datetime

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from scipy.special import expit as logistic
```

## Exploratory data analysis

```python
data = pd.read_csv(
    "./plot_data/raw_polls.csv", parse_dates=True, index_col="Unnamed: 0"
)

data["year"] = data.index.year
data["month"] = data.index.month

data["sondage"] = data["sondage"].replace("Yougov", "YouGov")
data["method"] = data["method"].replace("face-to-face&internet", "face to face")
print("columns: ", data.columns, "\n")

minimum = np.min(data[["year"]].values)
maximum = np.max(data[["year"]].values)
pollsters = data["sondage"].unique()

comment = f"""The dataset contains {len(data)} polls between the years {minimum} and {maximum}.
There are {len(pollsters)} pollsters: {', '.join(list(pollsters))}
"""
print(comment)
```

```python
# need to confirm if we wanna do that
data.loc[(data.sondage == 'Kantar') & (data.method == "internet"), "method"] = "face to face"
```

Let us look at simple stats on the pollsters. First the total number of polls they've produced:

```python
data["sondage"].value_counts()
```

For most pollsters we should be able to estimate their bias quite accurately, however `YouGov` has only 3 points and its estimated bias will heavily depend on the prior we chose.


There are substantially more polls in the years 2017, 2018 and 2019. The lower count for 2002 and 2021 is explained by the fact that we don't have the full year.

```python
data["year"].value_counts().sort_index()
```

The number of polls is homogeneous among months, except in the summer because, well, France:

```python
data["month"].value_counts().sort_index()
```

When it comes to the method it seems that pollsters prefer the phone over the internet and face-to-face. There is still a substantial number of each should we have to estimate biais.

```python
data["method"].value_counts()
```

## Let's look at the data!


Let us now plot the approval rate of presidents over time:

```python
az.style.use("arviz-darkgrid")

approval_rates = data["p_approve"].values
disapproval_rates = data["p_disapprove"].values
newterm_dates = data.reset_index().groupby("president").first()["index"].values
doesnotrespond = 1 - approval_rates - disapproval_rates

dates = data.index

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 16))
ax1.plot(dates, approval_rates, "o", alpha=0.5)
ax1.set_ylim(0, 1)
ax1.set_ylabel("Does approve")
for date in newterm_dates:
    ax1.axvline(date)

ax2.plot(dates, disapproval_rates, "o", alpha=0.5)
ax2.set_ylabel("Does not approve")
ax2.set_ylim(0, 1)
for date in newterm_dates:
    ax2.axvline(date)

ax3.plot(dates, doesnotrespond, "o", alpha=0.5)
ax3.set_ylabel("Does not respond")
for date in newterm_dates:
    ax3.axvline(date)
```

We notice two things when looking at these plots:

1. Approval rates strikingly systematically decrease as the terms comes along;
2. While that's true, some events seem to push the approval rate back up, even though temporarily. This happened in every term, actually. Can that variance be explained solely with a random walk?
3. Non-response rate is quite high during Macron's term.


## Computing the biais: naive method


### Method bias

```python
method_vals = {
    method: data[data["method"] == method]["p_approve"].values
    for method in list(data["method"].unique())
}

colors = plt.rcParams["axes.prop_cycle"]()
fig, ax = plt.subplots(figsize=(11, 5))

for method, vals in method_vals.items():
    c = next(colors)["color"]
    ax.hist(vals, alpha=0.3, color=c, label=method)
    ax.axvline(x=np.mean(vals), color=c, linestyle="--")

ax.set_xlabel(r"$p_{+}$")
ax.legend();
```

### Pollster bias


We plot the distribution of the approval rates for each pollster. Note that this is not very scientific: pollsters like YouGov appeared recently so their rates will be tilted towards the last presidents' popularity while the other will be more evenly distributed.

It is already interesting to see that the bulk of the distributions is below 0.5:

```python
pollster_vals = {
    pollster: data[data["sondage"] == pollster]["p_approve"].values
    for pollster in list(pollsters)
}

colors = plt.rcParams["axes.prop_cycle"]()
fig, axes = plt.subplots(ncols=2, nrows=5, sharex=True, figsize=(12, 16))

for ax, (pollster, vals) in zip(axes.ravel(), pollster_vals.items()):
    c = next(colors)["color"]
    ax.hist(vals, alpha=0.3, color=c, label=pollster)
    ax.axvline(x=np.mean(vals), color=c, linestyle="--")
    ax.set_xlabel(r"$p_{+}$")
    ax.legend()
```

## Rolling standard deviation

We now compute the rolling variance of the approval rates. We weigh each poll equally, even though we probably should weigh them according to their respective sample size.

```python
rolling_std = (
    data.reset_index()
    .groupby(["year", "month"])
    .std()
    .reset_index()[["year", "month", "p_approve"]]
)
rolling_std
```

```python
years = [f"{i}" for i in range(2003, 2021)]
values = rolling_std[rolling_std["year"].between(2003, 2020)]["p_approve"].values
dates = [
    datetime.datetime.strptime(f"{year}-{month}", "%Y-%m")
    for year in years
    for month in range(1, 13)
]

newterm_dates = data.reset_index().groupby("president").first()["index"].values

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(dates, values, "o", alpha=0.5)
for date in newterm_dates:
    ax.axvline(date, linestyle="--")
```

There is an abnormally high variance for Chirac's second term, and for the beggining of Macron's term. As a matter of fact, the previous scatterplot of $p_{approve}$ clearly shows almost two different curves. Let's look at the data for Chirac's term directly:

```python
chirac = data[data["president"] == "chirac2"]
chirac2007 = chirac[chirac["year"] >= 2006]
chirac2007
```

And now for the beggining of Macron's term:

```python
macron = data[data["president"] == "macron"]
macron2017 = macron[macron["year"] == 2017]
macron2017.head(20)
```

For Chirac's term it seems that difference stems from the polling method; face-to-face approval rates seem to be much lower. Let's visualize it:

```python
face = data[data["method"] == "face to face"]
dates_face = face.index

other = data[data["method"] != "face to face"]
dates_other = other.index

newterm_dates = data.reset_index().groupby("president").first()["index"].values


fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(dates_face, face["p_approve"].values, "o", alpha=0.4, label="face to face")
ax.plot(dates_other, other["p_approve"].values, "o", alpha=0.4, label="other")
for date in newterm_dates:
    ax.axvline(date, linestyle="--")

ax.set_ylim(0, 1)
ax.set_ylabel("Does approve")
fig.legend();
```

```python
data[data["sondage"] == "Kantar"]["method"].value_counts()
```

```python
data[data["method"] == "face to face"]["sondage"].value_counts()
```

```python
data[data["sondage"] == "Ifop"]["method"].value_counts()
```

## A more serious analysis of bias

To investigate bias we now compute the rolling mean of the $p_{approve}$ values and compare each method's and pollster's deviation from the mean.

```python
data = (
    data.reset_index()
    .merge(
        data.groupby(["year", "month"])["p_approve"].mean().reset_index(),
        on=["year", "month"],
        suffixes=["", "_mean"],
    )
    .rename(columns={"index": "field_date"})
)
data["diff_approval"] = data["p_approve"] - data["p_approve_mean"]
data
```

```python
pollster_vals = {
    pollster: data[data["sondage"] == pollster]["diff_approval"].values
    for pollster in list(pollsters)
}

colors = plt.rcParams["axes.prop_cycle"]()
fig, axes = plt.subplots(ncols=2, nrows=5, sharex=True, figsize=(12, 16))

for ax, (pollster, vals) in zip(axes.ravel(), pollster_vals.items()):
    c = next(colors)["color"]
    ax.hist(vals, alpha=0.3, color=c, label=pollster)
    ax.axvline(x=np.mean(vals), color=c, linestyle="--")
    ax.axvline(x=0, color="black")
    ax.legend()

plt.xlabel(r"$p_{approve} - \bar{p}_{approve}$", fontsize=25);
```

And now for the bias per method:

```python
method_vals = {
    method: data[data["method"] == method]["diff_approval"].values
    for method in list(data["method"].unique())
}

colors = plt.rcParams["axes.prop_cycle"]()
fig, ax = plt.subplots(figsize=(11, 5))

for method, vals in method_vals.items():
    c = next(colors)["color"]
    ax.hist(vals, alpha=0.3, color=c, label=method)
    ax.axvline(x=np.mean(vals), color=c, linestyle="--")

ax.axvline(x=0, color="black")
ax.set_xlabel(r"$p_+ - \bar{p}_{+}$", fontsize=25)
ax.legend();
```

## Todo

### Bias

There are many things worth exploring before moving on to modeling. First, bias:

- Bias by method: does one method tend to produce higher approval rates? More non-response?

- We can use the rolling average by method/pollster and look at the distribution of the difference between values and the average per method/pollster.

### Variance

Look at a rolling estimate of the variance of approval rates.

### Trend

Look at the successive difference of the average approval rate. Is there a trend
here?

# Model

Each poll $i$ at month $m$ from the beginning of a president’s term finds that
$y_i$ individuals have a positive opinion of the president’s action over
$n_i$ respondents. We model this as

$$y_{i,m} \sim Binomial(p_{i,m}, n_{i,m})$$

We loosely call $p_{i,m}$ the *popularity* of the president, $m$ month into his
presidency. This is the quantity we would like to model.

Why specify the month when the time information is already contained in the
succession of polls? Because French people tend to be less and less satisfied
with their president as their term moves, regardless of their action.

We model $p_{i,m}$ with a random walk logistic regression:

$$p_{i,m} = logit^{-1}(\mu_m + \alpha_k + \zeta_j)$$

$\mu_m$ is the underlying support for the president at month $m$. $\alpha_k$ is
the bias of the pollster, while $\zeta_j$ is the inherent bias of the polling
method. The biases are assumed to be completely unpooled at first, i.e we model
one bias for each pollster and method:

$$\alpha_k \sim Normal(0, \sigma_k)\qquad \forall pollster k$$

and 

$$\zeta_j \sim Normal(0, \sigma_j)\qquad \forall method j$$

We treat the time variation of $\mu$ with a correlated random walk:

$$\mu_m | \mu_{m-1} \sim Normal(\mu_{m-1}, \sigma_m)$$

For the sake of simplicity, we choose not to account at first for a natural
decline in popularity $\delta$, the unmeployment at month $m$, $U_m$, or
random events that can happen during the term. 

```python
data["num_approve"] = np.floor(data["samplesize"] * data["p_approve"]).astype("int")
data
```

```python
pd.crosstab(data.sondage, data.method)
```

We can only estimate the bias for internet and phone, not for face-to-face and phone&internet, as they are only conducted by one pollster (Kantar and Ifop respectively). Similarly, we can use an interaction term only for those pollsters which use more than one method, i.e BVA, Ifop and Ipsos.


Each observation is uniquely identified by `(pollster, field_date)`:

```python
# for coords and indexing
#pollster_id, pollsters = data["sondage"].factorize(sort=True)
#method_id, methods = data["method"].factorize(sort=True)
pollster_by_method_id, pollster_by_methods = data.set_index(["sondage", "method"]).index.factorize(sort=True)
month_id = np.hstack(
    [
        pd.Categorical(
            data[data.president == president].field_date.dt.to_period("M")
        ).codes
        for president in data.president.unique()
    ]
)
months = np.arange(max(month_id) + 1)
```

```python
COORDS = {
    #"pollster": pollsters,
    #"method": methods,
    "pollster_by_method": pollster_by_methods,
    "month": months,
    "observation": data.set_index(["sondage", "field_date"]).index,
}
```

### Fixed `mu` for GRW

```python
with pm.Model(coords=COORDS) as pooled_popularity:

    bias = pm.Normal("bias", 0, 0.15, dims="pollster_by_method")
    mu = pm.GaussianRandomWalk("mu", sigma=1., dims="month")

    popularity = pm.Deterministic(
        "popularity",
        pm.math.invlogit(
            mu[month_id] + bias[pollster_by_method_id]
        ),
        dims="observation",
    )

    N_approve = pm.Binomial(
        "N_approve",
        p=popularity,
        n=data["samplesize"],
        observed=data["num_approve"],
        dims="observation",
    )
    
    idata = pm.sample(return_inferencedata=True)
```

We plot the posterior distribution of the pollster and method biases:

```python
az.plot_trace(idata, var_names=["~popularity"], compact=True);
```

Since we are performing a logistic regression, these coefficients can be tricky to interpret. When the bias is positive, this means that we need to add to the latent popularity to get the observation, which means that the pollster/method tends to be biased towards giving higher popularity scores.

```python
az.summary(idata, round_to=2, var_names=["~popularity"])
```

```python
mean_pollster_bias = (
    idata.posterior["bias"].mean(("chain", "draw")).to_dataframe()
)
mean_pollster_bias.round(2)
```

```python
ax = mean_pollster_bias.plot.bar(figsize=(14, 8), rot=30)
ax.set_title("$>0$ bias means pollster overestimates true popularity");
```

### TODO: Interpret and compare to data

There is a strong discrepancy for Kantar between what we observe and what the model returns. We saw earlier that all face-to-face polls were from Kantar. The bias may thus be fully captured by the face-to-face interaction. It is almost a philosophical question here whether the bias is due to the method or to the pollster, and this region of the posterior is weakly identified by the data so the coefficients corresponding to Kantar and face-to-face should not be interpreted separately.

On the other hand, the posterior coefficients for the method bias agree with our observation of the data. In other words, the model has decided, Kantar does nothing special: it considers the negative bias is all due to face-to-face :D

We now plot the posterior values of `mu`. Since the model is completely pooled, we only have 60 values, which correspond to a full term:

```python
post_pop = logistic(idata.posterior["mu"].stack(sample=("chain", "draw")))

fig, ax = plt.subplots()
for i in np.random.choice(post_pop.coords["sample"].size, size=1000):
    ax.plot(
        idata.posterior.coords["month"],
        post_pop.isel(sample=i),
        alpha=0.01,
        color="blue",
    )
post_pop.mean("sample").plot(ax=ax, color="orange", lw=2)
ax.set_ylabel("Popularity")
ax.set_xlabel("Months into term");
```

```python
hdi_data = az.hdi(logistic(idata.posterior["mu"]))
hdi_data
```

```python
ax = az.plot_hdi(idata.posterior.coords["month"], hdi_data=hdi_data)
ax.vlines(idata.posterior.coords["month"], hdi_data.sel(hdi="lower")["mu"], hdi_data.sel(hdi="higher")["mu"])
post_pop.median("sample").plot(ax=ax);
```

```python
az.plot_posterior(logistic(idata.posterior["mu"].sel(month=42)));
```

### Infer `mu` of GRW

```python
with pm.Model(coords=COORDS) as pooled_popularity:

    bias = pm.Normal("bias", 0, 0.15, dims="pollster_by_method")
    sigma_mu = pm.HalfNormal("sigma_mu", 0.5)
    mu = pm.GaussianRandomWalk("mu", sigma=sigma_mu, dims="month")

    popularity = pm.Deterministic(
        "popularity",
        pm.math.invlogit(
            mu[month_id] + bias[pollster_by_method_id]
        ),
        dims="observation",
    )

    N_approve = pm.Binomial(
        "N_approve",
        p=popularity,
        n=data["samplesize"],
        observed=data["num_approve"],
        dims="observation",
    )
    
    idata = pm.sample(tune=2000, draws=2000, return_inferencedata=True)
```

```python
az.plot_trace(idata, var_names=["~popularity"], compact=True);
```

```python
az.summary(idata, round_to=2, var_names=["~popularity"])
```

```python
post_pop = logistic(idata.posterior["mu"].stack(sample=("chain", "draw")))

fig, ax = plt.subplots()
for i in np.random.choice(post_pop.coords["sample"].size, size=1000):
    ax.plot(
        idata.posterior.coords["month"],
        post_pop.isel(sample=i),
        alpha=0.01,
        color="blue",
    )
post_pop.mean("sample").plot(ax=ax, color="orange", lw=2)
ax.set_ylabel("Popularity")
ax.set_xlabel("Months into term");
```

### Model overdispersion of polls

```python
with pm.Model(coords=COORDS) as pooled_popularity:
    
    bias = pm.Normal("bias", 0, 0.15, dims="pollster_by_method")
    sigma_mu = pm.HalfNormal("sigma_mu", 0.5)
    mu = pm.GaussianRandomWalk("mu", sigma=sigma_mu, dims="month")

    popularity = pm.Deterministic(
        "popularity",
        pm.math.invlogit(
            mu[month_id] + bias[pollster_by_method_id]
        ),
        dims="observation",
    )

    # overdispersion parameter
    theta = pm.Exponential("theta_offset", 1.0) + 10.0

    N_approve = pm.BetaBinomial(
        "N_approve",
        alpha=popularity * theta,
        beta=(1.0 - popularity) * theta,
        n=data["samplesize"],
        observed=data["num_approve"],
        dims="observation",
    )
    
    idata = pm.sample(tune=2000, draws=2000, return_inferencedata=True)
```

```python
az.plot_trace(idata, var_names=["~popularity"], compact=True);
```

```python
az.summary(idata, round_to=2, var_names=["~popularity"])
```

```python
post_pop = logistic(idata.posterior["mu"].stack(sample=("chain", "draw")))

fig, ax = plt.subplots()
for i in np.random.choice(post_pop.coords["sample"].size, size=1000):
    ax.plot(
        idata.posterior.coords["month"],
        post_pop.isel(sample=i),
        alpha=0.01,
        color="blue",
    )
post_pop.mean("sample").plot(ax=ax, color="orange", lw=2)
ax.set_ylabel("Popularity")
ax.set_xlabel("Months into term");
```

### Hierarchical model

```python
president_id, presidents = data["president"].factorize(sort=False)
COORDS["president"] = presidents
```

```python
with pm.Model(coords=COORDS) as hierarchical_popularity:
    
    bias = pm.Normal("bias", 0, 0.15, dims="pollster_by_method")
    
    #sigma_mu = pm.HalfNormal("sigma_mu", 0.5)
    sigma_mu = pm.Gamma("ls_trend", alpha=20, beta=1)
    mu = pm.GaussianRandomWalk("mu", sigma=sigma_mu, dims=("president", "month"))

    popularity = pm.Deterministic(
        "popularity",
        pm.math.invlogit(
             mu[president_id, month_id] + bias[pollster_by_method_id]
        ),
        dims="observation",
    )

    N_approve = pm.Binomial(
        "N_approve",
        p=popularity,
        n=data["samplesize"],
        observed=data["num_approve"],
        dims="observation",
    )
```

```python
hierarchical_popularity.check_test_point()
```

```python
with hierarchical_popularity:
    idata = pm.sample(return_inferencedata=True)
```

```python
az.plot_trace(idata, var_names=["~popularity"], compact=True);
```

```python
az.summary(idata, round_to=2, var_names=["~popularity"])
```

```python
post_pop = logistic(idata.posterior["mu"].stack(sample=("chain", "draw")))

fig, ax = plt.subplots()
for i in np.random.choice(post_pop.coords["sample"].size, size=1000):
    ax.plot(
        idata.posterior.coords["month"],
        post_pop.isel(sample=i),
        alpha=0.01,
        color="blue",
    )
post_pop.mean("sample").plot(ax=ax, color="orange", lw=2)
ax.set_ylabel("Popularity")
ax.set_xlabel("Months into term");
```

## TODO

- Model an interaction between method and pollster.

- Posterior predictive analysis: distribution of $p_{\mathrm{approve}}$ for each pollster and method. We can plot the approval rates for each poll for each president but we do not except anything to come from it because we mixed all the terms (although we may see a difference due to new pollsters appearing).

- Re-read the paper by Gellman et al. on predicting the US presidential election. We may be able to catch something new given our experience with this first model.

- Try out-of-sample popularity prediction.

- Test the sensitivity to $\sigma$ in the random walk.

- Learn $\sigma$ from data?

The natural next step is partial pooling for the values of `mu`

```python
%load_ext watermark
%watermark -n -u -v -iv
```
