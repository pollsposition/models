---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.1
  kernelspec:
    display_name: elections-models
    language: python
    name: elections-models
---

# French presidents' popularity

```python
import datetime

import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
```

## Exploratory data analysis

```python
data = pd.read_csv('./plot_data/raw_polls.csv', parse_dates = True, index_col="Unnamed: 0")
data['year'] = data.index.year
data['month'] = data.index.month
data['month_name'] = data.index.month_name()
data['sondage'] = data['sondage'].replace('Yougov', 'YouGov')
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
print(data[data.president == "macron"].shape)
print(data[data.president == "macron"].index.to_period('M').shape)
pd.Categorical(data[data.president == "macron"].index.to_period('M')).codes
```

```python
pd.Categorical(data["sondage"]).codes
```

Let us look at simple stats on the pollsters. First the total number of polls they've produced:

```python
data["sondage"].value_counts()
```

For most pollsters we should be able to estimate their bias quite accurately, however `YouGov` has only 3 points and its estimated bias will heavily depend on the prior we chose.


There are substantially more polls in the years 2017, 2018 and 2019. The lower count for 2002 and 2021 is explained by the fact that we don't have the full year.

```python
data['year'].value_counts().sort_index()
```

The number of polls is homogeneous among months, except in the summer because, well, France:

```python
data['month'].value_counts().sort_index()
```

When it comes to the method it seems that pollsters prefer the phone over the internet and face-to-face. There is still a substantial number of each should we have to estimate biais.

```python
data['method'].value_counts()
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

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12,16))
ax1.plot(dates, approval_rates, 'o')
ax1.set_ylim(0, 1)
ax1.set_ylabel("Does approve")
for date in newterm_dates:
    ax1.axvline(date)

ax2.plot(dates, disapproval_rates, 'o', alpha=.5)
ax2.set_ylabel("Does not approve")
ax2.set_ylim(0, 1)
for date in newterm_dates:
    ax2.axvline(date)
    
ax3.plot(dates, doesnotrespond, 'o', alpha=.5)
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
phone = data[data["method"] == "phone"]["p_approve"].values
internet = data[data["method"] == "internet"]["p_approve"].values
facetoface = data[data["method"] == "face to face"]["p_approve"].values

colors = plt.rcParams["axes.prop_cycle"]()
fig, ax = plt.subplots(figsize=(11,5))

c = next(colors)["color"]
ax.hist(phone, label="phone", color=c, alpha=.4)
ax.axvline(np.mean(phone), color=c, linestyle='--')

c = next(colors)["color"]
ax.hist(internet, label="internet", color=c, alpha=.4)
ax.axvline(np.mean(internet), color=c, linestyle='--')

c = next(colors)["color"]
ax.hist(facetoface, label="face to face", color=c, alpha=.4)
ax.axvline(np.mean(facetoface), color=c, linestyle='--')

ax.set_xlabel(r"$p_{+}$")
plt.legend();
```

### Pollster bias


We plot the distribution of the approval rates for each pollster. Note that this is not very scientific: pollsters like YouGov appeared recently so their rates will be tilted towards the last presidents' popularity while the other will be more evenly distributed.

It is already interesting to see that the bulk of the distributions is below 0.5:

```python
pollster_vals = {pollster: data[data["sondage"] == pollster]["p_approve"].values for pollster in list(pollsters)}

colors = plt.rcParams["axes.prop_cycle"]()
fig, axes = plt.subplots(ncols=2, nrows=5, sharex=True, figsize=(12,16))
axes = [ax for axs in axes for ax in axs]
for ax, (pollster, vals) in zip(axes, pollster_vals.items()):
    c = next(colors)["color"]
    ax.hist(vals, alpha=.3, color=c, label=pollster)
    ax.axvline(x=np.mean(vals), color=c, linestyle='--')
    ax.set_xlabel(r"$p_{+}$")
    ax.legend()
```

## Rolling standard deviation

We now compute the rolling variance of the approval rates. We weigh each poll equally, even though we probably should weigh them according to their respective sample size.

```python
rolling_std = data.reset_index().groupby(['year', 'month']).std().reset_index()[["year", "month", "p_approve"]]
rolling_std
```

```python
years = [f"{i}" for i in range(2003, 2021)]
values = rolling_std[rolling_std["year"].between(2003, 2020)]["p_approve"].values
dates = [datetime.datetime.strptime(f"{year}-{month}", '%Y-%m') for year in years for month in range(1, 13) ]

newterm_dates = data.reset_index().groupby("president").first()["index"].values

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, values, 'o')
for date in newterm_dates:
    ax.axvline(date, linestyle='--')
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
face = data[data['method'] == 'face to face']
dates_face = face.index

other = data[data['method'] != 'face to face']
dates_other = other.index

newterm_dates = data.reset_index().groupby("president").first()["index"].values


fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates_face, face['p_approve'].values, 'o', alpha=0.4, label='face to face')
ax.plot(dates_other, other['p_approve'].values, 'o', alpha=0.4, label='other')
for date in newterm_dates:
    ax.axvline(date, linestyle='--')

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
data = data.reset_index().merge(data.groupby(['year', 'month'])['p_approve'].mean().reset_index(), on=['year', 'month'], suffixes=["", "_mean"]).rename(
    columns={"index": "field_date"}
)
data['diff_approval'] = data['p_approve'] - data['p_approve_mean']
data
```

```python
pollster_vals = {pollster: data[data["sondage"] == pollster]["diff_approval"].values for pollster in list(pollsters)}

colors = plt.rcParams["axes.prop_cycle"]()
fig, axes = plt.subplots(ncols=2, nrows=5, sharex=True, figsize=(12,16))
axes = [ax for axs in axes for ax in axs]

for ax, (pollster, vals) in zip(axes, pollster_vals.items()):
    c = next(colors)["color"]
    ax.hist(vals, alpha=.3, color=c, label=pollster)
    ax.axvline(x=np.mean(vals), color=c, linestyle='--')
    ax.axvline(x=0, color='black')
    ax.legend()

plt.xlabel(r"$p_{approve} - \bar{p}_{approve}$", fontsize=25);
```

And now for the bias per method:

```python
phone = data[data["method"] == "phone"]["diff_approval"].values
internet = data[data["method"] == "internet"]["diff_approval"].values
facetoface = data[data["method"] == "face to face"]["diff_approval"].values

colors = plt.rcParams["axes.prop_cycle"]()
fig, ax = plt.subplots()

c = next(colors)["color"]
ax.hist(phone, label="phone", color=c, alpha=.4)
ax.axvline(np.mean(phone), color=c, linestyle='--')

c = next(colors)["color"]
ax.hist(internet, label="internet", color=c, alpha=.4)
ax.axvline(np.mean(internet), color=c, linestyle='--')

c = next(colors)["color"]
ax.hist(facetoface, label="face to face", color=c, alpha=.4)
ax.axvline(np.mean(facetoface), color=c, linestyle='--')

ax.axvline(x=0, color="black")
ax.set_xlabel(r"$p_+ - \bar{p}_{+}$", fontsize=25)
plt.legend();
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
data["num_approve"] = np.floor(data["samplesize"] * data["p_approve"]).astype('int')
pollster_id, pollsters = data["sondage"].factorize(sort=True)
method_id, methods = data["method"].factorize(sort=True)
month_correspondence = {i - 1: datetime.date(2021, i, 1).strftime('%B') for i in range(1, 13)}
month_id, months = (data["month"] - 1).values, list(month_correspondence.keys())

data
```

Each observation is uniquely identified by (president, pollster, field_date):

```python
COORDS = {
    "pollster": pollsters,
    "method": methods,
    "month": months,
    "observation": data.set_index(["president", "sondage", "field_date"]).index
}
```

```python
with pm.Model(coords=COORDS) as pooled_popularity:
    
    pollster_bias = pm.Normal("pollster_bias", 0, .15, dims="pollster")
    method_bias = pm.Normal("method_bias", 0, .15, dims="method")
    # sigma_mu = pm.HalfNormal("sigma_mu", .1)
    mu = pm.GaussianRandomWalk(
        "mu",
        sigma=.25,
        dims="month"
        )

    popularity = pm.Deterministic(
        "popularity",
        pm.math.invlogit(mu[month_id] + pollster_bias[pollster_id] + method_bias[method_id]),
        dims="observation"
    )

    N_approve = pm.Binomial("N_approve", p=popularity, n=data["samplesize"], observed=data["num_approve"], dims="observation")
```

```python
with pooled_popularity:
    idata = pm.sample(return_inferencedata=True)
```

```python
az.plot_trace(idata, var_names=['pollster_bias', 'method_bias'], compact=True);
```

```python
from scipy.special import expit as logistic
```

```python
idata.posterior["mu"]
```

```python
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(range(60), np.mean(inv_logit(posterior['mu']), axis=0))
ax.plot(range(60), np.var(inv_logit(posterior['mu']), axis=0))
ax.set_ylabel("Popularity")
ax.set_xlabel("Months into term");
```

```python
def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

fig, ax = plt.subplots(figsize=(12,8))
for i in range(1000):
    ax.plot(range(60), inv_logit(posterior['mu'][i,:]), alpha=.005, color="blue")
ax.set_ylabel("Popularity")
ax.set_xlabel("Months into term");
```

```python
np.mean(inv_logit(posterior['alpha_p']), axis=0)
```

```python
np.mean(inv_logit(posterior['alpha_m']), axis=0)
```

```python
pd.Categorical(data['method']).unique()
```

```python

```
