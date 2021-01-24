# Forecasting French elections with Bayesian Statistics
### _Electoral forecasting models for the [PollsPosition project](https://alexandorra.github.io/pollsposition_blog/)_

Beyond electoral forecasting, this repository represents a sandbox that I use to learn and experiment with new statistical methods on real data. All models are open-sourced and built with [PyMC3](https://docs.pymc.io/) and [ArviZ](https://arviz-devs.github.io/arviz/). Notice a bug or have a suggestion? It goes without saying that issue tickets and pull requests are always welcome :star_struck:

I am currently migrating the old website to a new infrastructure. This is still in the making, but going forward the idea is that each model we develop will have its own interactive dashboard (ideally all hosted at the same place, but that's not guaranteed) and its own tutorial notebooks (which are all gathered in a readable format [here](https://alexandorra.github.io/pollsposition_blog/))

- The most recent model we worked on is a **Gaussian Process regression** to predict how French president's popularity evolves with time.
  - [Interactive dashboard](https://share.streamlit.io/alexandorra/pollsposition_website/main/gp-popularity-app.py)
  - [Tutorial notebook](https://alexandorra.github.io/pollsposition_blog/popularity/macron/gaussian%20processes/polls/2021/01/18/gp-popularity.html)

- In 2020, I experimented with a [hierarchical multinomial model](https://nbviewer.jupyter.org/github/AlexAndorra/pollsposition_models/blob/master/district-level/munic_model_prod.ipynb) to forecast Paris city-council elections. You can vizualize the plots and results in an interactive [Voilà](https://voila.readthedocs.io/en/stable/) web app by clicking this button [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AlexAndorra/pollsposition_models/master?urlpath=%2Fvoila%2Frender%2Fdistrict-level%2Fmunic_model_analysis.ipynb). Lots of amazing things -- coded by awesome people -- are happening under the hood when you click this button, so there may sometimes be some latency. But I think it will be worth your 10-second wait :partying_face:

- You like videos? Well, who doesn't? I'm happy to say it's your lucky day: I just gave a [talk at PyMCon 2020](https://youtu.be/EYdIzSYwbSw), going through and explaining this exact model. So, sit down in your long chair, get the :popcorn:, and enjoy the ride :clapper:

You are free to use the models and data uploaded here -- just make sure you properly cite and link to this repository. [Feel free to reach out](https://twitter.com/alex_andorra) for any question, comment or contribution. 

Best Bayesian wishes :vulcan_salute:
