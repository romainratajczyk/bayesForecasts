
# A Bayesian hurdle model for bilateral migration forecasting

Research project. Work in progress, will be submitted to Journal of Royal Statistics, Serie A.
Authors: Romain Ratajczyk, Varnel Tientcheu, Nicolas Chopin, Adrian E.Raftery, Nathan G.Welch. 


# Abstract 

Bayesian hierarchical models of international migration flows, most notably that of Welch and Raftery, forecast well over long horizons by relying only on the temporal inertia of migration corridors. They therefore cannot react to macroeconomic and geopolitical shocks that are observed when the forecast is made. We propose a hurdle model that reintroduces structural covariates while remaining fully Bayesian. A probit component built on Bayesian additive regression trees models whether a corridor is open, which matters because half of all bilateral corridors record no migrants. Conditional on opening, a zero-truncated negative binomial component with a constrained second-order autoregression and a hierarchy over countries and United Nations sub-regions models the volume. Posterior draws from both components are combined into a joint predictive distribution. On an out-of-sample forecast one five-year period ahead, the model reduces the mean absolute percentage error from 76.0% to 29.5% and the mean absolute error from about 1,200 to 1,121 migrants per corridor, with 95.0% empirical coverage of 95% prediction intervals against 93.0% for the benchmark. The threshold that turns opening probabilities into decisions is left to the user, and we report the trade-off it controls between missed and spurious corridors.




