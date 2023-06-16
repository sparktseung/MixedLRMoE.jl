# MixedLRMoE Package

**MixedLRMoE** is an extention to the **LRMoE** package in order to incorporate random effects, which are useful for 
modelling multilevel data in insurance and more general statistical applications.

The theoretical development of Mixed LRMoE (or Mixed Mixture of Experts, MMoE in short) is given in [Fung and Tseung (2022+)](https://arxiv.org/abs/2209.15207),
where it is shown to possess the desirable property of _denseness_. In other words, the Mixed LRMoE model has a potential 
to accurately resemble almost all characteristics inherited in multilevel data,
including the marginal distributions, dependence structures, regression links, random intercepts and random slopes.
In a particular case where the multilevel data is hierarchical, we further show that a nested version of the MMoE universally approximates a broad range of dependence structures of the random effects among different factor levels.

An application of the Mixed LRMoE in insurance contexts is given in [Tseung et al. (2023)](https://arxiv.org/abs/2209.15212),
where the model is shown to outperform classical benchmark models (Generalized Linear (Mixed) Models, or GL(M)M in short) for better differentiation of risky
and safe drivers in a real-world dataset, as well as providing intuitive and interpretable ratemaking results that accurately reflect
the unobserved heterogeneity of the drivers' risk profiles.