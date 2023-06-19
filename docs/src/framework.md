# Modelling Framework

The Mixed LRMoE model is formulated very similarly to the [LRMoE model](https://actsci.utstat.utoronto.ca/LRMoE.jl/stable/framework/),
with the addition of random effects that describe unobserved effects.

Let ``(\mathbf{x}_i, \mathbf{y}_i), i = 1, 2, \dots, n`` denote a set of observations, where ``\mathbf{x}_i`` denotes the covariates and ``\mathbf{y}_i`` the response(s).
A vector ``\mathbf{w}_i`` of random effects are also associated with each observation, which may be one or multiple levels with varying number of factors.
A detailed description of ``\mathbf{w}_i`` is given in [Random Effects](@ref).

Given ``\mathbf{x}_i`` and ``\mathbf{w}_i``, the ``i``-th observation is classified into one of ``g`` latent classes by the so-called **logit gating function**. The probability of belonging to the ``j``-th latent class is given by

```math
\pi_j(\mathbf{x}_i, \mathbf{w}_i; \mathbf{\alpha}, \mathbf{\beta}) = \frac{\exp (\mathbf{\alpha}_j^T \mathbf{x}_i + \mathbf{\beta}_j^T \mathbf{w}_i)}{\sum_{j'=1}^{g} \exp (\mathbf{\alpha}_{j'}^T \mathbf{x}_i + \mathbf{\beta}_{j'}^T \mathbf{w}_i)}, \quad j = 1, 2, \dots, g-1
```
For model identifiability reasons, we assume ``\mathbf{\alpha}_g = \mathbf{0}`` and ``\mathbf{\beta}_g = \mathbf{0}`` which correspond to the reference class.
We also fix ``\mathbf{\beta}_1 = \mathbf{1}`` to avoid arbtrary scaling and sign-switching of the random effects.

Conditional on the latent class ``j``, the distribution of the response ``\mathbf{y}_i`` is given by an **expert function** with density

```math
f_j(\mathbf{y}_i; \mathbf{\psi}_j) = \prod_{d=1}^{D} f_{jd}(\mathbf{y}_{id}; \mathbf{\psi}_{jd})
```
where we assume conditional independence of dimensions ``1, 2, \dots, D`` of ``\mathbf{y}_i``, if it is a vector of responses.

The (partial) likelihood function given the random effects ``\mathbf{w} = (\mathbf{w}_1, \mathbf{w}_2, \dots, \mathbf{w}_n)`` is therefore
```math
L(\mathbf{\alpha}, \mathbf{\beta}, \mathbf{\psi}; \mathbf{x}, \mathbf{w}, \mathbf{y}) = \prod_{i=1}^{n} \left\{ \sum_{j=1}^{g} \pi_j(\mathbf{x}_i, \mathbf{w}_i; \mathbf{\alpha}, \mathbf{\beta}) f_j(\mathbf{y}_i; \mathbf{\psi}_j) \right\}
```

The parameters to estimate are the regression coefficients ``\mathbf{\alpha}_j`` and ``\mathbf{\beta}_j``, as well as the parameters of the expert functions ``\mathbf{\psi}_j``, which is implemented by the a variational Expectation-Conditional-Maximization algorithm (details omitted, see [Tseung et al. (2023)](https://arxiv.org/abs/2209.15212)). Simultaneously, the **approximated** posterior distributions of the random effects ``\mathbf{w}`` are also obtained, which is especially useful for making predictions.




