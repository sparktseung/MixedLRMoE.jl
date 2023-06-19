# Random Effects

Below are two classical examples where random effects are used for modelling unobserved effects.
We also illustrate how they are represented in the **MixedLRMoE.jl** package.

## Panel Data
In a panel data study, suppose there are ``N_1`` unique individuals in a population of size ``n``, where each individual may be observed multiple times.
* We assume the individual-level unobserved effects are ``\{w_1^{(l)}\}_{l = 1, 2, \dots, N_1}``, where the subscript ``1`` denotes
  the first (and only, in this example) level of random effects and the superscript ``(l)`` indicates the ``l``-th level, i.e. the ``l``-th unique individual.
* For each observation ``i``, ``\mathbf{w}_i = (w_{i1})`` is a one-element vector that contains the unobserved effect of the ``i``-th observation,
  where ``w_{i1} \in \{w_1^{(l)}\}_{l = 1, 2, \dots, N_1}``.
* Correspondingly, we could construct a ``n``-length mapping vector ``t_1`` such that
  the ``i``-th element maps observation ``i`` to one of the levels in ``\{w_1^{(l)}\}_{l = 1, 2, \dots, N_1}``.

Suppose we have a dataframe of the insurance claim history for the following individuals across different years,
where the number of drivers ``N_1 = 3`` and the number of observations ``n = 10``.

| Observation | Driver | Year | ... (other columns) |
|-------------|--------|------|-----|
| 1  | Amy    | 2015 | ... |
| 2  | Amy    | 2016 | ... |
| 3  | Amy    | 2017 | ... |
| 4  | Bob    | 2014 | ... |
| 5  | Bob    | 2015 | ... |
| 6  | Bob    | 2016 | ... |
| 7  | Bob    | 2017 | ... |
| 8  | Bob    | 2018 | ... |
| 9  | Sam    | 2018 | ... |
| 10 | Sam    | 2019 | ... |

We represent the driver-level random effects by a set of random variables ``\{ w_1^{(1)}, w_1^{(2)}, w_1^{(3)} \}``, for Amy, Bob and Sam respectively.
For convenience (also generality, see the next example), this is represented by a vector/tuple of vectors
`re_list = [w_1]` where `w_1 = [w_11, w_12, w_13]` in the package implementation.

The mapping vector is constructed as `t_1 = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3]`.
For convenience (also generality, see the next example),
this is reshaped as a ``n``-by-``1`` matrix `mapping_matrix = hcat(t_1)` in the package implementation.

Correspondingly, the mapped vectors of random effects to each observation are ``\mathbf{w}_i = (w_1^{(1)})`` for ``i = 1, 2, 3`` (Amy),
``\mathbf{w}_i = (w_1^{(2)})`` for ``i = 4, 5, \dots, 8`` (Bob), and ``\mathbf{w}_i = (w_1^{(3)})`` for ``i = 9, 10`` (Sam).


## Hierarchical Data
We consider a more complex hierarchical data structure with two levels of random effects.
A classical example is the school-teacher effects on student performance (see e.g. [here](https://www.jstor.org/stable/1170147)),
whereby the teacher-level random effects are nested within the school-level random effects
(assuming each teacher only works at one school). The same hierarchical data structure can be used to model many other scenarios, such as country-(province/state)-city effects
on the probability of fire accidents.

Let us consider a Canadian province-city data.
* We assume the province-level unobserved effects are ``\{w_1^{(l)}\}_{l = 1, 2, \dots, N_1}``, where the subscript ``1`` denotes
  the first level of random effects and the superscript ``(l)`` indicates the ``l``-th level, i.e. the ``l``-th province.
* We also assume the city-level unobserved effects are ``\{w_2^{(l)}\}_{l = 1, 2, \dots, N_2}``, where the subscript ``2`` denotes
  the second level of random effects and the superscript ``(l)`` indicates the ``l``-th level, i.e. the ``l``-th city. Note that
  cities are nested within provinces, i.e. each city belongs to one and only one province.
* For each observation ``i``, ``\mathbf{w}_i = (w_{i1}, w_{i2})`` is a two-element vector that contains the unobserved effects of the ``i``-th observation,
  where ``w_{i1} \in \{w_1^{(l)}\}_{l = 1, 2, \dots, N_1}`` (province) and ``w_{i2} \in \{w_2^{(l)}\}_{l = 1, 2, \dots, N_2}`` (city).
* Correspondingly, we could construct two ``n``-length mapping vectors ``t_1`` and ``t_2`` to map each observation to one of the levels in
  ``\{w_1^{(l)}\}_{l = 1, 2, \dots, N_1}`` and ``\{w_2^{(l)}\}_{l = 1, 2, \dots, N_2}`` respectively.

Suppose we have a dataframe of fire accidents history for the following cities across different provinces in Canada,
where the number of provinces ``N_1 = 2``, the number of cities ``N_2 = 4`` and the number of observations ``n = 10``.

| Observation | Province | City | Year | ... (other columns) |
|-------------|----------|------|------|-----|
| 1  | Ontario  | Toronto | 2019 | ... |
| 2  | Ontario  | Toronto | 2020 | ... |
| 3  | Ontario  | Ottawa  | 2019 | ... |
| 4  | Ontario  | Ottawa  | 2020 | ... |
| 5  | Ontario  | Ottawa  | 2021 | ... |
| 6  | Quebec   | Montreal  | 2018 | ... |
| 7  | Quebec   | Montreal  | 2019 | ... |
| 8  | Quebec   | Montreal  | 2020 | ... |
| 9  | Quebec   | Quebec City  | 2019 | ... |
| 10 | Quebec   | Quebec City  | 2020 | ... |

We represent the province-level random effects by a set of random variables ``\{ w_1^{(1)}, w_1^{(2)} \}``, for Ontario and Quebec respectively,
and the city-level random effects by a set of random variables ``\{ w_2^{(1)}, w_2^{(2)}, w_2^{(3)}, w_2^{(4)}\}``, for Toronto, Ottawa, Montreal and Quebec City respectively.

These random effects are represented by a vector/tuple of vectors `re_list = [w_1, w_2]`
where `w_1 = [w_11, w_12]` and `w_2 = [w_21, w_22, w_23, w_24]` in the package implementation.

The mapping vectors are constructed as `t_1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]` and `t_2 = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4]`,
which are reshaped as a ``n``-by-``2`` matrix `mapping_matrix = hcat(t_1, t_2)` in the package implementation.

Correspondingly, the mapped vectors of random effects to each observation are ``\mathbf{w}_i = (w_1^{(1)}, w_2^{(1)})`` for ``i = 1, 2`` (Ontario-Toronto),
``\mathbf{w}_i = (w_1^{(1)}, w_2^{(2)})`` for ``i = 3, 4, 5`` (Ontario-Ottawa), ``\mathbf{w}_i = (w_1^{(2)}, w_2^{(3)})`` for ``i = 6, 7, 8`` (Quebec-Montreal),
and ``\mathbf{w}_i = (w_1^{(2)}, w_2^{(4)})`` for ``i = 9, 10`` (Quebec-Quebec City).

## Other Generalizations

The above are two classical examples where random effects may be incorporated in a model. The package implementation is more general and can be applied to other scenarios,
such as crossed interactions (as opposed to hierarchical), and more than two levels of random effects.
In these cases, the implementations of the random effects and mapping vectors/matrix are similar to the above.
