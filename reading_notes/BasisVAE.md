Common data analysis workflows additionally use clustering techniques to iden- tify groups of similar features. 

This usually leads to a two-stage process, however, it would be desirable to construct a joint modelling framework for simultaneous dimensionality reduction and clustering of features. 

BasisVAE: a combination of the VAE and a probabilistic clustering prior, which lets us learn a one-hot basis function representation as part of the decoder network.

for scenarios where not all features are aligned, we develop an extension to handle translation- invariant basis functions.

a general pur- pose approach for joint dimensionality reduction and clustering of features within the Variational Autoen- coder (VAE) framework (Kingma

Our methodology particularly applies to any tabular data problem where individual features have a distinct physical meaning (e.g. a gene, a physiological mea- surement, the position of a joint on a moving body) and there is an interest in understanding the relation- ship between these features.

our approach, which we refer to as Basis- VAE, assumes that the features vary as functions of latent dimensions z, and that there exists a finite set of such (basis) functions. 

BasisVAE aims to discover the scale-invariant clustering of features whilst simultaneously inferring the latent variable z (in

similar shapes which are merely shifted versions of each other. To capture this, we introduce translation invariance as part of the model so it would allow these features to be further grouped together, whilst maintaining an explicit parameter- isation that would allow the unique properties of each feature to be maintained

For easier interpretability, each feature has its own scale and translation parame- ters

we endow VAEs with the capability to cluster features in the data. We propose to achieve this via introducing additional probabilistic structure within the decoder, thus combining the scalability of VAEs with a Bayesian mixture prior for the purpose of clustering features (architecture

A typical decoder network is used to map z to the basis functions, whereas a specialised layer with Categorical random variables is used to map the latter to features. The posterior distribution of these Categorical variables provides us with the inferred clustering.

But as our interest lies in modelling tabular data where individual features have a distinct meaning, we additionally introduce notation f(j)
decoder
where j indexes the j-th output dimension of the de- coder network (i.e. predictions for feature j)

The goal of our work is to extend the VAE to address the scientific problem of identifying groups of features with similar trends. That is, we would like to cluster features. We

We propose to achieve this via a mixture prior as part of the VAE decoding model. The

The idea is to introduce a set of “basis” functions  parameterised by the neural network, and probabilistically assign every feature y(j) to one of those basis functions . We refer to this as a one- hot-basis functional representation.

we achieve this by replacing the decoder network fdecoder : RQ → RP with a basis decoder network fbasis : RQ → RK (where the number of basis functions K < P) whose output is mapped to the data via Categorical random variables

that this can be seen as a mixture model on the features rather than observations

how to represent two signals which have the same functional shape, but which may be shifted or scaled relative to one another.

Sometimes, we would instead like the basis function to only capture the shape of the pattern. This

This behaviour could be achieved via introducing feature-specific scale and translation parameters

added value from scale and translation invariance is two-fold:

1. Interpretability: Feature-specific scale and/or translation parameters lets us explicitly interpret how features within a cluster relate to each other

2. Increased model capacity: In model given by equa- tion (1), a large number of basis functions K may be needed to capture the range of observed pat- terns, whereas (2) and (3) can provide the same flexibility with a significantly smaller K

we have presented BasisVAE as if the number of underlying basis functions was known. However, this is rarely the case in practice, usually we do not know K a priori. Ideally,

this would require care- ful specification of π = (π1, . . . , πK) which would be problem dependent. Hierarchical Bayesian models offer a way to get around this by placing a distribution over π. One; a mechanism to learn an appropriately sparse π rather than pre-specify its value. This

for non-linear latent variable models, such as VAEs, analytic updates are not available in closed form any more. Also it is not straightforward to exploit conjugacy in

one hand we want to utilise the computational benefits (scalability, modularity, computational efficiency) of the VAE-framework, while on the other hand we would like to make use of the sparsity properties that are present in more classical (“pre automatic differentiation era”) hierarchical Bayesian mixture models.

One approach would be to introduce an approximate posterior -> non-collapsed inference; this inference scheme can get stuck in local modes, resulting in undesirable behaviour

improve our previously decribed non-collapsed inference by marginalising out π.

Adaptation of ELBO for large data sets: For

https://github.com/kasparmartens/BasisVAE.

BasisVAE has identified a number of clusters with distinct gene expression behaviour , whereas (B) its translation-invariant version has grouped genes with similar profiles together , thus aiding further interpretability.

example single-cell with 1LD as pseudotime

Manually interpreting individual genes in such a large-scale setting is infeasible, however the inferred BasisVAE cluster allocations make interpretation eas- ier. In

they have the same shape, but we can also interpret their relative shift by inspecting the δ values.

the inferred clustering structure (e.g. where the behaviour of gene might be delayed relative to another, but exhibiting the same shape) does not imply any bi- ological functional relatedness, BasisVAE methodology can add to the exploratory toolkit for investigators to explore and rank such potential relationships between genes explicitly rather than retrospectively through ad hoc approaches that are common in genomics.