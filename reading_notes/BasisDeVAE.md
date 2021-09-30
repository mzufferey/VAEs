BasisDeVAE

t DeVAE, a novel VAE-based model with a derivative-based forward mapping, allowing for greater control over decoder behaviour via specifi- cation of the decoder function in derivative space

DeVAE can be paired with a sparse clustering prior to create BasisDe- VAE and perform interpretable simultaneous di- mensionality reduction and feature-level cluster- ing. We

an approach that combines dimensionality reduction and feature-level clustering within a VAE framework (Basis- VAE) has been developed (M¨artens & Yau, 2020). This type of model is of particular utility on tabular datasets where each feature may have a distinct standalone meaning (e.g. it represents a protein, gene or age). BasisVAE groups subsets of features together whose behaviour is similar over the latent dimensions. Figure

the application of dimensionality reduction via a standard VAE uncovers structure only in the rows (samples) of tabular data X ∈ RN×d, whereas simultaneous dimensionality reduc- tion and feature-level clustering aims to uncover structure in both the rows (samples) and columns (features).
These

, specify- ing particular functional characteristics with current VAE decoders remains an open challenge; Basis- VAE allows us to identify features which share exactly the same functional shape or dynamics, but it does not provide a mechanism to define a broader group of shared patterns among features, e.g. to cluster together all monotonically increasing functions. The

we would like to i) learn a low-dimensional representation z (dimensionality reduction) and ii) cluster features according to their behaviour as a function of z (feature-level clustering). The standard VAE framework allows us to perform only i). With our proposed BasisDeVAE model we can perform both tasks simultaneously and guarantee interpretability of the inferred cluster assignments

DeVAE and BasisDeVAE, which model feature-level be- haviour x in terms of derivatives with respect to the latent dimensions z

this allows for greater control over decoder behaviour in settings where certain forms of a priori knowl- edge and/or physical constraints, such as monotonicity and transience, are desirable without having to explicitly define parametric functional forms

Our work is particularly motivated by modelling disease or biological progression from cross-sectional data. In this problem, a cross-sectional collection of input samples is projected on to a one-dimensional latent space. If the domi- nant latent source of variation in the cross-sectional data is temporal, the positioning of samples in the latent space then corresponds to relative ordering in time or pseudotime.

If the dominant source of variation in a cross-sectional dataset is associated with biological or disease progression, we can use a VAE to encode and map the samples onto a one-dimensional latent space (pseudotime) and decode to understand the feature-level variability with pseudotime

BasisVAE (M¨artens & Yau, 2020) enables simultaneous dimensionality reduction and feature-level clustering within the VAE framework. It achieves this by modifying the form of the decoder function, introducing additional cluster- associating latent variables into the generative model and devising an efficient collapsed variational inference scheme for learning.

Inference is carried out on this model using collapsed vari- ational inference (Hensman et al., 2012; Hensman et al., 2015) by introducing a categorical variational posterior over cluster assignments, and a variational lower bound on the marginal log-likelihood is derived by repeated applica- tion of Jensen’s inequality and by additionally marginalising out π

Inference in the BasisVAE setting therefore equates to minimising L jointly over the decoder parameters
θ˜, posterior cluster assignments ξ and encoder parameters φ, which is performed via mini-batch gradient descent

The functional form of the decoder in a standard VAE is challenging to control due to its specification as a DNN and training being performed in the DNN’s weight space

 In our
biological progression modelling applications, biomarkers often only exhibit a limited number of high-level behaviours, namely monotonic increases, monotonic decreases or a time- limited transient signal (see

While a standard VAE could “learn” these behaviours given a sufficiently large number of low-noise samples, we would like to be able to robustly enforce such structures in low-sample or high- noise settings while still maintaining flexibility to model complex feature behaviours

DeVAE specifies the decoder via its derivatives with respect to the latent variable z

The final form of the decoder computation (10) therefore reduces to a weighted sum of neural network evaluations. Hence, backpropagation through the decoder is straightfor- ward and the computation is fully parallelisable

https: //github.com/djdanks/BasisDeVAE and

We next embed DeVAE within the BasisVAE framework to perform simultaneous feature-level clustering and dimen- sionality reduction with control over the behaviour and meaning of the feature clusters.

removing the notion of feature j being obtained via the translation and scaling of one ofK underlying basis functions and replacing it with the idea that feature j is described by a function from one ofK families, each with interpretable properties specified via their derivatives.
In

One can therefore constrain the output range of these decoder constituents (e.g. by pass- ing through a scaled sigmoid function) in addition to their monotonicity within this framework.

Inference in the BasisDeVAE model is carried out using the collapsed variational inference scheme applied to BasisVAE, noting the removal of the necessity to learn the translation and scale parameters δ and λ.

that it is the derivative-based approach of DeVAE that has allowed us to specify monotonic components without having to adopt parametric constraints (e.g. linearity, sigmoid-like models) or neural network weight constraints. It has also allowed us to specify a general form of Gaussian-like transient com- ponent. The BasisDeVAE framework has then enabled the data-driven learning of cluster assignments, linking each feature with an interpretable behaviour cluster.
5.

The analysis task on such data is to i) recreate a representation of the temporal variable via a one-dimensional latent pseu dotime z and learn the gene expression profiles with respect to z, and ii) to group features with similar expression pro- files into interpretable clusters (Figure 1). These two tasks can be performed simultaneously within the BasisVAE and BasisDeVAE frameworks

d DeVAE and BasisDeVAE, two novel VAE models with decoders specified in terms of their derivatives with respect to the latent variable z

the derivative-based construction of the decoder employed in these models allows the specification of functional forms with both expressivity and interpretability, showing in par- ticular how to specify monotonicity and transience in the context of pseudotemporal models of biological progression

Our demonstration of BasisDeVAE in the context of pseudotemporal analysis of single-cell data can be seen as a generalisation of Campbell & Yau (2018) capable of capturing any form of monotonic or Gaussian-like transient behaviour, not just sigmoidal or parametric Gaussian pro- files. It also automatically assigns features to interpretable clusters without having to rely on pre-assigned genes as in Campbell & Yau (2018) or on a fully data-driven forward model as in M¨artens & Yau (2020).