

### constraint formulation

neural decomposition: introduces functional constraints

- integral constraints to constrain marginal effects of every neural network $f^θ_I(x)$ to be zero
  $$
  \int f_I(x_I)dx_i = 0\textrm{ }for\textrm{ }all\textrm{ } i\in I
  $$
  $\rightarrow$ the functional subspaces corresponding to  $f^θ_I$ and  $f^θ_{I,J}$ do not overlap anymore (apart from the constant zero function)

  $\rightarrow$ these functional subspaces are orthogonal in $L_2$

- example for a 2D-input $(x_1, x_2)$, the integral constraints are:

  - $\int f^θ_1(x_1)dx_1 = 0$ and $\int f^θ_2(x_2)dx_2 = 0$
  - $\int f^θ_{12}(x_1, x_2)dx_1 = 0$ for all $x_2$
  - $\int f^θ_{12}(x_1, x_2)dx_2 = 0$ for all $x_1$

- from these integral constraints: no overlap (sufficient for identifiability) + orthogonality (leads to an easily interpretable variance decomposition)



### optimization

$\Rightarrow$ constrained optimization problem, solved with **Augmented Lagrangian method**

- example for a single constraint (i.e. the decoder $f^θ(x)$  has a univariate input $x$)
  - aim: **optimize the ELBO, subject to the constraint $\int f^θ(x)dx = 0$**

    (= restrict $f^θ$ to a subspace such that  $\int f^θ(x)dx = 0$)

    

- <u>how to enforce this constraint</u> ? idea: **augment the ELBO with additional penalty term(s) which will be equal to 0 when integral constraints are fulfilled**

  - the resulting objective  function is not necessarily a lower bound but reduces to the ELBO once the constraints become fulfilled during optimization

  - 2 possible approaches

    1. *penalty method*: incorporate a penalty term $c (\int f^θ (x)dx)^ 2$ with fixed penalty $c$
       - disadvantage: for a fixed value of $c$ we do not have any guarantees that constraints would be fulfilled exactly
         

    2. *BDMM* (Platt and Barr 1988): introduce a penalty $λ \int f^θ (x)dx$ where $\lambda$ is treated as a parameter $\rightarrow$ analogous to the use of Lagrange multipliers

       - when optimizing $\lambda$, gradients updates will be  $λ^{t+1} =λ^t + η(\int f^θ (x)dx)$  rather than $λ^{t+1} =λ^t - η(\int f^θ (x)dx)$ 
         (i.e. ? gradient descent for the parameters, gradient ascent for the multipliers ?) ( = learning rate)

         

  - finally, combine the 2 terms in an hybrid constrained optimization objective =  ***MDMM*** (Platt and Barr 1988): $\underset{θ,\phi}{min}\{-L^{θ,\phi}+λ \int f^θ (x)dx+c (\int f^θ (x)dx)^ 2\}$ 

    - $\lambda$ is optimized, $c$ is kept constant, $L^{θ,\phi}$ is the ELBO for the VAE
    - more robust behaviour (theoretical + empirical evidence)
    - observation: using a sequence $c^1 ≤ · · · ≤ c^T$ can empirically lead to even faster convergence.

  - empirically, they show that using a fixed penalty does not necessarily lead to fulfilled constraints, whereas the oscillating behaviour of the BDMM leads to the integrals converging towards 0



visually: plot showing the traces for $\int f^θ(x_1, x_2)dx_2$ on a grid of $x_1$ values (each line corresponds to one grid point) over 100 000 iterations

- penalty method: the constraints have not been fulfilled with a fixed $c$
- BDMM: oscillating behaviour, integrals are slowly converging towards 0
- MDMM: combines the two penalties (using the same $c$ and same learning rate $η$) and leads to optimization which results in $≈ 0$ integral values much more quickly



- <u>how to handle **multiple constraints**</u> ? illustrated with the case of 2-dimensional input $(x_1, x_2)$
  - e.g. for $x_1$
    - for satisfying the constraint $\int f^θ_{12}(x_1, x_2)dx_2 = 0$ for all $x_1$ in some intervals:
      - introduce a Lagrange multiplier $\lambda_{x_1}(x_1)$ indexed by a continuous-valued $x_1$
      - the additional penalty term will be: $\int \lambda (x_1)(\int f^θ_{12}(x_1, x_2)dx_1)dx_2$ 
  - similarly for $x_2$, the ELBO will be augmented by: $\int \lambda (x_2)(\int f^θ_{12}(x_1, x_2)dx_2)dx_1$ 
  - in addition with to penalty terms $\lambda_1 \int f^θ_1(x_1)dx_1)$ and $\lambda_2 \int f^θ_2(x_2)dx_2)$



* <u>how to **estimate these integrals** in practice</u> ? $\rightarrow$ using either quadrature or Monte Carlo estimates



* <u>how to know **whether the constraints have been (approximately) satisfied**</u> ?
  - establish a desired tolerance threshold $ε$ 
  - evaluate the integrals after optimisation to make sure that all NNs have been constrained to the desired functional subspaces within the desired tolerance