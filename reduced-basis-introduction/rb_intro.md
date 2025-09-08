---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# enable logging widget
%load_ext pymor.tools.jupyter
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
%%html
<style>
.rise-enabled .jp-RenderedHTMLCommon table {
         font-size: 150%;
}

.rise-enabled .jp-RenderedHTMLCommon p {
    font-size: 1.5rem;
}

.rise-enabled .jp-RenderedHTMLCommon li {
    font-size: 1.5rem;
}


.rise-enabled .jp-RenderedHTMLCommon h2 {
    font-size: 2.9rem;
    font-weight: bold;
}

.rise-enabled .jp-RenderedHTMLCommon h3 {
    font-size: 2.0rem;
    font-weight: bold;
}

.rise-enabled .jupyter-widget-Collapse-header {
    font-size: 1rem;
}

.rise-enabled .jupyter-widget-Collapse-header i{
    font-size: 1rem;
}

.rise-enabled .cm-editor {
    font-size: 1.25rem;
}
</style>
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

# Reduced Basis Methods with pyMOR

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Our Goal

We want to do model order reduction (MOR) for parametric problems.

This means:

- We are given a full-order model (FOM), usually a PDE model, which depends on some set of parameters $\mu \in \mathbb{R}^Q$.
- We can simulate/solve the FOM for any given $\mu$. But this is costly.
- We want to simulate the model for many different $\mu$.

**Task:**

- Replace the FOM by a surrogate reduced-order model (ROM).
- The ROM should be much faster to simulate/solve.
- The error between the ROM and FOM solution should be small and controllable.

Note: In this tutorial we will only cover the mere basics of reduced basis (RB) methods. The approach has been extended to other types of models (systems, non-linear, inf-sup stable, outputs, ...) and is largely independent of the specific choice of discretization method.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Building the FOM

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Thermal-block problem

Find $u(x,\mu)$ for $\mu\in\mathcal{P}$ such that

$$
\begin{align*}
-\nabla \cdot [d(x, \mu) \nabla u(x,\mu)] &= f(x) & x &\in \Omega, \\
u(x,\mu) &= 0 & x &\in \partial \Omega,
\end{align*}
$$

where $\Omega := [0,1]^2 = \Omega_1 \cup \Omega_2 \cup \Omega_3 \cup \Omega_4$, $f \in L^2(\Omega)$,


$$
d(x, \mu) \equiv \mu_i \quad x \in \Omega_i
$$

and $\mu \in [\mu_{\min}, \mu_{\max}]^4$.


```
        (0,1)-----------------(1,1)
        |            |            |
        |            |            |
        |     μ_2    |     μ_3    |
        |            |            |
        |            |            |
        |--------------------------
        |            |            |
        |            |            |
        |     μ_0    |     μ_1    |
        |            |            |
        |            |            |
        (0,0)-----------------(1,0)
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Setting up an analytical description of the thermal block problem

The thermal block problem already comes with pyMOR:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from pymor.basic import *
p = thermal_block_problem([2,2])
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

Our problem is parameterized:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
p.parameters
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Looking at the definition

We can easily look at the definition of `p` by printing its `repr`:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
p
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

It is easy to [build custom problem definitions](https://docs.pymor.org/latest/tutorial_builtin_discretizer.html).

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Weak formulation

Find $u(\mu) \in H^1_0(\Omega)$ such that

$$
\underbrace{\int_\Omega d(x, \mu(x)) \nabla u(x, \mu) \cdot \nabla v(x) \,dx}
    _{=:a(u(\mu), v; \mu)}
= \underbrace{\int_\Omega f(x)v(x) \,dx}
    _{=:\ell(v)}
    \qquad \forall v \in H^1_0(\Omega).
$$

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

### Galerkin projection onto finite-element space

Let $\mathcal{T}_h$ be an admissible triangulation of $\Omega$ and $V_h:=\mathcal{S}_{h,0}^1(\mathcal{T}_h)$ the corresponding space of piece-wise linear finite-element functions over $\mathcal{T}_h$ which vanish at $\partial\Omega$.
The finite-element approximation $u_h(\mu) \in V_h$ is then given by


$$
    a(u_h(\mu), v_h;\mu) = \ell(v_h)
    \qquad \forall v_h \in V_h.
$$

Céa's Lemma states that $u_h(\mu)$ is a quasi-best approximation of $u(\mu)$ in $V_h$:

$$
    \|\nabla u(\mu) - \nabla u_h(\mu)\|_{L^2(\Omega)}
    \leq \frac{\mu_{max}}{\mu_{min}} \inf_{v_h \in V_h} \|\nabla u(\mu) - \nabla v_h\|_{L^2(\Omega)}.
$$

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Linear system assembly

Let $\varphi_{h,1}, \ldots, \varphi_{h,n}$ be the finite-element basis of $\mathcal{S}_{h,0}^1(\mathcal{T}_h)$.
Let $A(\mu) \in \mathbb{R}^{n\times n}$, $\underline{\ell} \in \mathbb{R}^n$ be given by

$$
    A(\mu)_{j,i} := a(\varphi_{h,i}, \varphi_{h,j};\mu) \qquad
    \underline \ell_j := \ell(\varphi_{h,j}).
$$

Then with
$$
    u_h(\mu) = \sum_{i=1}^{n} \underline{u}_h(\mu)_i \cdot \varphi_{h,i},
$$

we get

$$
    A(\mu) \cdot \underline{u}_h(\mu) = \underline{\ell}.
$$

Note that $A(\mu)$ is a sparse matrix.

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### FOM assembly with pyMOR

We use the builtin discretizer `discretize_stationary_cg` to compute a finite-element discretization of the problem:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
fom, data = discretize_stationary_cg(p, diameter=1/100)
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

`fom` is a `Model`. It has the same `Parameters` as `p`:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
fom.parameters
```

```{code-cell} ipython3
from pymor.discretizers.builtin.cg import InterpolationOperator
diffusion_field_1 = InterpolationOperator(data['grid'], p.diffusion).as_vector(fom.parameters.parse({'diffusion': [1., 0.01, 0.1, 1.]}))
diffusion_field_2 = InterpolationOperator(data['grid'], p.diffusion).as_vector(fom.parameters.parse({'diffusion': [0.02, 1.5, 0.3, 0.01]}))
fom.visualize((diffusion_field_1, diffusion_field_2))
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Solving the FOM

To `solve` the FOM, we need to specify values for those parameters:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
U = fom.solve({'diffusion': [1., 0.01, 0.1, 1.]})
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

`U` is a `VectorArray`, an ordered collection of vectors of the same dimension:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
U
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

`U` only contains a single vector:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
len(U)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

For a time-dependent problem, `U` would have contained a time-series of vectors. `U` corresponds to the coefficient vector $\underline{u}_h(\mu)$.

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Looking at the solution

We can use the `visualize` method to plot the solution (we also plot the solution for the second diffusivity field here for comparison):

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
fom.visualize((U, fom.solve({'diffusion': [0.02, 1.5, 0.3, 0.01]})))
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Parameter separability

Remember the special form of $a(\cdot, \cdot; \mu)$:

$$
\begin{align}
    a(u, v; \mu) &:= \int_\Omega d(x, \mu) \nabla u(x) \cdot \nabla v(x) \,dx \\
    &:=\int_\Omega \Bigl(\sum_{q=1}^Q \mu_q \mathbf{1}_q(x)\Bigr) \nabla u(x) \cdot \nabla v(x) \,dx \\
    &:=\sum_{q=1}^Q  \ \underbrace{\mu_q}_{:=\theta_q(\mu)} \ \ 
        \underbrace{\int_\Omega \mathbf{1}_q(x) \nabla u(x) \cdot \nabla v(x) \,dx}_{=:a_q(u,v)}.
\end{align}
$$

Hence, $a(\cdot, \cdot; \mu)$ admits the affine decomposition

$$
    a(u, v; \mu) = \sum_{q=1}^Q \theta_q(\mu) \cdot a_q(u,v).
$$

Consequently, for $A(\mu)$ we have the same structure:

$$
    A(\mu) = \sum_{q=1}^Q \theta_q(\mu) \cdot A_q,
$$

where $(A_q)_{j,i} := a_q(\varphi_{h,i}, \varphi_{h,j})$.

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Parameter-separable FOM

Remember that our problem definition encoded the affine decomposition of $d(x, \mu)$ using a `LincombFunction`:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
p.diffusion
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

pyMOR's builtin `discretizer` automatically preserves this structure when assembling the system matrices. Let's look at the `fom` in more detail. The system matrix $A(\mu)$ is stored in the `Model`'s `operator` attribute:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
fom.operator
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We see that the `LincombFunction` has become a `LincombOperator` of `NumpyMatrixOperators`.
pyMOR always interprets matrices as linear `Operators`.

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

The right-hand side vector $\underline{\ell}$ is stored in the `rhs` attribute:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
fom.rhs
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

`fom.rhs` is not a `VectorArray` but a vector-like operator in order to support parameter-dependent right-hand sides. Only `Operators` can depend on a parameter in `pyMOR`, not `VectorArrays`.

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Other ways of obtaining the FOM

> Using an `analyticalproblem` and a `discretizer` is just one way
  to build the FOM.
>  
> Everything that follows works the same for a FOM that is built using an external PDE solver.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Reduced basis methods

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Projection-based MOR

Going back to the definition of the FOM

$$
    a(u_h(\mu), v_h; \mu) = \ell(v_h) \qquad \forall v_h \in V_h,
$$

our MOR approach is based on the idea of replacing the generic finite-element space $V_h$ by a problem-adapted reduced space $V_N\subset V_h$ of low dimension. I.e., we simply define our ROM by a Galerkin projection of the solution onto the reduced space $V_N$. So the reduced approximation $u_N(\mu) \in V_N$ of $u_h(\mu)\in V_h$ is given as the solution of

$$
    a(u_N(\mu), v_N; \mu) = \ell(v_N) \qquad \forall v_N \in V_N.
$$

Again, we can apply Céa's Lemma:

$$
    \|\nabla u_h(\mu) - \nabla u_N(\mu)\|_{L^2(\Omega)}
    \leq \frac{\mu_{max}}{\mu_{min}} \inf_{\color{red}v_N \in V_N} \|\nabla u_h(\mu) - \nabla v_N\|_{L^2(\Omega)}.
$$

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Does a good reduced space $V_N$ exist?

Thanks to Céa's lemma, our only job is to come up with a good low-dimensional approximation space $V_N$. In RB methods, our definition of 'good' is usually that we want to miminize the worst-case best-approximation error over all parameters $\mu \in \mathcal{P}$. I.e.,

$$
    \sup_{\mu \in \mathcal{P}} \inf_{v_N \in V_N} \|\nabla u_h(\mu) - \nabla v_N\|_{L^2(\Omega)}
$$

should not be much larger than the Kolmogorov $N$-width

$$
    d_N:=\inf_{\substack{V'_N \subset V_h\\ \dim V'_N \leq N}}\sup_{\mu \in \mathcal{P}} \inf_{v'_N \in V'_N} \|\nabla u_h(\mu) - \nabla v'_N\|_{L^2(\Omega)}.
$$

We won't go into details here, but it can be shown that for parameter-separable coercive problems like the thermal-block problem, the Kolmogorov $N$-widths decay at a subexponential rate, so good reduced spaces $V_N$ of small dimension $N$ do exist.

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Snapshot-based MOR

The question remains how to find a good $V_N$ algorithmically. RB methods are snapshot based which means that $V_N$ is constructed from 'solution snapshots' $u_{h}(\mu_i)$ of the FOM, i.e.

$$
    V_N \subset \operatorname{span} \{u_h(\mu_1), \ldots, u_h(\mu_n)\}.
$$

We will start by just randomly picking some snapshot parameters $\mu_i\in\mathcal{P}$ for $i=1,\dots,N$ with $N=10$ and set $V_N=\operatorname{span}\{u_h(\mu_1), \ldots, u_h(\mu_N)\}$:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
snapshot_parameters = p.parameter_space.sample_randomly(10)
snapshots = fom.solution_space.empty()
for mu in snapshot_parameters:
    snapshots.append(fom.solve(mu))
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

For numerical stability, it's a good idea to orthonormalize the basis:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
basis = gram_schmidt(snapshots)
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Is our basis any good?

Let's see if we actually constructed a good approximation space by computing the best-approximation error in this space for some further random solution snapshot. We can do so via orthogonal projection:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
mu = p.parameter_space.sample_randomly()
U_test = fom.solve(mu)
coeffs = basis.inner(U_test)
U_test_proj = basis.lincomb(coeffs)
fom.visualize((U_test, U_test_proj, U_test-U_test_proj),
              legend=('U', 'projection', 'error'),
              separate_colorbars=True)
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

Let's also compute the relative norm of the error:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
(U_test - U_test_proj).norm().item() / U_test.norm().item()
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Assembling the reduced system matrix

In order to compute a reduced solution, we need to choose a reduced basis $\psi_{1}, \ldots, \psi_{N}$ of $V_N$ and assemble the reduced system matrix $A_{N}(\mu) \in \mathbb{R}^{N\times N}$ and right-hand side vector $\underline{\ell}_N \in \mathbb{R}^N$ given by

$$
    A_N(\mu)_{j,i} := a(\psi_i, \psi_j; \mu) \qquad\text{and}\qquad
    \underline{\ell}_{N,j} := \ell(\psi_j).
$$

Expanding each basis vector $\psi_i$ w.r.t. the finite-element basis $\varphi_{h,i}$,

$$
    \psi_i = \sum_{k=1}^N \underline{\psi}_{i,k} \varphi_{h,k},
$$

we get

$$
    A_N(\mu)_{i,j} = \underline{\psi}_i^{\operatorname{T}} \cdot A(\mu) \cdot \underline{\psi}_j
$$

or more compactly written as

$$
    A_N(\mu) = \underline{V}^{\operatorname{T}} \cdot A(\mu) \cdot \underline{V},
$$

where the columns of $\underline{V}\in\mathbb{R}^{n\times N}$ are given by the basis vectors $\psi_{1},\dots,\psi_{N}$.

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

Thus, we could compute $A_N(\mu)$ in pyMOR using `W = fom.operator.apply(basis, mu=mu)` (multiplication from the right) and then using `basis.inner(W)` to multiply the basis from the left. We can use the `apply2` method as a (potentially more efficient) shorthand:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
A_N = fom.operator.apply2(basis, basis, mu=mu)
A_N.shape
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Note that, contrary to the finite-element system matrix $A(\mu)$, the reduced matrix $A_N(\mu)$ is a dense but small matrix.

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Assembling the reduced right-hand side

For the right-hand side we have

$$
    \underline{\ell}_{N,j} = \underline{\psi}_j^{\operatorname{T}} \cdot \underline{\ell}
$$

or

$$
    \underline{\ell}_{N} = \underline{V}^{\operatorname{T}} \cdot \underline{\ell},
$$

which we compute using `inner`:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
l_N = basis.inner(fom.rhs.as_vector())
l_N.shape
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Solving the reduced system

Finally, writing

$$
    u_N(\mu) = \sum_{i=1}^N \underline{u}_N(\mu)_i \cdot \psi_i
$$

we have

$$
    A_N(\mu) \cdot \underline{u}_N(\mu) = \underline{\ell}_N
$$

or equivalently

$$
    \underline{V}^{\operatorname{T}} \cdot A(\mu) \cdot \underline{V}\cdot \underline{u}_N(\mu) = \underline{V}^{\operatorname{T}} \cdot \underline{\ell}.
$$

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

So, let's solve the linear system and compare the reduced solution to the FOM solution:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import numpy as np
u_N = np.linalg.solve(A_N, l_N)
U_N = basis.lincomb(u_N.ravel())
U = fom.solve(mu)
fom.visualize((U, U_N, U-U_N),
              legend=('FOM', 'ROM', 'Error'),
              separate_colorbars=True)
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

Let's also compute the relative norm of the error:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
(U - U_N).norm().item() / U.norm().item()
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Automatic structure-preserving operator projection

For each new parameter $\mu$ we want to solve the ROM for, we have to assemble a new $A_N(\mu)$, which requires $\mathcal{O}(N^2)$ high-dimensional operations. This can significantly diminish the efficiency of our ROM. However, we can avoid this issue by exploiting the parameter separability of $A(\mu)$,

$$
    A(\mu) = \sum_{q=1}^Q \theta_q(\mu) \cdot A_q,
$$

which is inherited by $A_N(\mu)$:

$$
    A_N(\mu) = \sum_{q=1}^Q \theta_q(\mu) \cdot A_{N,q},
$$
where $(A_{N,q})_{i,j} = \underline{\psi}_i^{\operatorname{T}} \cdot A_q \cdot \underline{\psi}_j$ and $A_{N,q}=\underline{V}^{\operatorname{T}}\cdot A_q \cdot \underline{V}$.

Thus, we have to project all operators in `fom.operator.operators` individually and then later form a linear combination of these matrices.

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

This is getting tedious, so we let pyMOR do the work for us:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
op_N = project(fom.operator, basis, basis)
op_N
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

Similarly, we can project the right-hand side:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
rhs_N = project(fom.rhs, basis, None)
rhs_N
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

Now, we could assemble a matrix operator from `op_N` for a specific `mu` using the `assemble` method:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
op_N_mu = op_N.assemble(mu)
op_N_mu
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Then, we can extract it's system matrix:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
op_N_mu.matrix.shape
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

From that, we can proceed as before. However, it is more convenient, to use the operator's `apply_inverse` method to invoke an (`Operator`-dependent) linear solver with a given input `VectorArray` as right-hand side:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
u_N_new = op_N.apply_inverse(rhs_N.as_vector(), mu=mu)
u_N_new
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

Note that the result is a `VectorArray`. For `NumpyVectorArray` and some other `VectorArray` types, we can extract the internal data using the `to_numpy` method. We use it to check whether we arrived at the same solution:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
np.linalg.norm(u_N.ravel() - u_N_new.to_numpy().ravel())
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Projecting the entire Model

In pyMOR, ROMs are built using a `Reductor` which appropriately projects all of the `Models` operators and returns a reduced `Model` comprised of the projected `Operators`. Let's pick the most basic `Reductor`
available for a `StationaryModel`:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
reductor = StationaryRBReductor(fom, basis)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Every reductor has a `reduce` method, which builds the ROM:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
rom = reductor.reduce()
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

Let's compare the structure of the FOM and of the ROM

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
fom
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
rom
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Solving the ROM

To solve the ROM, we just use `solve` again,

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
u_rom = rom.solve(mu)
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

to get the reduced coefficients:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
u_rom
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

It is the same coefficient vector we have computed before:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
(u_rom - u_N_new).norm()
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

A high-dimensional representation is obtained from the `reductor`:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
U_rom = reductor.reconstruct(u_rom)
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Computing the MOR error

Let's compute the error again:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
U = fom.solve(mu)
ERR = U - U_rom
ERR.norm() / U.norm()
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

and look at it:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
fom.visualize(ERR)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Proper orthogonal decomposition for reduced basis construction

+++ {"slideshow": {"slide_type": "subslide"}}

### Extracting dominant directions in high-dimensional data

+++

Proper orthogonal decomposition (POD) is a standard method in order to compress a set of high-dimensional vectors by approximation in a suitable low-dimensional subspace. In other disciplines it is also known (up to minor adjustments) under names like Karhunen-Loève transform or principal component analysis (PCA). The approximation space $V_N$ is computed purely based on the available snapshot data (without any information about the FOM).

+++ {"slideshow": {"slide_type": "fragment"}}

Given a set of training snapshots $u^1=u_h(\mu_1),\dots,u^n=u_h(\mu_n)\in V_h$, the method aims at minimizing the mean squared projection error of the space $V_N$ on the training set:

$$
    V_N = \underset{\substack{V'_N \subset V_h\\ \dim V'_N \leq N}}{\operatorname{argmin}} \frac{1}{n}\sum\limits_{i=1}^{n} \lVert u^i - P_{V'_N}(u^i) \rVert^2,
$$

where $P_{V'_N}(u^i)$ denotes the orthogonal projection of $u^i$ onto $V'_N$.

+++ {"slideshow": {"slide_type": "fragment"}}

In pratice, POD is usually computed using a singular value decomposition (SVD) and returns an orthonormal basis of $V_N$ (i.e. no additional orthonormalization required).

+++ {"slideshow": {"slide_type": "subslide"}}

### POD applied in the thermalblock example

+++

We first compute a larger set of solution snapshots:

```{code-cell} ipython3
snapshot_parameters = p.parameter_space.sample_randomly(50)
snapshots = fom.solution_space.empty()
for mu in snapshot_parameters:
    snapshots.append(fom.solve(mu))
```

+++ {"slideshow": {"slide_type": "fragment"}}

Now we can run the `pod`-algorithm and construct a corresponding `reductor` as well as a reduced model:

```{code-cell} ipython3
pod_basis, svals = pod(snapshots, rtol=1e-7)
reductor = StationaryRBReductor(fom, pod_basis)
rom = reductor.reduce()

rom
```

+++ {"slideshow": {"slide_type": "subslide"}}

Let us look at the singular values (coming from the SVD and providing a measure for the approximation quality of a reduced space of the respective dimension) and their decay:

```{code-cell} ipython3
import matplotlib.pyplot as plt
plt.semilogy(svals / svals[0])
plt.show()
```

+++ {"slideshow": {"slide_type": "subslide"}}

And finally, let us compute the error for our sample parameter:

```{code-cell} ipython3
u_rom = rom.solve(mu)
U_rom = reductor.reconstruct(u_rom)
U = fom.solve(mu)
ERR = U - U_rom
ERR.norm() / U.norm()
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Is it actually faster?

Finally, we check if our ROM is really any faster than the FOM:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from time import perf_counter
mus = p.parameter_space.sample_randomly(10)
tic = perf_counter()
for mu in mus:
    fom.solve(mu)
t_fom = perf_counter() - tic
tic = perf_counter()
for mu in mus:
    rom.solve(mu)
t_rom = perf_counter() - tic
print(f'Speedup: {t_fom/t_rom}')
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Large datasets

> For very big datasets, for instance from instationary problems, a hierarchical and approximate variant of POD, called **HaPOD**, is available, see https://docs.pymor.org/main/autoapi/pymor/algorithms/hapod/index.html#module-pymor.algorithms.hapod

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Certified Reduced Basis Method

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Error estimator
Model order reduction introduces an additional approximation error which we need to control in order to be able to use a ROM as a reliable surrogate for a given FOM. While Céa's lemma provides a rigorous a priori bound, this error bound is not computable in general. Instead, we use a residual-based a posteriori error estimator. As in a posteriori theory for finite-element methods, we have:

$$
    \|\nabla u_h(\mu) - \nabla u_N(\mu)\|_{L^2(\Omega)}
    \leq \frac{1}{\mu_{min}} \sup_{v_h\in V_h} \frac{\ell(v_h) - a(u_N(\mu), v_h; \mu)}{\|\nabla v_h\|_{L^2(\Omega)}}.
$$

For this estimate to hold, it is crucial that we use the right norms. I.e., instead of the Euclidean norm of the coefficient vectors, which we have used so far, we need to use the $H^1$-seminorm. 

The inner product matrix of the $H^1$-seminorm is automatically assembled by pyMOR's builtin discretizer and available as `fom.h1_0_semi_product`. We can pass it as the `product`-argument to methods like `norm`, `inner` or `gram_schmidt` to perform these operations w.r.t. the correct inner product/norm. Further, we need a lower bound for the coercivity constant of $a(\cdot, \cdot; \mu)$.

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

Using this information, we can replace `StationaryRBReductor` by `CoerciveRBReductor`, which will add a reduction-error estimator to our ROM:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
basis = gram_schmidt(snapshots, product=fom.h1_0_semi_product)
reductor = CoerciveRBReductor(
    fom,
    basis,
    product=fom.h1_0_semi_product,
    coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameters)
)
rom = reductor.reduce()
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

We won't go into details here, but an 'offline-online decomposition' of the error estimator is possible similar to what we did for the projection of the system operator:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
rom.error_estimator.residual
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

Let's check if the estimator works:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
U = fom.solve(mu)
u_N = rom.solve(mu)
est = rom.estimate_error(mu).item()
err = (U - reductor.reconstruct(u_N)).norm(product=fom.h1_0_semi_product).item()
print(f'error: {err}, estimate: {est}')
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Greedy basis generation

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

So far, we have built the reduced space $V_N$ by just randomly picking snapshot parameters. A theoretically well-founded approach which leads to quasi-optimal approximation spaces it the so-called weak greedy algorithm. In the weak greedy algorithm, $V_N$ is constructed iteratively by enlarging $V_N$ by an element $u_h(\mu_{N+1})$ such that

$$ \inf_{v_N \in V_N} \|\nabla u_h(\mu_{N+1}) - \nabla v_N\|_{L^2(\Omega)}
\geq C \cdot \sup_{\mu \in \mathcal{P}}\inf_{v_N \in V_N} \|\nabla u_h(\mu) - \nabla v_N\|_{L^2(\Omega)}, $$

for some fixed constant $0 < C \leq 1$.

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

In RB methods, we find such a $\mu_{N+1}$ by picking the parameter for which the estimated reduction error is maximized. 

In order to make this maximization procedure computationally feasible, the infinite set $\mathcal{P}$ is replaced by a finite subset of training parameters:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
training_set = p.parameter_space.sample_uniformly(4)
len(training_set)
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

Given this training set, we can use `rb_greedy` to compute $V_N$. In order to start with an empty basis, we create a new reductor that, by default, is initialized with an empty basis:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
reductor = CoerciveRBReductor(
    fom,
    product=fom.h1_0_semi_product,
    coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameters)
)
greedy_data = rb_greedy(fom, reductor, training_set, max_extensions=20)
print(greedy_data.keys())
rom = greedy_data['rom']
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Testing the ROM

Let's compute the error again:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
mu = p.parameter_space.sample_randomly()
U = fom.solve(mu)
u_rom = rom.solve(mu)
ERR = U - reductor.reconstruct(u_rom)
ERR.norm(fom.h1_0_semi_product)
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

and compare it with the estimated error:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
rom.estimate_error(mu)
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Is it actually faster?

Finally, we check if our ROM is really any faster than the FOM:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from time import perf_counter
mus = p.parameter_space.sample_randomly(10)
tic = perf_counter()
for mu in mus:
    fom.solve(mu)
t_fom = perf_counter() - tic
tic = perf_counter()
for mu in mus:
    rom.solve(mu)
t_rom = perf_counter() - tic
print(f'Speedup: {t_fom/t_rom}')
```

**Important note:** In contrast to POD, the FOM is **not** solved for all training parameters, but only for selected ones!

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Some possible exercises

- Plot the MOR error vs. the dimension of the reduced space. (Use `reductor.reduce(N)` to project onto a sub-basis of dimension `N`.)
 
- Plot the speedup vs. the dimension of the reduced space.

- Compute the maximum/minimum efficiency of the error estimator over the parameter space.

- Try different numbers of subdomains.

+++ {"slideshow": {"slide_type": "slide"}}

## Instationary problems and POD-greedy

We are interested in solving a time-dependent problem of the form

$$
    (\partial_t u_h(\mu),v_h)_{L^2} + a(u_h(\mu), v_h;\mu) = \ell(v_h)
    \qquad \forall v_h \in V_h,
$$

where $u_h(\mu)\colon[0,T]\to V_h$ is now a function of time. Discretization in time using the implicit Euler scheme with step size $\Delta t>0$ yields

$$
    (m + \Delta t\cdot a)(u_{h,n+1}(\mu),v_h) = \Delta t\cdot l(v_h) + m(u_{h,n},v_h) \qquad \forall v_h \in V_h,
$$

where $m(u,v)=(u,v)_{L^2}$ is a mass operator. A single step of the time stepping scheme therefore looks similar to an elliptic problem and we can - given a suitable reduced basis - apply projection-based MOR as before.

+++

### How to compute a reduced basis?

In the time-dependent case, the solution to every parameter consists of a whole trajectory in time. Hence, running a greedy algorithm and adding the complete trajectory to the reduced basis in every step is not a good idea! Moreover, due to error accumulation over time, selecting a suitable time step in a greedy fashion is also tricky (some heuristics exist though), i.e. doing a greedy in parameter and time is usually difficult. On the other hand, computing a POD of many (potentially long) trajectories can become computationally infeasible very quickly as well.

The POD-greedy method combines both approaches by **compressing the projection error** of the trajectory corresponding to the **selected parameter** in every step of the greedy algorithm. Typically, only a small number of modes (often actually only a single one) is then added to the reduced basis in every greedy iteration.

In pyMOR, the greedy algorithm performs this compression of projection errors automatically (within the `extend_basis` method of the respective reductor) without requiring any adjustments.

+++

### Example of a parametric heat equation

+++

Consider on the spatial domain $\Omega=[0,1]^2$ and the time interval $[0,1]$ the following parabolic equation with a high-conductivity and two parametrized channels ($\mathcal{P}=[1,100]$):
$$
    \partial_t u(\mu) - \nabla (d(\mu)\nabla u(\mu)) = f(t)
$$
with parametric diffusivity
$$
    d(\mu) = 1 + \underbrace{99\cdot\mathbf{1}_{(0.45,0.55)\times(0,0.7)}}_{\text{high-conductivity channel}} + (\mu - 1)\cdot(\underbrace{\mathbf{1}_{(0.35,0.4)\times(0.3,1)}+\mathbf{1}_{(0.6,0.65)\times(0.3,1)}}_{\text{parametrized channels}}),
$$
time-dependent right-hand side
$$
    f(t) = 100\cdot\sin(10\pi t)
$$
and Neumann boundary condition
$$
    \partial_n u(\mu, t, (x,y))=\begin{cases}-1000 & 0.45 < x < 0.55\\0 & \text{else}\end{cases}
$$
at the bottom of the domain ($\Gamma_{\text{bottom}}=[0,1]\times\{0\}$) and homogeneous Dirichlet boundary conditions everywhere else.
The initial condition is set to
$$
    u(\mu, 0) = 10\cdot \mathbf{1}_{(0.45,0.55)\times(0,0.7)}.
$$

+++

We create this example using pyMOR's builtin discretizer (the FOM for this example is also directly available in `pymor/models/examples.py`, however we also need the problem definition and the discretization data here):

```{code-cell} ipython3
grid_intervals = 50
nt = 50

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction, LincombFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.discretizers.builtin import discretize_instationary_cg
from pymor.parameters.functionals import ExpressionParameterFunctional

# setup analytical problem
problem = InstationaryProblem(

    StationaryProblem(
        domain=RectDomain(top='dirichlet', bottom='neumann'),

        diffusion=LincombFunction(
            [ConstantFunction(1., dim_domain=2),
             ExpressionFunction('(0.45 < x[0] < 0.55) * (x[1] < 0.7) * 1.',
                                dim_domain=2),
             ExpressionFunction('(0.35 < x[0] < 0.40) * (x[1] > 0.3) * 1. + '
                                '(0.60 < x[0] < 0.65) * (x[1] > 0.3) * 1.',
                                dim_domain=2)],
            [1.,
             100. - 1.,
             ExpressionParameterFunctional('top[0] - 1.', {'top': 1})]
        ),

        rhs=ConstantFunction(value=100., dim_domain=2) * ExpressionParameterFunctional('sin(10*pi*t[0])', {'t': 1}),

        dirichlet_data=ConstantFunction(value=0., dim_domain=2),

        neumann_data=ExpressionFunction('(0.45 < x[0] < 0.55) * -1000.', dim_domain=2),
    ),

    T=1.,

    initial_data=ExpressionFunction('(0.45 < x[0] < 0.55) * (x[1] < 0.7) * 10.', dim_domain=2)
)

# discretize using continuous finite elements
fom, data = discretize_instationary_cg(analytical_problem=problem, diameter=1./grid_intervals, nt=nt)

parameter_space = fom.parameters.space(1, 100)
```

Let us look at the diffusivity field for two different parameters:

```{code-cell} ipython3
from pymor.discretizers.builtin.cg import InterpolationOperator
diffusion_field_1 = InterpolationOperator(data['grid'], problem.stationary_part.diffusion).as_vector(fom.parameters.parse({'top': [2.]}))
diffusion_field_2 = InterpolationOperator(data['grid'], problem.stationary_part.diffusion).as_vector(fom.parameters.parse({'top': [100.]}))
fom.visualize((diffusion_field_1, diffusion_field_2))
```

And the corresponding solutions:

```{code-cell} ipython3
fom.visualize((fom.solve(2.), fom.solve(100.)))
```

We make use of the `ParabolicRBReductor`:

```{code-cell} ipython3
coercivity_estimator = ExpressionParameterFunctional('1.', fom.parameters)
reductor = ParabolicRBReductor(fom, product=fom.h1_0_semi_product, coercivity_estimator=coercivity_estimator)
```

And apply the `rb_greedy` as before (internally, the greedy method performs a POD on the time trajectory that returns a single mode):

```{code-cell} ipython3
training_set = parameter_space.sample_uniformly(50)
greedy_data = rb_greedy(fom, reductor, training_set, max_extensions=10)
rom = greedy_data["rom"]
```

**Note:** It is also possible to add more POD modes to the reduced basis in every greedy step by passing `extension_params={'pod_modes': k}` to the `rb_greedy`-method, where `k` is the number of modes to use.

+++

Let us look at a reduced solution and the speedup:

```{code-cell} ipython3
mu = parameter_space.sample_randomly()
tic = perf_counter()
U = fom.solve(mu)
t_fom = perf_counter() - tic
tic = perf_counter()
U_RB = reductor.reconstruct(rom.solve(mu))
t_rom = perf_counter() - tic
print(f"Speedup: {t_fom / t_rom}")
fom.visualize((U, U_RB, U - U_RB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
              separate_colorbars=True)
```

## Nonlinear Problems with POD-DEIM

+++

For stationary nonlinear problems, we need to solve nonlinear discrete equations of the form

$$ A(\underline{u}_h(\mu); \mu) = \underline{\ell}.$$

We can do Galerkin projection as usual, which leads to

$$ \underline{V}^{\operatorname{T}} \cdot A(\underline{V}\cdot \underline{u}_N(\mu); \mu) = \underline{V}^{\operatorname{T}} \cdot \underline{\ell}.$$

However, we cannot "precompute" $\underline{V}^{\operatorname{T}} \cdot A(\underline{V} \cdot (\ldots); \mu): \mathbb{R}^N \to \mathbb{R}^N$, even for a single $\mu$!

+++

### (Discrete) Empirical Interpolation Method ((D)EIM)

- Only evalutate
  $$A(\cdot; \mu)_{i_m} \quad\text{for $M$ interpolation DOFs} \quad i_1, \ldots i_M$$
  For FD, FV, FEM, et al., this only requires local low-dimensional computations!

- Approximate
  $$A(\cdot; \mu) \approx \sum_{m=1}^M \hat\psi_m \cdot A(\cdot; \mu)_{i_m},$$
  for Lagrange interpolation basis $\hat\psi_1, \ldots \hat\psi_M$. (In practice, different basis leading to triangular interpolation matrix is used.)

- Compute $i_1, \ldots, i_M$ and $\hat\psi_1, \ldots, \hat\psi_M$ offline from data (EI-Greedy).

- Solve
  $$\underline{V}^{\operatorname{T}} \cdot \sum_{m=1}^M \hat\psi_m \cdot A(\underline{V}\cdot \underline{u}_N(\mu); \mu)_{i_m} = \underline{V}^{\operatorname{T}} \cdot \underline{\ell}$$

- Offline-online decomposition by precomputing all products $(\psi_i, \hat\psi_j)$ and storing the coefficients of $\psi_i$ in "neighborhoods" of the $i_m$ (recall that $\psi_1,\dots,\psi_N$ are the columns of $\underline{V}$, i.e. the basis vectors of the reduced space).

+++

### Example

We solve a parameterized version of the FEniCS [nonlinear Poisson](https://fenics.readthedocs.io/projects/dolfin/en/2017.2.0/demos/nonlinear-poisson/python/demo_nonlinear-poisson.py.html) demo:

$$ 
\begin{align} 
-\nabla \cdot \left[(1 + c\cdot u(x,y;\mu)^2) \nabla u(x,y;\mu)\right] &= x\cdot \sin(y) & (x,y) &\in (0,1) \times (0,1), \\
u(1, y) &= 1, \\
\nabla u(0, y) \cdot n = \nabla u(x, 0) \cdot n = \nabla u(x, 1) \cdot n &= 0,
\end{align}$$

where $c \in [0, 1000]$ is our parameter.

```{code-cell} ipython3
import dolfin as df
mesh = df.UnitSquareMesh(100, 100)
V = df.FunctionSpace(mesh, 'CG', 2)

class DirichletBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < df.DOLFIN_EPS and on_boundary
bc = df.DirichletBC(V, 1., DirichletBoundary())

u = df.Function(V)
v = df.TestFunction(V)
f = df.Expression('x[0]*sin(x[1])', degree=2)
c = df.Constant(1.)
F = df.inner((1 + c*u**2)*df.grad(u), df.grad(v))*df.dx - f*v*df.dx
```

### pyMOR Wrapping
To use pyMOR, we need to wrap the FEniCS objects as pyMOR `Operators`. Then, we can create generic `StationaryModel` from it:

```{code-cell} ipython3
from pymor.bindings.fenics import FenicsOperator, FenicsVectorSpace, FenicsVisualizer
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import VectorOperator

space = FenicsVectorSpace(V)
op = FenicsOperator(F, space, space, u, (bc,),
                    parameter_setter=lambda mu: c.assign(mu['c'].item()),
                    parameters={'c': 1},
                    solver_options={'inverse': {'type': 'newton', 'rtol': 1e-6}})
rhs = VectorOperator(op.range.zeros())

fom = StationaryModel(op, rhs, visualizer=FenicsVisualizer(space))

parameter_space = fom.parameters.space((0, 1000.))
```

We can solve the model and visualize the solution:

```{code-cell} ipython3
U = fom.solve(1.)
fom.visualize(U)
```

### Reduction

First, we need to generate snapshot data. We directly call into pyMOR's Newton algorithm, to also get the Newton residuals as additional data:

```{code-cell} ipython3
U = fom.solution_space.empty()
residuals = fom.solution_space.empty()
for mu in parameter_space.sample_uniformly(10):
    UU, data = newton(fom.operator, fom.rhs.as_vector(), mu=mu, rtol=1e-6, return_residuals=True)
    U.append(UU)
    residuals.append(data['residuals'])
```

`fom.operator` vanishes on the solution. So we generate the interpolation data only from the resiudals:

```{code-cell} ipython3
dofs, cb, _ = ei_greedy(residuals, rtol=1e-4)
ei_op = EmpiricalInterpolatedOperator(fom.operator, collateral_basis=cb, interpolation_dofs=dofs, triangular=True)
```

We compute a POD basis:

```{code-cell} ipython3
rb, svals = pod(U, rtol=1e-4)
```

Finally, we replace the operator with the interpolated operator and project:

```{code-cell} ipython3
fom_ei = fom.with_(operator=ei_op)
reductor = StationaryRBReductor(fom_ei, rb)
rom = reductor.reduce()
rom = rom.with_(operator=rom.operator.with_(solver_options=fom.operator.solver_options))
```

Let's see if it works:

```{code-cell} ipython3
mu = parameter_space.sample_randomly()
U = fom.solve(mu)
U_rom = reductor.reconstruct(rom.solve(mu))
fom.visualize(U - U_rom)
print(f"Relative error: {(U - U_rom).norm().item() / U.norm().item()}")
```

### Is it faster?

```{code-cell} ipython3
from time import perf_counter
mus = parameter_space.sample_randomly(10)
tic = perf_counter()
for mu in mus:
    fom.solve(mu)
t_fom = perf_counter() - tic
tic = perf_counter()
for mu in mus:
    rom.solve(mu)
t_rom = perf_counter() - tic
print(f'Speedup: {t_fom / t_rom}')
```
