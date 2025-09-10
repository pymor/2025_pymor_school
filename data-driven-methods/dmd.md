---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Dynamic Mode Decomposition (DMD)


Consider a discrete-time system

$$
x_{k + 1} = T(x_k)
$$

where $x_k \in \mathbb{R}^n$.


Suppose we have data as sequential snapshots $x_0, x_1, \ldots, x_m \in \mathbb{R}^n$.

The idea is to approximate the dynamics with a linear system

$$
\tilde{x}_{k + 1} = A \tilde{x}_k.
$$


Then it should be that $x_{k + 1} \approx A x_k$, which we can write as

$$
\begin{align*}
  \underbrace{
    \begin{bmatrix}
      x_1 & x_2 & \cdots & x_m
    \end{bmatrix}
  }_{=: Y}
  & \approx
  A
  \underbrace{
    \begin{bmatrix}
      x_0 & x_1 & \cdots & x_{m - 1}
    \end{bmatrix}
  }_{=: X} \\
  Y & \approx A X
\end{align*}
$$


Thus, we could determine $A$ by solving the least squares problem

$$
A =
\operatorname*{arg\,min}_{\mathcal{A} \in \mathbb{R}^{n \times n}}
\lVert Y - \mathcal{A} X \rVert_F^2,
$$

i.e.

$$
A = Y X^+.
$$


It can be extended to input-output systems (ioDMD)

$$
\begin{align*}
  x_{k + 1} & = A x_k + B u_k, \\
  y_k & = C x_k + D u_k,
\end{align*}
$$

with input data $u_0, u_1, \ldots, u_m$,
state data $x_0, x_1, \ldots, x_m$,
and output data $y_0, y_1, \ldots, y_m$.


## Exercise

Apply `pymor.algorithms.dmd.dmd`.


# Time-delay embeddings


Recall that the transfer function of an continuous-time LTI system

$$
H(s) = C (s I - A)^{-1} B
$$

is a stricly proper rational function.
Therefore, it can be represented as a ratio of polynormials:

$$
H(s) = \frac{p_{n - 1} s^{n - 1} + \cdots + p_1 s + p_0}{s^n + q_{n - 1} s^{n - 1} + \cdots + q_1 s + q_0}.
$$

Then, from $Y(s) = H(s) U(s)$, it follows that

$$
(s^n + q_{n - 1} s^{n - 1} + \cdots + q_1 s + q_0) Y(s) = (p_{n - 1} s^{n - 1} + \cdots + p_1 s + p_0) U(s).
$$

Applying the inverse Laplace transformation gives

$$
y^{(n)}(t) + q_{n - 1} y^{(n - 1)}(t) + \cdots + q_1 y'(t) + q_0 y(t) =
p_{n - 1} u^{(n - 1)}(t) + \cdots + p_1 u'(t) + p_0 u(t).
$$

Similar procedure works for discrete-time systems:

$$
y_{k + n} + q_{n - 1} y_{k + n - 1} + \cdots + q_1 y_{k + 1} + q_0 y_k =
p_{n - 1} u_{k + n - 1} + \cdots + p_1 u_{k + 1} + p_0 u_k.
$$
