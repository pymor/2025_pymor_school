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

<!-- #region slideshow={"slide_type": "slide"} -->
# Head-related Transfer Function Modelling from Data


## Overview
- Data: Head-related Transfer Functions
- Theory: Data-driven Balanced Truncation (= Eigensystem Realization Algorithm)
- Practice: Implementation in pyMOR
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Head-related Transfer Functions (HRTFs)
- characterizes how the ear receives sound from a point in space (in free field conditions)
- HRTF encodes the head and pinnae geometries
- conventionally, it depends on angles of incident (spherical coordinates)
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### MIT KEMAR dummy head dataset
[Source](https://sound.media.mit.edu/resources/KEMAR.html)
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
import numpy as np

data = np.load('files/KEMAR.npy', allow_pickle=True).item(0)
ir = data['ir']
fs = data['fs'][0]
```

```python slideshow={"slide_type": "slide"}
n, p, m = ir.shape  # n: number of samples, p: number of outputs, m: number of inputs
fs  # sampling rate

print(f'number of inputs:\t{m}')
print(f'number of outputs:\t{p}')
print(f'number of samples:\t{n}')
print(f'sampling rate:\t\t{fs}')
print(f'sampling duration:\t{1000*n/fs:.2f} ms')
```

```python slideshow={"slide_type": "slide"}
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 4.5)
```

```python slideshow={"slide_type": "slide"}
input_idx = 469

t = np.arange(ir.shape[0])/fs * 1000

fig, axes = plt.subplots(1, 2)
fig.suptitle(f'Head-Related Impulse Response (input {input_idx + 1})')
for i, ax in enumerate(axes):
    ax.plot(t, ir[:, i, input_idx], c='grey', marker='o', mec='k', markersize='3')
    ax.set(xlabel=r'Time (ms)', ylabel='Amplitude', title=f'{"left" if i == 0 else "right"} ear', xlim=(t[0], t[-1]), ylim=(-0.75, 0.75))
```

```python slideshow={"slide_type": "slide"}
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

ax.scatter(*np.squeeze(data['rpos']).T, color='g', depthshade=False, label='receiver location (ears)')

azim = (data['spos'][:, 0]) * np.pi / 180
elev = (data['spos'][:, 1] - 90) * np.pi / 180
r = data['spos'][:, 2]
x, y, z = r*np.sin(elev)*np.cos(azim), r*np.sin(elev)*np.sin(azim), r*np.cos(elev)

center_idx = (azim == 0) & (elev == -np.pi/2)
plot_idx = ~center_idx
plot_idx[input_idx] = False
ax.scatter(x[plot_idx], y[plot_idx], z[plot_idx], label='source location')
ax.scatter(x[center_idx], y[center_idx], z[center_idx], color='r', marker='s', label='center', depthshade=False)
ax.scatter(x[input_idx], y[input_idx], z[input_idx], color='m', marker='s', label=f'input {input_idx + 1}', depthshade=False)
ax.view_init(azim=190, elev=5)
ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5), title='Measurement Geometry', xlabel='x', ylabel='y', zlabel='z')
ax.set_aspect('equal')
fig.tight_layout()
_ = fig.legend()
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Aim: Identify a reduced order LTI-system from the measurements
Consider discrete-time state equations:

$$
\begin{align}
  x_{k + 1} & = A x_k + B u_k \\
  y_k & = C x_k + D u_k,
\end{align}
$$

with $A \in \mathbb{R}^{n \times n}$, $B \in \mathbb{R}^{n \times m}$, $C \in \mathbb{R}^{p \times n}$, $D \in \mathbb{R}^{p \times m}$,
where $m \in \mathbb{N}$ is the number of inputs, $p \in \mathbb{N}$ the number of outputs and $n \in \mathbb{N}$ the state dimension.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Recap: Balanced Truncation (square-root version)

1. Compute Cholesky factors

   $$
   \begin{align}
     Z_P Z_P^T & = P \\
     Z_Q Z_Q^T & = Q
   \end{align}
   $$

   of the solutions of the Stein (discrete-time Lyapunov) equations

   $$
   \begin{align}
     A P A^T - P - B B^T & = 0 \\
     A^T Q A - Q + C^T C & = 0
   \end{align}
   $$
2. Compute balancing transformations from a (truncated) SVD of the product of factors

   $$
   \begin{align}\label{eq:hsv}\tag{1}
     Z_P^T Z_Q = U \Sigma V^T =
     \begin{bmatrix}
       U_r & U_\ast
     \end{bmatrix}
     \begin{bmatrix}
       \Sigma_r & 0 \\
       0 & \Sigma_\ast
      \end{bmatrix}
      \begin{bmatrix}
        V_r^T \\
        V_\ast^T
      \end{bmatrix}
   \end{align}
   $$

   Form transformations

   $$
   \begin{align}
     T_1 & = Z_Q V \Sigma^{-1/2} \\
     T_2 & = Z_P U \Sigma^{-1/2}
   \end{align}
   $$

   that balance the system, i.e.

   $$
   \begin{equation}
     T_1^T P T_1 = T_2^T Q T_2 = \Sigma,
   \end{equation}
   $$
3. Transform the system with balancing transformations

   $$
   \begin{align}
     T_1^T A T_2 & =
     \begin{bmatrix}
       A_{1,1} & A_{1,2} \\
       A_{2,1} & A_{2,2}
     \end{bmatrix}
     &
     T_1^T B & =
     \begin{bmatrix}
       B_1 & B_2
     \end{bmatrix}
     &&
     C T_2 =
     \begin{bmatrix}
       C_1 \\
       C_2
     \end{bmatrix}
   \end{align}
   $$

   and truncate such that the reduced system is given by

   $$
   \begin{align}
     \left(A_{1,1}, ~B_1, ~C_1, ~D\right)
     = \left(\Sigma_r^{-1/2} V_r^T Z_Q^T A Z_P U \Sigma^{-1/2},
       ~\Sigma_r^{-1/2} V_r^T Z_Q^T B,
       ~C Z_P U \Sigma^{-1/2},
       ~D\right)
   \end{align}
   $$
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Markov parameters

\begin{equation}
h_0=D,\qquad h_i=\left.\frac{\mathrm{d}^i}{\mathrm{d}s^i}H(s)\,\right|_{s=\infty}=CA^{i-1}B
\end{equation}
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### The Hankel Operator
The Hankel matrix of a system is defined as

\begin{equation}
  \mathcal{H}=
  \begin{bmatrix}
    CB & CAB & \cdots & CA^{s-1}B\\
    CAB & CA^2B &\cdots & CA^sB \\
    \vdots & \vdots &  &\vdots\\
    CA^{s-1}B & CA^{s} & \cdots & CA^{2s-2}B
  \end{bmatrix}=
  \underbrace{\begin{bmatrix}
                C\\
                \vdots\\
                CA^{s-1}
              \end{bmatrix}}_{=:\mathcal{O}}
  \underbrace{
  \begin{bmatrix}
    B&\cdots&A^{s-1}B
  \end{bmatrix}}_{=: \mathcal{C}}.
\end{equation}
The singular values of $\mathcal{H}$ are called the Hankel singular values.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## A data-driven formulation

- The diagonal entries of $\Sigma$ from (1) are equivalent to the Hankel singular values:

  $$
  Z_P Z_P^T Z_Q Z_Q^T
  = P Q
  = \mathcal{C} \mathcal{C}^T \mathcal{O}^T \mathcal{O}
  $$
- Observe that $\mathcal{H}$ can be constructed entirely from data (non-intrusively)
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Eigensystem Realization Algorithm
**Input:** Impulse response (Markov parameter) measurements $h$
1. Form Hankel matrix
2. Compute SVD

   $$
   \mathcal{H} = U \Sigma V^T =
   \begin{bmatrix}
     U_r & U_\ast
   \end{bmatrix}
   \begin{bmatrix}
     \Sigma_r & 0 \\
     0 & \Sigma_\ast
   \end{bmatrix}
   \begin{bmatrix}
     V_r^T \\
     V_\ast^T
   \end{bmatrix}
   $$
3. Choose approximate Gramian factors

   $$
   \begin{align}
     \mathcal{O} = U_r \Sigma_r^{1/2}, &&
     \mathcal{C} = \Sigma_r^{1/2} V_r^T
   \end{align}
   $$
4. Construct a (partial) realization

   $$
   \begin{align}
     &
     A = \mathcal{O}_{f}^{\dagger} \mathcal{O}_{l},
     &&
     B = \mathcal{C}
     \begin{bmatrix}
       I_{m} \\
       0
     \end{bmatrix},
     &
     C =
     \begin{bmatrix}
       I_p & 0
     \end{bmatrix}
     \mathcal{O},
     &&
     D = h_0
   \end{align}
   $$

**Output:** Partial realization $(A,B,C,D)$
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
from pymor.operators.numpy import NumpyHankelOperator

H = NumpyHankelOperator(np.concatenate((ir, np.zeros_like(ir)[1:])))
print(H)
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Tangential projections
Computing the SVD might be infeasible for large $\mathcal{H}$.

- The impulse response data $h$ contains $n = 511 = 2 s - 1$ samples after zero padding
- Rank of $\mathcal{H}$ is bounded

  $$
  \operatorname{rank}(\mathcal{H})
  \leq \min\{m s, p s\}
  = 512
  $$
- We can reduce the size of $\mathcal{H}$ with **tangential projections** by computing a skinny SVD of

  $$
  \Theta_\mathrm{in} =
  \begin{bmatrix}
    h_1 \\
    \vdots \\
    h_s
  \end{bmatrix}
  = U_\mathrm{in} \Sigma_\mathrm{in} V_\mathrm{in}^T
  \in \mathbb{R}^{p s \times m}
  $$

  and projecting the data with $V_\mathrm{in} \in \mathbb{R}^{m \times p s}$

  $$
  \hat{h}_i = h V_\mathrm{in} \in \mathbb{R}^{p \times p s}
  $$
- Let $\mathcal{V} = \operatorname{blk diag}\left(V,\,\dots,\,V\right) \in \mathbb{R}^{ms\times ps^2}$.
  Since $V$ is unitary, $\mathcal{V}$ is unitary.
  It follows that

  $$
  \sigma(\mathcal{H})
  = \sigma(\mathcal{H} \mathcal{V})
  = \sigma(\mathcal{\hat{H}})
  $$
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Using pyMOR
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
from pymor.reductors.era import ERAReductor

ERAReductor?
```

```python slideshow={"slide_type": "slide"}
ERAReductor.reduce??
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Exercise 1: Construct a partial realization
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
era = ERAReductor(ir, sampling_time=1/fs, force_stability=True, feedthrough=None)
```

```python slideshow={"slide_type": "slide"}
# your code here
```

```python slideshow={"slide_type": "slide"}
flim = np.array((200, fs/2))
wlim = 2 * np.pi * flim / fs
ylim = (-40, 20)

from pymor.algorithms.to_matrix import to_matrix
from pymor.operators.numpy import NumpyMatrixOperator

# slice the model for plotting
sliced_roms = dict()
for key, rom in roms.items():
    sliced_roms[key] = rom.with_(B=NumpyMatrixOperator(to_matrix(rom.B)[:, input_idx].reshape(-1, 1)),
                                 D=NumpyMatrixOperator(to_matrix(rom.D)[:, input_idx].reshape(-1, 1)))

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
ax = ax.T.ravel()
ax[0].semilogx(np.fft.rfftfreq(256, 1/fs), 20*np.log10(np.abs(np.fft.rfft(ir[:, 0, input_idx]))), c='grey', marker='o', mec='k', markersize='3', label='data')
ax[2].semilogx(np.fft.rfftfreq(256, 1/fs), 20*np.log10(np.abs(np.fft.rfft(ir[:, 1, input_idx]))), c='grey', marker='o', mec='k', markersize='3', label='data')
ax = ax.reshape(-1, 1)
for order, rom in sliced_roms.items():
    rom.transfer_function.bode_plot(wlim, ax=ax, dB=True, Hz=True, label=f'r = {order}')
ax = np.squeeze(ax)
ax[0].set(xlabel='Frequency (Hz)', title='left ear', xlim=flim, ylim=ylim)
ax[0].legend()
ax[0].grid()
fig.delaxes(ax[1])
ax[2].set(xlabel='Frequency (Hz)', title='right ear', xlim=flim, ylim=ylim)
ax[2].legend()
ax[2].grid()
fig.delaxes(ax[3])


fig, axes = plt.subplots(1, 2)
fig.suptitle(f'Error Impulse Response (input {input_idx})')
for rom in sliced_roms.values():
    rom = rom.with_(T=256)
    hrir = rom.impulse_resp()[1:, :, 0] / fs
    error = ir[..., input_idx] - hrir
    for i, ax in enumerate(axes):
        ax.plot(t, error[:, i], marker='o', mec='k', markersize='3', label='data')
        ax.set(xlabel=r'Time (s)', ylabel='Amplitude', title=f'{"left" if i == 0 else "right"} ear', ylim=(-0.075, 0.075))
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Exercise 2: Construct a partial realization with tangential projections
<!-- #endregion -->
