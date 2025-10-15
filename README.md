# CNN2D-GDF: Convolutional Neural Network Analysis of 2D Gaussian Density Fields

This r- Applied Gaussian smoothing kernels of width 1 px and 2 px.
- CNN retrained for each case.
- Since no analytical Fisher bound exists for smoothed maps, the corresponding effective $k_{\max}$ was inferred from the CNN's measured error using:

  $$N_{\text{modes}} = \frac{A_{\min}^2 + A_{\min}A_{\max} + A_{\max}^2}{\sigma_A^2}$$ry contains code and methods for training convolutional neural networks (CNNs) on 2D Gaussian density field (GDF) maps to infer cosmological parameters from analytically defined power spectra. The project builds upon *Villaescusa-Navarro et al. (2020)* and explores the extent to which CNNs can extract information consistent with the Fisher information bound from purely Gaussian fields.

---

## Overview

The study focuses on the **AstroNone** toy dataset (as introduced in *Villaescusa-Navarro et al., 2020*), where 2D Gaussian density fields are generated from a simple analytical power spectrum

$$P(k) = \frac{A}{k}$$

with **A** as the only cosmological parameter.

The CNN model is trained to predict the value of **A** directly from the generated 2D Gaussian field maps. The experiment is designed to test whether deep learning can achieve the *Cramér–Rao lower bound* on parameter estimation.

---

## Data Generation and Pre-processing

- **Power spectrum definition:** $P(k) = A / k$
- **Parameter sampling:**  
  $A \sim \mathcal{N}(1.0, \sigma = 0.2)$, clipped to [0.8, 1.2].
- **Map generation:**  
  100 000 Gaussian density field maps of size $64 \times 64$ pixels were generated using the [**Pylians**](https://pylians3.readthedocs.io/en/master/) library.
- **Normalization:**
  - Field maps: Z-score normalization using the dataset mean and standard deviation.
  - Parameter A: Min-max scaling to [0, 1].
- **Dataset split:** 70 % train  |  15 % validation  |  15 % test.

Each map represents a realization of a Gaussian density field defined by its power spectrum amplitude A.  
The goal is to recover A using supervised regression with a CNN.

---

## CNN Architecture

- Framework: [**PyTorch**](https://pytorch.org/)  
- **Architecture:**
  - 5 convolutional layers  
  - Kernel size = 4, stride = 2, padding = 1  
  - Activation: LeakyReLU (α = 0.2)  
  - Flatten → fully connected → single output neuron (predicting A)
- **Loss function:**  
  Mean Squared Error (MSE)  

  $$L = \frac{1}{N}\sum_{i=1}^{N}(A_\text{true} - A_\text{NN})^2$$
- **Optimizer:** Adam  
- **Regularization:** Weight decay (no dropout)
- **Hyperparameter optimization:** [**Optuna**](https://optuna.org/)  
  - Algorithm: Tree-structured Parzen Estimator (TPE)  
  - Parameters tuned: learning rate, weight decay, number of filters per conv layer  
  - 50 trials × 200 epochs per trial

---

## Theoretical Validation — Fisher Matrix Formalism

To assess whether the CNN extracts the *maximum possible information*, its predictive uncertainty is compared with the theoretical limit derived from the **Fisher information matrix**.

For a single parameter A in $P(k) = A/k$:

$$F = \frac{N_{\text{modes}}}{2A^2}, \quad
\sigma(A) = A \sqrt{\frac{2}{N_{\text{modes}}}}$$

Averaging over $A \in [A_{\min}, A_{\max}]$:

$$\langle \sigma(A) \rangle =
\sqrt{ \frac{A_{\min}^2 + A_{\min}A_{\max} + A_{\max}^2}{1.5\,N_{\text{modes}}} }$$

The CNN’s empirical prediction error on A is compared against this theoretical bound to evaluate its optimality.

---

## Experiments and Analysis

### 1. **Original Gaussian Density Field Maps**
- CNN trained on unfiltered 64×64 maps.  
- Compared predicted errors in A with Fisher-matrix theoretical errors for $A \in [0.8, 1.2]$.
- Additional test sets of 20 000 samples each were created with fixed A = 0.82, 1.0, 1.18 to study prediction distributions.

### 2. **Fourier-Filtered Maps (Top-Hat k Filters)**
- Applied sharp cutoffs in Fourier space at:
  - $k_{\max} = 0.2,\ 0.15,\ 0.1$ ( $k_{\min} = 0$ in all cases )
- CNN retrained for each filter scale.
- Neural-network errors compared with Fisher predictions computed using Nmodes for each $k_{\max}$.

### 3. **Gaussian-Smoothed Maps**
- Applied Gaussian smoothing kernels of width 1 px and 2 px.
- CNN retrained for each case.
- Since no analytical Fisher bound exists for smoothed maps, the corresponding effective \( k_{\max} \) was inferred from the CNN’s measured error using:

  \[
  N_{\text{modes}} = \frac{A_{\min}^2 + A_{\min}A_{\max} + A_{\max}^2}{\sigma_A^2}
  \]

---

## Implementation Details

| Component | Description |
|------------|-------------|
| **Language** | Python 3 |
| **Libraries** | PyTorch · NumPy · Matplotlib · Pylians · Optuna |
| **Hardware** | GPU recommended (e.g., CUDA support) |
| **Metrics** | Mean Squared Error (MSE), Predicted vs True Scatter |

---

## Outcomes

- CNN recovers parameter A with precision close to the Fisher bound, confirming that it captures nearly all information contained in the Gaussian density fields.  
- As $k_{\max}$ decreases (or smoothing increases), fewer Fourier modes → larger error in A.  
- The network therefore acts as an effective *information extractor* consistent with theoretical expectations.

---

## Repository Structure

```
CNN2D-GDF/
├── src/
│   ├── Dataset.py ← data generation & preprocessing
│   ├── Network.py ← CNN architecture
│   ├── Training.py ← model training with Optuna tuning
│   ├── Testing.py ← model evaluation
├── results/ ← figures
├── requirements.txt
└── README.md
```

---

## References

- [Villaescusa-Navarro et al. 2025](https://arxiv.org/abs/2109.09747)  
- [Pylians Library](https://pylians3.readthedocs.io/en/master/)  
- [Optuna Hyperparameter Optimization](https://optuna.org/)  
- [PyTorch Framework](https://pytorch.org/)

---

## 🪐 Author

**Arnab Lahiry**  
Graduate Researcher in Astrophysics  
📍 IISER Tirupati, India | CCA, Flatiron Institute, New York, USA

