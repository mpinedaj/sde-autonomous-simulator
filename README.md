# Autonomous Stochastic Differential Equation Simulator

An interactive simulation application designed to compute, plot, and analyze multiple numerical trajectories of an **Autonomous Stochastic Differential Equation (SDE)** using the **Euler Method (Euler-Maruyama approximation)**. The tool evaluates the system against its exact analytical mean and variance over a defined time interval.

## 🚀 Overview

This repository contains a computational simulator developed to model autonomous SDEs defined by the general linear form:

$$dX_t = (aX_t + b)dt + (cX_t + d)dB_t$$

Where:
* $(aX_t + b)$ represents the **drift coefficient**.
* $(cX_t + d)$ represents the **diffusion coefficient**.
* $B_t$ is a standard **Brownian Motion** process discretized via independent Gaussian increments: $\Delta B_i \sim N(0, \Delta t) = \sqrt{\Delta t} \cdot \mathcal{N}(0, 1)$.

The application simulates $M$ independent paths simultaneously, plotting them in a unified space alongside the exact theoretical mean and variance functions for statistical validation.

---

## 🛠️ Features

* **Interactive Configuration Menu:** Easily customize simulation constraints:
  * System coefficients ($a, b, c, d$)
  * Initial condition ($X_0$)
  * Time interval ($0, T$) and step size ($\Delta t$)
  * Total number of trajectories ($M$)
* **Stochastic Solvers:** High-fidelity implementation of the Euler numerical discretization scheme.
* **Statistical Overlay:** Real-time generation and rendering of exact analytical benchmarks (Expected Value and Variance) overlaid directly onto the simulated trajectory cloud.
* **Data Visualization:** Clean, color-coded, professional-grade plots optimized for research reporting.

---

## 📊 Discretization Scheme

The Brownian paths are generated sequentially starting from $B_0 = 0$ using:

$$B_{t_i} = B_{t_{i-1}} + \sqrt{\Delta t} \cdot \epsilon_i \quad \text{where } \epsilon_i \sim N(0,1)$$

Each path of the autonomous SDE is advanced iteratively through the Euler scheme:

$$X_{t_i} = X_{t_{i-1}} + (aX_{t_{i-1}} + b)\Delta t + (cX_{t_{i-1}} + d)\Delta B_i$$

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.8+
* NumPy
* Matplotlib

### Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/sde-autonomous-simulator.git](https://github.com/your-username/sde-autonomous-simulator.git)
   cd sde-autonomous-simulator
