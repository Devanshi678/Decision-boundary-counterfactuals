# Decision-boundary-counterfactuals
Optimization-based counterfactual explanations that explicitly compute CNN decision boundaries in feature space.
### Using Constrained Optimization Methods

## 📌 Project Overview
This project focuses on generating **counterfactual explanations** for deep convolutional neural networks (CNNs) by explicitly computing **decision boundary images** using **constrained optimization**.

Given a query image that is confidently classified by a CNN, the goal is to generate a **minimally modified image** that lies on the **decision boundary** between the original class and a target counterfactual class. This boundary image provides a principled and interpretable explanation of *what must change* for the model’s decision to change.

This work was completed as a course project for **IMSE 505 – Optimization** and will directly extend into my **Master’s thesis**, with a focus on:
- Trustworthy AI
- Explainable AI (XAI)
- Optimization-based reasoning for neural networks

---

## 🎯 Research Objectives
- Formulate counterfactual explanation generation as a **constrained optimization problem**
- Accurately locate CNN **decision boundaries**
- Preserve **perceptual and feature-space similarity** to the original image
- Compare classical optimization methods with modern adaptive optimizers

---

## 🧠 Problem Formulation
Let a CNN be defined as:

\[
O = g(f(I))
\]

where:
- \( f(\cdot) \) is the feature extractor (CNN backbone)
- \( g(\cdot) \) is the classifier head

Given:
- Query image \( I_q \) classified as class \( k \)
- Target counterfactual class \( k' \)

### Optimization Problem
\[
\min_I \; F_2(I) \quad \text{subject to} \quad F_1(I) = 0
\]

#### Constraint (Contour Function)
\[
F_1(I) = \max_{i \ne k'} \left( g_i(f(I)) - g_{k'}(f(I)) \right)
\]

- \( F_1(I) = 0 \) defines the **decision boundary** for class \( k' \)

#### Objective (Regularization Function)
\[
F_2(I) = \|f(I) - f(I_q)\|_1 + \|I - I_q\|_1
\]

- Ensures **minimal perturbation** in both feature and pixel space

---

## ⚙️ Optimization Methods Implemented

### 1️⃣ Augmented Lagrangian Method
- Combines Lagrange multipliers with penalty enforcement
- Strong theoretical convergence guarantees
- Dual enforcement of constraint satisfaction

**Key Properties**
- Extremely precise boundary detection
- Slower runtime
- Ideal for research and safety-critical applications

---

### 2️⃣ Penalty Method with Adam Optimizer
- Uses adaptive learning rates
- Faster convergence
- Practical for large-scale or interactive settings

**Key Properties**
- ~30× faster than Augmented Lagrangian
- Slightly relaxed constraint precision
- Better similarity preservation

---

## 📊 Experimental Setup
- **Dataset**: CUB-200-2011 (Bird Species)
- **Model**: ResNet50-based CNN
- **Query Image**: Blue Jay (Class 72)
- **Target Class**: Class 73
- **Optimization Space**: 2048-dimensional feature space  
  *(instead of 150,528-dimensional pixel space)*

---

## 📈 Results Summary

| Metric | Augmented Lagrangian | Adam Penalty |
|------|----------------------|--------------|
| Runtime | 140.22 s | **4.68 s** |
| Total Iterations | 27,600 | **3,000** |
| Final Constraint \( F_1 \) | **0.000099** | 0.000795 |
| Final Objective \( F_2 \) | 549.19 | **457.66** |
| Class Change | 72 → 73 | 72 → 73 |

✔ Both methods successfully generate valid counterfactuals  
✔ Adam achieves **dramatic speedup**  
✔ Augmented Lagrangian achieves **maximum precision**

---

## 🔍 Key Insights
- **Decision boundaries can be explicitly computed**, not approximated
- Optimization in **feature space** is essential for tractability
- Constraint enforcement strategy significantly affects stability
- Adaptive optimizers (Adam) are highly effective even in constrained settings
- Counterfactual explanations provide **model-level interpretability**, not heuristics

---

## 🚀 How to Run

### This project was developed and tested using Anaconda with a predefined Conda environment to ensure reproducibility and dependency consistency.
1️⃣ Install Anaconda
If Anaconda is not already installed, download and install it from:
https://www.anaconda.com/products/distribution
Make sure conda is available in your terminal

2️⃣ Create the Conda Environment
Navigate to the project root directory (where environment.yml is located) and run:
conda env create -f environment.yml
This will create a Conda environment with all required dependencies (PyTorch, NumPy, etc.).

3️⃣ Activate the Environment
conda activate <environment_name>
Replace <environment_name> with the name specified at the top of environment.yml.

4️⃣ Run the Optimization Methods
Augmented Lagrangian Method
python main.py

Penalty Method with Adam Optimizer
python main_adam.py

5️⃣ Expected Output

Console logs showing:
- Constraint value 𝐹1(𝐼) decreasing toward zero
- Objective value 𝐹2(𝐼)

Convergence status and runtime
Generated counterfactual images saved to the results/ directory
Plots illustrating optimization convergence

## 🧪 Notes on Reproducibility
-The project was tested on a local machine using Anaconda
-Optimization is performed in feature space (2048 dimensions)
-Runtime may vary depending on hardware
-GPU is not required, but optional acceleration may improve speed

## 📁 Repository Structure
Optimization-Counterfactuals/
├── main.py              # Augmented Lagrangian method
├── main_adam.py         # Adam-based penalty method
├── models/              # CNN backbone
├── utils/               # Optimization utilities
├── results/             # Generated counterfactuals & plots
├── IMSE505_Report.pdf   # Full technical report

## 📚 What I Learned
-How to translate theoretical optimization concepts into working ML systems
-How CNN decision boundaries behave in high-dimensional feature spaces
-Trade-offs between theoretical rigor and computational efficiency
-Why optimization-based explainability is more reliable than post-hoc heuristics
-How optimization connects AI trust, interpretability, and robustness

## 📌 Acknowledgment
This project was completed as part of IMSE 505 – Optimization under Prof. Jian Hu. (University of Michigan- Dearborn)
