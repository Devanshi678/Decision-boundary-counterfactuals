import autograd.numpy as np
from autograd import grad
import time
import matplotlib.pyplot as plt


class AdamOptimizer:
    """Adam optimizer implementation for numpy arrays"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def step(self, params, grads):
        """
        Update parameters using Adam optimization
        
        Args:
            params: Current parameters (numpy array)
            grads: Gradients (numpy array)
        
        Returns:
            Updated parameters
        """
        # Initialize moments on first call
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        # Increment time step
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        params_new = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params_new


def penalty_method_adam_solver(
    objective_func,
    constraint_func,
    x0,
    rho_init=10.0,
    rho_max=1e8,
    rho_increase=3.0,
    adam_lr=0.2,
    adam_beta1=0.9,
    adam_beta2=0.999,
    inner_max_iter=200,
    outer_max_iter=200,
    tol_constraint=1e-3,
    tol_obj=1e-6,
    log_interval=20,
    verbose=True
):
    """
    Penalty Method with Adam Optimizer for solving:
        min F2(x)
        subject to F1(x) = 0
    
    Uses penalty function:
        P(x, ρ) = F2(x) + (ρ/2)*F1(x)²
    
    Args:
        objective_func: Function F2(x) to minimize
        constraint_func: Function F1(x) that should equal 0
        x0: Initial point (feature values)
        rho_init: Initial penalty parameter
        rho_max: Maximum penalty parameter
        rho_increase: Factor to increase rho each outer iteration
        adam_lr: Adam learning rate
        adam_beta1: Adam beta1 parameter (first moment decay)
        adam_beta2: Adam beta2 parameter (second moment decay)
        inner_max_iter: Max iterations for inner loop
        outer_max_iter: Max iterations for outer loop
        tol_constraint: Tolerance for constraint satisfaction |F1(x)| < tol
        tol_obj: Tolerance for objective change
        log_interval: Print/log every N iterations (for inner loop)
        verbose: Whether to print progress
    
    Returns:
        x_opt: Optimized feature values
        info: Dictionary with optimization history and statistics
    """
    
    # Initialize
    x = x0.copy()
    rho = rho_init
    
    # History tracking
    history = {
        'F1': [],
        'F2': [],
        'rho': [],
        'penalty': [],
        'outer_iter': [],
        'inner_iter': [],
        'total_iterations': 0
    }
    
    # Define Penalty Function
    def penalty_function(x_var, rho_var):
        """P(x, ρ) = F2(x) + (ρ/2)*F1(x)²"""
        f1_val = constraint_func(x_var)
        f2_val = objective_func(x_var)
        return f2_val + (rho_var / 2.0) * f1_val ** 2
    
    start_time = time.time()
    
    if verbose:
        print("\n" + "="*80)
        print("PENALTY METHOD WITH ADAM OPTIMIZER")
        print("="*80)
        print(f"Initial F1(x): {constraint_func(x):.6f}")
        print(f"Initial F2(x): {objective_func(x):.6f}")
        print(f"Initial ρ: {rho:.6f}")
        print(f"Adam hyperparameters: lr={adam_lr}, β1={adam_beta1}, β2={adam_beta2}")
        print("="*80 + "\n")
        
        # Print compact header
        print(f"{'Outer':<7} | {'ρ':<12} | {'F1':<10} | {'F2':<10}")
        print("-" * 48)
    
    # Outer loop: Increase penalty parameter
    for outer_iter in range(outer_max_iter):
        # Create new Adam optimizer for this outer iteration
        optimizer = AdamOptimizer(learning_rate=adam_lr, beta1=adam_beta1, beta2=adam_beta2)
        
        # Compute gradient of penalty function w.r.t. x
        grad_P = grad(lambda x_var: penalty_function(x_var, rho))
        
        # Inner loop: Minimize penalty function w.r.t. x using Adam
        prev_obj = float('inf')
        for inner_iter in range(inner_max_iter):
            # Compute gradient
            g = grad_P(x)
            
            # Adam optimization step
            x = optimizer.step(x, g)
            
            # Compute current values
            f1_val = constraint_func(x)
            f2_val = objective_func(x)
            P_val = penalty_function(x, rho)
            
            # Store history
            history['F1'].append(f1_val)
            history['F2'].append(f2_val)
            history['rho'].append(rho)
            history['penalty'].append(P_val)
            history['outer_iter'].append(outer_iter)
            history['inner_iter'].append(inner_iter)
            history['total_iterations'] += 1
            
            # Check convergence of inner loop
            if abs(P_val - prev_obj) < tol_obj and inner_iter > 50:
                break
            
            prev_obj = P_val
        
        # Evaluate constraint after inner optimization
        f1_final = constraint_func(x)
        f2_final = objective_func(x)
        
        # Print compact one-line summary for this outer iteration
        if verbose:
            print(f"{outer_iter+1:<7} | {rho:<12.2e} | {f1_final:<10.6f} | {f2_final:<10.2f}")
        
        # Check if constraint is satisfied using absolute value
        if abs(f1_final) < tol_constraint:
            if verbose:
                print("\n" + "="*80)
                print(f"✓ CONVERGED: |F1(x)| = {abs(f1_final):.6f} < {tol_constraint}")
                print("="*80)
            break
        
        # Increase penalty parameter: ρ ← ρ * factor
        if rho < rho_max:
            rho = min(rho * rho_increase, rho_max)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Final evaluation
    f1_final = constraint_func(x)
    f2_final = objective_func(x)
    
    if verbose:
        print(f"\n{'='*80}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total runtime: {runtime:.2f} seconds")
        print(f"Total iterations: {history['total_iterations']}")
        print(f"Final F1(x): {f1_final:.6f} (target: 0)")
        print(f"Final F2(x): {f2_final:.2f}")
        print(f"Final ρ: {rho:.6f}")
        
        if abs(f1_final) < tol_constraint:
            print(f"✓ Constraint satisfied: |F1(x)| = {abs(f1_final):.6f} < {tol_constraint}")
        else:
            print(f"✗ Constraint NOT satisfied: |F1(x)| = {abs(f1_final):.6f} >= {tol_constraint}")
        print(f"{'='*80}\n")
    
    # Prepare info dictionary
    info = {
        'history': history,
        'runtime': runtime,
        'total_iterations': history['total_iterations'],
        'final_F1': f1_final,
        'final_F2': f2_final,
        'final_rho': rho,
        'converged': abs(f1_final) < tol_constraint
    }
    
    return x, info


def plot_optimization_history(info, save_path='optimization_history_adam.png'):
    """
    Plot F1 and F2 values over iterations.
    
    Args:
        info: Dictionary returned by penalty_method_adam_solver
        save_path: Path to save the plot
    """
    history = info['history']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    iterations = range(len(history['F1']))
    
    # Plot F1 (constraint)
    ax1.plot(iterations, history['F1'], 'b-', linewidth=2, label='F1(x) - Constraint')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Target (F1=0)')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('F1(x)', fontsize=12)
    ax1.set_title('Constraint F1(x) vs Iteration (Adam Optimizer)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot F2 (objective)
    ax2.plot(iterations, history['F2'], 'g-', linewidth=2, label='F2(x) - Objective')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('F2(x)', fontsize=12)
    ax2.set_title('Objective F2(x) vs Iteration (Adam Optimizer)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Optimization history plot saved to: {save_path}")
    plt.close()