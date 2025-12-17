import autograd.numpy as np
from autograd import grad
import time
import matplotlib.pyplot as plt


def augmented_lagrangian_solver(
    objective_func,
    constraint_func,
    x0,
    rho_init=0.1,
    rho_max=1e6,
    rho_increase=1.5,
    lambda_init=0.0,
    inner_lr=0.005,
    inner_max_iter=100,
    outer_max_iter=20,
    tol_constraint=1e-4,
    tol_obj=1e-8,
    log_interval=5,
    verbose=True
):
    """
    Augmented Lagrangian Method for solving:
        min F2(x)
        subject to F1(x) = 0
    
    The Augmented Lagrangian is:
        L(x, λ, ρ) = F2(x) + λ*F1(x) + (ρ/2)*F1(x)²
    
    Args:
        objective_func: Function F2(x) to minimize
        constraint_func: Function F1(x) that should equal 0
        x0: Initial point (feature values)
        rho_init: Initial penalty parameter
        rho_max: Maximum penalty parameter
        rho_increase: Factor to increase rho each outer iteration
        lambda_init: Initial Lagrange multiplier
        inner_lr: Learning rate for inner optimization (gradient descent)
        inner_max_iter: Max iterations for inner loop
        outer_max_iter: Max iterations for outer loop
        tol_constraint: Tolerance for constraint satisfaction |F1(x)| < tol
        tol_obj: Tolerance for objective change
        log_interval: Print/log every N iterations
        verbose: Whether to print progress
    
    Returns:
        x_opt: Optimized feature values
        info: Dictionary with optimization history and statistics
    """
    
    # Initialize
    x = x0.copy()
    lambda_val = lambda_init
    rho = rho_init
    
    # History tracking
    history = {
        'F1': [],
        'F2': [],
        'lambda': [],
        'rho': [],
        'augmented_lagrangian': [],
        'outer_iter': [],
        'inner_iter': [],
        'total_iterations': 0
    }
    
    # Define Augmented Lagrangian
    def augmented_lagrangian(x_var, lambda_var, rho_var):
        """L(x, λ, ρ) = F2(x) + λ*F1(x) + (ρ/2)*F1(x)²"""
        f1_val = constraint_func(x_var)
        f2_val = objective_func(x_var)
        return f2_val + lambda_var * f1_val + (rho_var / 2.0) * f1_val ** 2
    
    # Compute gradient of augmented Lagrangian w.r.t. x
    grad_L = grad(lambda x_var: augmented_lagrangian(x_var, lambda_val, rho))
    
    start_time = time.time()
    
    if verbose:
        print("\n" + "="*80)
        print("AUGMENTED LAGRANGIAN OPTIMIZATION")
        print("="*80)
        print(f"Initial F1(x): {constraint_func(x):.6f}")
        print(f"Initial F2(x): {objective_func(x):.6f}")
        print(f"Initial λ: {lambda_val:.6f}, ρ: {rho:.6f}")
        print("="*80 + "\n")
        
        # Print compact header
        print(f"{'Outer':<7} | {'λ':<12} | {'ρ':<12} | {'F1':<10} | {'F2':<10}")
        print("-" * 60)
    
    # Outer loop: Update Lagrange multiplier and penalty parameter
    for outer_iter in range(outer_max_iter):
        # Update gradient function with current lambda and rho
        grad_L = grad(lambda x_var: augmented_lagrangian(x_var, lambda_val, rho))
        
        # Inner loop: Minimize augmented Lagrangian w.r.t. x
        prev_obj = float('inf')
        for inner_iter in range(inner_max_iter):
            # Compute gradient
            g = grad_L(x)
            
            # Gradient descent step
            x = x - inner_lr * g
            
            # Compute current values
            f1_val = constraint_func(x)
            f2_val = objective_func(x)
            L_val = augmented_lagrangian(x, lambda_val, rho)
            
            # Store history
            history['F1'].append(f1_val)
            history['F2'].append(f2_val)
            history['lambda'].append(lambda_val)
            history['rho'].append(rho)
            history['augmented_lagrangian'].append(L_val)
            history['outer_iter'].append(outer_iter)
            history['inner_iter'].append(inner_iter)
            history['total_iterations'] += 1
            
            # Check convergence of inner loop
            if abs(L_val - prev_obj) < tol_obj and inner_iter > 50:
                break
            
            prev_obj = L_val
        
        # Evaluate constraint after inner optimization
        f1_final = constraint_func(x)
        f2_final = objective_func(x)
        
        # Print compact one-line summary for this outer iteration
        if verbose:
            print(f"{outer_iter+1:<7} | {lambda_val:<12.2f} | {rho:<12.2e} | {f1_final:<10.6f} | {f2_final:<10.2f}")
        
        # Check if constraint is satisfied using absolute value
        if abs(f1_final) < tol_constraint:
            if verbose:
                print("\n" + "="*80)
                print(f"✓ CONVERGED: |F1(x)| = {abs(f1_final):.6f} < {tol_constraint}")
                print("="*80)
            break
        
        # Update Lagrange multiplier: λ ← λ + ρ*F1(x)
        lambda_val = lambda_val + rho * f1_final
        
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
        print(f"Final λ: {lambda_val:.6f}")
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
        'final_lambda': lambda_val,
        'final_rho': rho,
        'converged': abs(f1_final) < tol_constraint
    }
    
    return x, info


def plot_optimization_history(info, save_path='optimization_history.png'):
    """
    Plot F1 and F2 values over iterations.
    
    Args:
        info: Dictionary returned by augmented_lagrangian_solver
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
    ax1.set_title('Constraint F1(x) vs Iteration', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot F2 (objective)
    ax2.plot(iterations, history['F2'], 'g-', linewidth=2, label='F2(x) - Objective')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('F2(x)', fontsize=12)
    ax2.set_title('Objective F2(x) vs Iteration', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Optimization history plot saved to: {save_path}")
    plt.close()