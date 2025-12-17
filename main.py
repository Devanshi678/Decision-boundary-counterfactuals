import autograd.numpy as np
import torch
from autograd import grad
from model import ObjectiveFunction
from utils import load_and_transform_image, get_image_transform, load_cub_net_components, invert_features_to_image, plot_image
from solver import augmented_lagrangian_solver, plot_optimization_history


# Define current objective function
def objective(x):
    # x now represents feature values
    return model.build_F2(x) 

def constraint(x):
    return model.build_F1(x) 


def init():
    # Load models and data
    feature_extractor, classifier = load_cub_net_components()
    print("Models loaded successfully")
    
    # Load and transform the image to get pixel data
    transform = get_image_transform()
    image_tensor = load_and_transform_image("./img/Blue_Jay_0083_61492.jpg", transform=transform)
    query_pixels = np.array(image_tensor.detach().cpu())
    
    print("Image loaded and transformed successfully")
    
    # Extract features from the image - THIS is what we'll optimize
    feature = feature_extractor(image_tensor.unsqueeze(0))
    query_feature = np.array(feature.squeeze(0).detach().cpu())

    # Objective with softened probabilities and smooth max
    model = ObjectiveFunction(query=query_feature, query_pixels=query_pixels, weights={'f2': 1}, 
                             norm2=False, temp=1.8, smooth_alpha=15.0)

    return model, query_feature, feature_extractor, classifier, query_pixels


if __name__ == "__main__":
    model, x, feature_extractor, classifier, query_pixels = init()
    
    # Calculate the initial objective value and constraint value
    constr_val = constraint(x)
    obj_val = objective(x)
    print(f"\n{'='*80}")
    print(f"INITIAL VALUES")
    print(f"{'='*80}")
    print(f"Initial F2 (objective): {obj_val:.2f}")
    print(f"Initial F1 (constraint): {constr_val:.6f}")
    print(f"Feature vector shape: {x.shape}")
    print(f"{'='*80}\n")
    
    # Verify initial classification
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        probs = classifier(x_tensor)
        initial_class = probs.argmax(dim=1).item()
        print(f"Initial classification: Class {initial_class} (Blue Jay)")
        print(f"Initial probability for class {initial_class}: {probs[0, initial_class].item():.4f}\n")
    
    # Run Augmented Lagrangian optimization IN FEATURE SPACE
    print("Starting Augmented Lagrangian optimization IN FEATURE SPACE...\n")
    
    optimized_features, info = augmented_lagrangian_solver(
        objective_func=objective,
        constraint_func=constraint,
        x0=x,
        rho_init=1.0,
        rho_max=1e7,
        rho_increase=2.0,
        lambda_init=0.0,
        inner_lr=0.1,
        inner_max_iter=200, 
        outer_max_iter=400,
        tol_constraint=1e-4,
        tol_obj=1e-8,
        log_interval=20,
        verbose=True
    )
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Runtime: {info['runtime']:.2f} seconds")
    print(f"Total iterations: {info['total_iterations']}")
    print(f"Initial F1: {info['history']['F1'][0]:.6f}")
    print(f"Final F1: {info['final_F1']:.6f} (target: 0)")
    print(f"Initial F2: {info['history']['F2'][0]:.2f}")
    print(f"Final F2: {info['final_F2']:.2f}")
    print(f"Converged: {info['converged']}")
    print(f"{'='*80}\n")
    
    # Verify final classification on optimized features
    print("Verifying final classification on optimized features...")
    with torch.no_grad():
        opt_tensor = torch.tensor(optimized_features, dtype=torch.float32).unsqueeze(0)
        probs_opt = classifier(opt_tensor)
        initial_class_prob = probs_opt[0, initial_class].item()
        final_class = probs_opt.argmax(dim=1).item()
        final_class_prob = probs_opt[0, final_class].item()
        
        print(f"Original classification: Class {initial_class} (Blue Jay)")
        print(f"Final classification: Class {final_class}")
        print(f"Final probability for original class {initial_class}: {initial_class_prob:.4f}")
        print(f"Final probability for counterfactual class {final_class}: {final_class_prob:.4f}")
        
        if final_class != initial_class:
            print(f"✓ SUCCESS: Image successfully classified to a different class (Class {final_class})!")
            print(f"  The counterfactual image is at the decision boundary (F1 ≈ 0)")
        else:
            print(f"✗ Image still classified as original class {initial_class}")
    
    # Plot optimization history
    print("\nGenerating optimization history plot...")
    plot_optimization_history(info, save_path='optimization_history.png')
    
    # Convert optimized FEATURES back to displayable image
    print("\nInverting optimized features to final counterfactual image...")
    print("(This process converts the optimized features back to pixel space)")
    
    # Convert optimized features to torch tensor for inversion
    optimized_features_tensor = torch.tensor(optimized_features, dtype=torch.float32).unsqueeze(0)
    
    # Invert to clean image
    output_image = invert_features_to_image(
        feature_extractor=feature_extractor,
        classifier=classifier,
        target_feat=optimized_features_tensor,
        original_image_path="./img/Blue_Jay_0083_61492.jpg",
        transform=get_image_transform(),
        device='cpu',
        inv_steps=600,
        inv_lr=0.003,
        inv_recon_weight=1.0,
        inv_tv_weight=1e-3,
        inv_cls_weight=1.0,
        save_path="counterfactual_image.jpg"
    )
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print("Generated files:")
    print("  1. counterfactual_image.jpg - Final counterfactual image")
    print("  2. optimization_history.png - F1 and F2 convergence plot")
    print("="*80)