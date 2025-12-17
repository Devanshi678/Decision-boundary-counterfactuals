import autograd.numpy as np  # Use autograd's numpy
import torch 
from utils import load_cub_net_components

class ObjectiveFunction:
    """
    Objective function for optimization IN FEATURE SPACE.
    
    build_F1: Computes probability margin from features
    build_F2: Computes L1 norm in feature space
    
    We optimize in feature space, then use invert_features_to_image() to get pixels.
    """
    def __init__(self, query, weights=None, norm2=False, query_pixels=None, *, temp: float = 1.0, smooth_alpha: float = 0.0):
        self.weights= weights
        self.norm2 = norm2  # Whether to include L2 norm in F2
        self.feature_extractor, self.classifier = load_cub_net_components()
        self.query = query  # Query features
        self.query_pixels = query_pixels  # Query pixels (not used in optimization, only for inversion)
        self.target_class = 11  # Target class for counterfactual
        self.query_class = 72   # Original class (Blue Jay)
        # Softmax temperature (T > 1 softens distribution -> stronger gradients when saturated)
        self.temp = float(max(1e-6, temp))
        # Smooth-max sharpness for approximating max over other probabilities (0 -> hard max)
        self.smooth_alpha = float(max(0.0, smooth_alpha))
    

    def prob_margin_numpy(self, feature):
        """
        Numpy-compatible version of prob_margin for autograd optimization.
        Approximates classifier linear layer: p = softmax(W @ feature + b)
        Returns the margin between query_class probability and target_class probability.
        """
        # Get classifier weights and bias (assuming linear layer)
        with torch.no_grad():
            # Extract weights and bias from the classifier's final layer
            if hasattr(self.classifier, 'fc'):
                W = self.classifier.fc.weight.detach().cpu().numpy()  # [num_classes, feature_dim]
                b = self.classifier.fc.bias.detach().cpu().numpy()    # [num_classes]
            else:
                # If classifier has different structure, adapt accordingly
                W = self.classifier.weight.detach().cpu().numpy()
                b = self.classifier.bias.detach().cpu().numpy()
        
        # Compute logits: logits = W @ feature + b
        logits = np.dot(W, feature) + b
        
        # Compute softmax probabilities with temperature: p = softmax((logits)/T)
        z = (logits - np.max(logits)) / self.temp
        exp_logits = np.exp(z)
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Get query class probability and target class probability
        query_class_prob = probabilities[self.query_class]    # p[72] - Blue Jay
        target_class_prob = probabilities[self.target_class]  # p[11] - Target class
        
        # Return margin: p[query_class] - p[target_class]
        # When this is 0, we're at the decision boundary
        # When this is negative, target_class has higher probability (success!)
        return query_class_prob - target_class_prob


    def build_F2(self, feature):
        """
        Build the F2 term of the objective function in FEATURE SPACE.
        Computes L1 norm of features.
        
        Args:
            feature (numpy.ndarray): Current feature values
            
        Returns:
            float: L1 norm of features
        """
        # Compute L1 norm of features (difference between current and query features)
        l1_features = np.sum(np.abs(feature - self.query))
        
        if self.norm2:
            # Add L2 norm component for features
            l2_features = np.sum((feature - self.query) ** 2)
            return l1_features + l2_features
        
        # Return L1 norm for features only
        return l1_features


    def build_F1(self, feature):
        """
        Get constraint violation for penalty method: F1(x)
        Constraint: F1(x) = 0 (at decision boundary)
        
        Args:
            feature (numpy.ndarray): Current feature values
            
        Returns:
            float: Constraint value (0 at boundary, negative when target_class dominates)
        """
        # Compute the probability margin using the features
        f1_value = self.prob_margin_numpy(feature)
        return f1_value