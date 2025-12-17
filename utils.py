from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from cub_net import CUBNet
import matplotlib.pyplot as plt 


def plot_image(image_tensor, title="Loaded Image"):
    """
    Plot an image tensor after converting it back to displayable format.
    
    Args:
        image_tensor (torch.Tensor or numpy.ndarray): Image tensor to plot
        title (str): Title for the plot
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(image_tensor, torch.Tensor):
        img_np = image_tensor.detach().cpu().numpy()
    else:
        img_np = image_tensor
    
    # Denormalize the image (reverse ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Denormalize: img = img * std + mean
    img_denorm = img_np * std[:, None, None] + mean[:, None, None]
    
    # Clip values to [0, 1] range
    img_denorm = np.clip(img_denorm, 0, 1)
    
    # Convert from CHW to HWC format for plotting
    img_plot = np.transpose(img_denorm, (1, 2, 0))
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.imshow(img_plot)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    
def get_image_transform():
    """
    Create and return the standard image transformation pipeline.
    
    Returns:
        transforms.Compose: A composition of image transforms including resize,
                          center crop, tensor conversion, and normalization.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_and_transform_image(image_path, transform=None):
    """
    Load an image from path and apply transformations.
    
    Args:
        image_path (str): Path to the image file
        transform (transforms.Compose, optional): Transform to apply. 
                                                If None, uses default transform.
    
    Returns:
        torch.Tensor: Transformed image tensor
    """
    if transform is None:
        transform = get_image_transform()
    
    image = Image.open(image_path).convert('RGB')
    return transform(image)


class FeatureExtractor(nn.Module):
    """Feature extraction network (CUB-Net without the final classifier)"""
    
    def __init__(self, cub_net):
        super(FeatureExtractor, self).__init__()
        self.conv1 = cub_net.conv1
        self.bn1 = cub_net.bn1
        self.relu = cub_net.relu
        self.maxpool = cub_net.maxpool
        self.layer1 = cub_net.layer1
        self.layer2 = cub_net.layer2
        self.layer3 = cub_net.layer3
        self.layer4 = cub_net.layer4
        self.avgpool = cub_net.avgpool
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        return feat


class Classifier(nn.Module):
    """Classifier network (just the final linear layer)"""
    
    def __init__(self, cub_net):
        super(Classifier, self).__init__()
        self.fc = cub_net.fc
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, features):
        logits = self.fc(features)
        return self.softmax(logits)


def load_cub_net_components(checkpoint_path='./weights/best_resnet50_cub200.pth', num_classes=200, device='cpu'):
    """
    Load CUB-Net from checkpoint and return separate feature extraction and classifier networks.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        num_classes (int): Number of classes for the classifier
        device (str): Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        tuple: (feature_extractor, classifier) - separate networks for feature extraction and classification
    """
    # Create the full CUB-Net model
    cub_net = CUBNet(num_classes=num_classes)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load the state dict into the model
    cub_net.load_state_dict(state_dict)
    
    # Set to evaluation mode
    cub_net.eval()
    
    # Create separate feature extractor and classifier
    feature_extractor = FeatureExtractor(cub_net)
    classifier = Classifier(cub_net)
    
    # Move to specified device
    feature_extractor.to(device)
    classifier.to(device)
    
    # Set to evaluation mode
    feature_extractor.eval()
    classifier.eval()
    
    return feature_extractor, classifier


def invert_features_to_image(feature_extractor, classifier, target_feat, original_image_path, 
                            transform, device='cpu', inv_steps=600, inv_lr=0.003, 
                            inv_recon_weight=1.0, inv_tv_weight=1e-3, inv_cls_weight=1.0,
                            save_path="inverted_image.jpg"):
    """
    Invert features back to an image using optimization.
    
    Args:
        feature_extractor: The feature extraction network
        classifier: The classification network
        target_feat: Target features to reconstruct (torch.Tensor)
        original_image_path: Path to the original image to start optimization from
        transform: Image transformation pipeline
        device: Device to run on ('cpu' or 'cuda')
        inv_steps: Number of optimization steps
        inv_lr: Learning rate for image inversion
        inv_recon_weight: Weight for feature reconstruction loss
        inv_tv_weight: Weight for total variation loss
        inv_cls_weight: Weight for classification loss
        save_path: Path to save the inverted image
    
    Returns:
        PIL.Image: The inverted image
    """
    # Move models and tensors to device
    feature_extractor = feature_extractor.to(device)
    classifier = classifier.to(device)
    feature_extractor.eval()
    classifier.eval()
    target_feat = target_feat.to(device)

    # Determine the desired class: the class of the optimized features
    with torch.no_grad():
        target_class_id = classifier(target_feat).argmax(dim=1).item()
    print(f"Inversion target class (from optimized feature): {target_class_id}")

    # Load and prepare the initial image
    img = load_and_transform_image(original_image_path, transform=transform).unsqueeze(0).to(device)
    img.requires_grad = True

    # Optimizer and Loss functions
    optim_img = optim.Adam([img], lr=inv_lr)
    mse = nn.MSELoss()

    def tv_loss(x):
        """Total variation loss for smoothness"""
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]
        return (dx.abs().mean() + dy.abs().mean())

    # Iterative optimization
    for step in range(inv_steps):
        optim_img.zero_grad()

        clamped_img = img.clamp(0, 1)

        # Forward pass through the networks
        feat = feature_extractor(clamped_img)
        feat = feat.view(feat.size(0), -1)
        logits = classifier(feat)

        # Compute losses
        loss_recon = mse(feat, target_feat) * inv_recon_weight
        loss_tv = tv_loss(clamped_img) * inv_tv_weight
        loss_cls = F.cross_entropy(logits, torch.tensor([target_class_id], device=device)) * inv_cls_weight
        loss = loss_recon + loss_tv + loss_cls

        # Backward pass and optimization step
        loss.backward()
        optim_img.step()

        # Print progress
        if step % 100 == 0 or step == inv_steps - 1:
            with torch.no_grad():
                pred_class = logits.argmax(dim=1).item()
            print(f"Step {step}: total={loss.item():.4f}, recon={loss_recon.item():.4f}, "
                  f"tv={loss_tv.item():.4f}, cls={loss_cls.item():.4f}, pred_class={pred_class}")

    # Save the results
    out = img.detach().clamp(0, 1).cpu().squeeze()
    to_pil = transforms.ToPILImage()
    out_img = to_pil(out)
    out_img.save(save_path)
    print(f"Saved inverted image to {save_path}")
    
    return out_img

