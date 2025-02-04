import argparse
from art.attacks.evasion import CarliniL0Method, CarliniL2Method, CarliniLInfMethod
from art.estimators.classification import PyTorchClassifier
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os


# Helper function to determine the device (GPU/CPU) for computation
def get_device():
    """Returns torch device object - uses CUDA if available, else CPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_preprocessing_transforms(input_shape, model_name=None, normalize=True, custom_mean=None, custom_std=None):
    """
    Get appropriate preprocessing transforms for the model
    Args:
        input_shape: Tuple of (channels, height, width)
        model_name: Name of the model (to determine normalization)
        normalize: Whether to apply normalization
        custom_mean: Optional custom mean values for normalization
        custom_std: Optional custom standard deviation values for normalization
    Returns:
        torchvision.transforms.Compose: Preprocessing pipeline
    """
    transform_list = [
        transforms.Resize((input_shape[1], input_shape[2])),
        transforms.ToTensor(),  # Scales pixels to 0-1 range
    ]
    
    if normalize:
        # Dictionary of known model normalizations
        normalize_params = {
            # ImageNet trained models
            'default_imagenet': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'mnist_example': {
                'mean': [0.1307],
                'std': [0.3081]
            },
        }
        
        # Use custom parameters if provided
        if custom_mean is not None and custom_std is not None:
            params = {
                'mean': custom_mean,
                'std': custom_std
            }
        else:
            # Choose normalization based on model
            if model_name and model_name.startswith('mnist'):
                params = normalize_params['mnist_example']
            else:
                # Default to ImageNet normalization for standard pretrained models
                params = normalize_params['default_imagenet']
            
        transform_list.append(
            transforms.Normalize(
                mean=params['mean'],
                std=params['std']
            )
        )
    
    return transforms.Compose(transform_list)


def preprocess_image(image_path, input_shape, model_name=None, normalize=True, custom_mean=None, custom_std=None):
    """
    Load and preprocess image for model input
    Args:
        image_path: Path to the input image
        input_shape: Tuple of (channels, height, width)
        model_name: Name of the model (to determine normalization)
        normalize: Whether to apply normalization
        custom_mean: Optional custom mean values for normalization
        custom_std: Optional custom standard deviation values for normalization
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Load image
    image = Image.open(image_path)
    
    # Get preprocessing pipeline
    preprocess = get_preprocessing_transforms(
        input_shape, 
        model_name, 
        normalize, 
        custom_mean, 
        custom_std
    )
    
    # Preprocess image
    input_tensor = preprocess(image)
    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)
    
    return input_tensor


def parse_float_list(s):
    """
    Parse a string of comma-separated floats into a list
    Args:
        s: String of comma-separated floats
    Returns:
        list: List of float values
    """
    try:
        if s is None:
            return None
        return [float(x.strip()) for x in s.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError('Values must be comma-separated floats')


def load_model(model_name):
    """
    Loads a pre-trained model from PyTorch Hub
    Args:
        model_name (str): Name of the model to load (e.g., 'mobilenet_v2')
    Returns:
        torch.nn.Module: Loaded and evaluated model on the appropriate device
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, weights='DEFAULT')
    model.eval()  # Set model to evaluation mode
    return model.to(get_device())


def create_classifier(model, input_shape, nb_classes):
    """
    Creates an ART classifier wrapper around PyTorch model
    Args:
        model: PyTorch model
        input_shape: Shape of input images (channels, height, width)
        nb_classes: Number of output classes
    Returns:
        PyTorchClassifier: ART classifier wrapper for the model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=nb_classes,
        device_type=get_device().type,
    )


def initialize_attack(classifier, attack_type, targeted, learning_rate=0.01, target_class=None):
    """
    Initializes the appropriate Carlini-Wagner attack
    Args:
        classifier: ART classifier wrapper
        attack_type: Type of attack ('L0', 'L2', or 'LInf')
        targeted: Boolean indicating if attack is targeted
        learning_rate: Float value for attack algorithm's learning rate (default: 0.01)
                      Lower values give better results but converge slower
        target_class: Target class ID for targeted attacks
    Returns:
        tuple: (initialized attack object, target labels if targeted)
    """
    attack_classes = {
        "L0": CarliniL0Method,
        "L2": CarliniL2Method,
        "LInf": CarliniLInfMethod
    }
    if attack_type not in attack_classes:
        raise ValueError(f"Invalid attack type: {attack_type}. Choose from 'L0', 'L2', or 'LInf'.")
    
    attack_class = attack_classes[attack_type]
    
    y_target = None
    if targeted:
        if target_class is None:
            raise ValueError("Target class ID must be provided for a targeted attack.")
        y_target = np.array([target_class])
    
    return attack_class(
        classifier=classifier,
        targeted=targeted,
        learning_rate=learning_rate,
    ), y_target


def get_prediction(classifier, x):
    """
    Gets model prediction for input
    Args:
        classifier: ART classifier wrapper
        x: Input tensor or numpy array
    Returns:
        np.ndarray: Predicted class IDs
    """
    if isinstance(x, torch.Tensor):  
        x = x.cpu().detach().numpy()
    preds = classifier.predict(x)
    return np.argmax(preds, axis=1)


def generate_adversarial_example(attack, x, y=None):
    """
    Generates adversarial example using specified attack
    Args:
        attack: Initialized attack object
        x: Input tensor
        y: Target labels for targeted attack
    Returns:
        np.ndarray: Generated adversarial example
    """
    x_numpy = x.cpu().detach().numpy()
    return attack.generate(x=x_numpy, y=y)


def save_adversarial_image(x_adversarial, filename="adversarial_example.png"):
    """
    Saves adversarial image to file
    Args:
        x_adversarial: Generated adversarial example
        filename: Output filename
    """
    x_adv_image = (x_adversarial[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    image = Image.fromarray(x_adv_image)
    image.save(filename)


def compute_perturbation(x, x_adversarial):
    """
    Computes L2 norm of perturbation
    Args:
        x: Original input tensor
        x_adversarial: Generated adversarial example
    Returns:
        float: L2 norm of perturbation
    """
    perturbation = np.abs(x_adversarial - x.cpu().detach().numpy())
    return np.linalg.norm(perturbation)


def plot_images(x, x_adversarial):
    """
    Plots original image, adversarial image, and their difference
    Args:
        x: Original input tensor
        x_adversarial: Generated adversarial example
    """
    x_orig_img = x.cpu().detach().numpy()[0].transpose(1, 2, 0)
    x_adv_img = x_adversarial[0].transpose(1, 2, 0)
    perturbation_img = np.abs(x_orig_img - x_adv_img)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(x_orig_img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(x_adv_img)
    ax[1].set_title("Adversarial Image")
    ax[1].axis("off")

    ax[2].imshow(perturbation_img)
    ax[2].set_title("Perturbation (Difference)")
    ax[2].axis("off")

    plt.show()


def main():
    """
    Main function to run adversarial attack pipeline
    Handles argument parsing and orchestrates the attack process
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--model', type=str, required=True, 
                       help='Model to load from Torch Hub')
    parser.add_argument('--input_shape', type=tuple, default=(3, 224, 224), 
                       help='Input shape of the model')
    parser.add_argument('--nb_classes', type=int, default=1000, 
                       help='Number of classes in the model')
    parser.add_argument('--attack_type', type=str, choices=['L0', 'L2', 'LInf'], 
                       default='L0', help='Type of attack (L0, L2, LInf)')
    parser.add_argument('--targeted', action='store_true', 
                       help='Perform a targeted attack')
    parser.add_argument('--target_class', type=int, 
                       help='Target class ID for targeted attack')
    parser.add_argument('--learning_rate', type=float, default=0.01, 
                       help='Learning rate for the attack algorithm (default: 0.01)')
    parser.add_argument('--show_images', action='store_true', 
                       help='Display original and adversarial images')
    parser.add_argument('--compute_metrics', action='store_true', 
                       help='Compute and display attack metrics')
    parser.add_argument('--output_path', type=str, default='adversarial_example.png',
                       help='Path to save the adversarial image')
    parser.add_argument('--normalize', type=bool, default=True,
                       help='Whether to apply normalization during preprocessing')
    parser.add_argument('--normalize_mean', type=parse_float_list, default=None,
                       help='Custom mean values for normalization (comma-separated floats)')
    parser.add_argument('--normalize_std', type=parse_float_list, default=None,
                       help='Custom std values for normalization (comma-separated floats)')
    args = parser.parse_args()

    # Initialize device and model
    device = get_device()
    model = load_model(args.model)
    classifier = create_classifier(model, args.input_shape, args.nb_classes)
    
     # Validate normalization parameters if provided
    if (args.normalize_mean is not None) != (args.normalize_std is not None):
        parser.error("Both --normalize_mean and --normalize_std must be provided together")
    
    if args.normalize_mean is not None and args.normalize_std is not None:
        if len(args.normalize_mean) != args.input_shape[0] or len(args.normalize_std) != args.input_shape[0]:
            parser.error(f"Number of normalization values must match number of channels ({args.input_shape[0]})")

    try:
        # Load and preprocess input image
        print("Loading and preprocessing input image...")
        x = preprocess_image(
            args.image_path, 
            args.input_shape,
            model_name=args.model,
            normalize=args.normalize,
            custom_mean=args.normalize_mean,
            custom_std=args.normalize_std
        )
        x = x.to(device)
        
        # Initialize attack
        print("Initializing attack...")
        attack, y_target = initialize_attack(
            classifier, 
            args.attack_type, 
            args.targeted, 
            learning_rate=args.learning_rate,
            target_class=args.target_class
        )

        # Generate and save adversarial example
        print("Generating adversarial example...")
        x_adversarial = generate_adversarial_example(attack, x, y_target)
        save_adversarial_image(x_adversarial, args.output_path)
        print(f"Adversarial example saved to {args.output_path}")

        # Get and display predictions
        original_class = get_prediction(classifier, x)
        adversarial_class = get_prediction(classifier, x_adversarial)

        print(f"\nResults:")
        print(f"Original class ID: {original_class[0]}")
        print(f"Adversarial class ID: {adversarial_class[0]}")

        if original_class[0] != adversarial_class[0]:
            print(f"Attack succeeded! The model misclassified the input as class {adversarial_class[0]}.")
        else:
            print("Attack failed. The model still classifies the input correctly.")

        if args.compute_metrics:
            perturbation_norm = compute_perturbation(x, x_adversarial)
            print(f"Total perturbation (L2 norm): {perturbation_norm}")

        if args.show_images:
            plot_images(x, x_adversarial)
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return

if __name__ == "__main__":
    main()