# Adversarial Attack Tool using Carlini-Wagner Methods

This repository provides a tool for performing adversarial attacks on deep learning models using the **Carlini-Wagner** attack methods. The tool is designed to work with pre-trained models from PyTorch and offers support for three types of Carlini-Wagner attacks:

- **L0 Norm Attack**
- **L2 Norm Attack**
- **L∞ Norm Attack**

The tool uses the **Adversarial Robustness Toolbox (ART)** to create adversarial examples and supports both **untargeted** and **targeted** attacks.

## Requirements

- Python 3.6 or higher
- PyTorch
- ART (Adversarial Robustness Toolbox)
- TorchVision
- Matplotlib
- NumPy
- PIL

You can install the required dependencies using `pip`:

```bash
pip install torch torchvision matplotlib numpy pillow adversarial-robustness-toolbox
```

## Usage

To run the adversarial attack, use the following command structure:

```bash
python3 test.py --image_path PATH_TO_IMAGE --model MODEL_NAME [options]
```

### Required Arguments:
- `--image_path`: Path to the input image.
- `--model`: Name of the pre-trained model to use from TorchHub. (e.g., `mobilenet_v2`, `resnet50`).

### Optional Arguments:
- `--attack_type`: Type of attack to perform. Choose from `'L0'`, `'L2'`, or `'LInf'`. Default is `L0`.
- `--targeted`: If specified, the attack will be targeted to a specific class.
- `--target_class`: The target class ID for a targeted attack.
- `--learning_rate`: Learning rate for the attack algorithm (default is 0.01).
- `--compute_metrics`: If specified, computes the L2 norm of the perturbation.
- `--show_images`: If specified, shows the original and adversarial images.
- `--normalize`: Whether to apply normalization during preprocessing (default is `True`).
- `--normalize_mean`: Custom mean values for normalization (comma-separated floats).
- `--normalize_std`: Custom std values for normalization (comma-separated floats).
- `--output_path`: Path to save the adversarial example image (default is `'adversarial_example.png'`).

## Examples

### 1. **Basic Example:**
Perform a default adversarial attack on a `mobilenet_v2` model with the image:

```bash
python3 test.py --image_path /path/to/your/image.jpg --model mobilenet_v2
```

### 2. **Targeted Attack:**
Perform a **targeted attack** on the `mobilenet_v2` model, targeting class 42:

```bash
python3 test.py --image_path /path/to/your/image.jpg --model mobilenet_v2 --targeted --target_class 42
```

### 3. **Custom Normalization:**
Use custom normalization values for preprocessing the image:

```bash
python3 test.py --image_path /path/to/your/image.jpg --model mobilenet_v2 --normalize_mean 0.5,0.5,0.5 --normalize_std 0.2,0.2,0.2
```

### 4. **L2 Attack Example:**
Perform a **L2 attack** with a custom learning rate of 0.005:

```bash
python3 test.py --image_path /path/to/your/image.jpg --model mobilenet_v2 --attack_type L2 --learning_rate 0.005
```

### 5. **Saving the Adversarial Example:**
Save the adversarial example with a custom filename:

```bash
python3 test.py --image_path /path/to/your/image.jpg --model mobilenet_v2 --output_path adversarial_output.png
```

### 6. **Full Command with All Options:**
Perform a **targeted LInf attack** on `resnet50` with custom normalization, show images, and compute metrics:

```bash
python3 test.py --image_path /path/to/your/image.jpg --model resnet50 --attack_type LInf --targeted --target_class 42 --learning_rate 0.01 --show_images --compute_metrics --normalize_mean 0.485,0.456,0.406 --normalize_std 0.229,0.224,0.225
```

## Results

After running the attack, the following results will be shown:

- **Original class ID**: The predicted class of the original image.
- **Adversarial class ID**: The predicted class of the adversarial image.
- **Attack success**: Whether the attack succeeded in misclassifying the image.
- **Perturbation L2 norm**: The L2 norm of the perturbation applied to the original image (if `--compute_metrics` is enabled).

The adversarial image will be saved at the specified `--output_path` or the default path `adversarial_example.png`.

## License

This project is licensed under the MIT License.