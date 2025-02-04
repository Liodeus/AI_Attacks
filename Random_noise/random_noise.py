import numpy as np
import argparse
from PIL import Image
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate random noise images.')
parser.add_argument('--count', type=int, default=1, help='Number of images to generate')
parser.add_argument('--width', type=int, default=224, help='Width of the image')
parser.add_argument('--height', type=int, default=224, help='Height of the image')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save images (mandatory)')
parser.add_argument('--format', type=str, choices=['png', 'jpg', 'jpeg'], default='png', help='Image format (png, jpg, jpeg)')
parser.add_argument('--quality', type=int, default=95, help='Quality for JPEG format (1-100, default: 95)')

args = parser.parse_args()

# Show help if no arguments are provided
if len(vars(args)) == 0:
    parser.print_help()
    exit()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

for i in range(args.count):
    # Generate random noise image as a NumPy array with shape (height, width, 3)
    noise = np.random.randint(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise) # Convert the NumPy array to a PIL Image object
    
    image_path = os.path.join(args.output_dir, f'{args.width}_{args.height}_{i}.{args.format}')
    
    if args.format in ['jpg', 'jpeg']:
        noise_img = noise_img.convert('RGB')  # Ensure RGB mode for JPEG
        noise_img.save(image_path, format='JPEG', quality=args.quality)
    else:
        noise_img.save(image_path, format=args.format.upper())
    
    print(f'Saved {i + 1} images')

print("Noise image generation complete!")