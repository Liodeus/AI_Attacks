# Random Noise Image Generator

This project provides a Python script to generate random noise images of a specified size, format, and quantity. It allows you to create images for testing or experimentation in various formats such as PNG, JPEG, and JPG.

## Features

- Generate random noise images in different sizes (height and width).
- Save images in `PNG`, `JPG`, or `JPEG` formats.
- Control the quality for `JPG`/`JPEG` format images.
- Automatically create an output directory to save the generated images.

## Requirements

- Python 3.x
- Pillow (for image processing)
- NumPy (for array operations)

You can install the necessary Python libraries using the following:

```bash
pip install pillow numpy
```

## Usage

To run the script, use the following command structure:

```bash
python generate_noise_images.py --count NUM_IMAGES --width IMAGE_WIDTH --height IMAGE_HEIGHT
```

### Arguments:

- `--count` (default: 1): Number of random noise images to generate.
- `--width` (default: 224): Width of each generated image.
- `--height` (default: 224): Height of each generated image.
- `--output_dir` (mandatory): Directory where the generated images will be saved.
- `--format` (default: `png`): Image format, which can be `png`, `jpg`, or `jpeg`.
- `--quality` (default: 95): JPEG quality (1-100), applicable only if the format is `jpg` or `jpeg`.

### Example Commands:

1. **Generate 5 random noise images with the default size (224x224) and save as PNG in the `output/` folder:**

```bash
python generate_noise_images.py --count 5 --output_dir output
```

2. **Generate 3 random noise images of size 512x512 and save them as JPG with 90% quality:**

```bash
python generate_noise_images.py --count 3 --width 512 --height 512 --output_dir output --format jpg --quality 90
```

3. **Generate 10 random noise images of size 128x128 and save them as JPEG in the `noise_images/` folder with default settings:**

```bash
python generate_noise_images.py --count 10 --width 128 --height 128 --output_dir noise_images --format jpeg
```

## Output

The images will be saved in the specified directory (`--output_dir`). The filename format is as follows:

```
<width>_<height>_<index>.<format>
```

For example, an image saved with the dimensions `224x224` and the index `0` would have the filename `224_224_0.png`.

## License

This project is licensed under the MIT License.