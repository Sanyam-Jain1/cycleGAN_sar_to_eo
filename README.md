**Made by- Sanyam Jain(24/A06/030) and Daksh Gangwar(24/A02/073)**

# SAR-to-RGB Image Translation with an Enhanced CycleGAN

This project uses a powerful, modified CycleGAN to translate complex Synthetic Aperture Radar (SAR) images into realistic, human-viewable RGB optical images. It's built on the original PyTorch-CycleGAN-and-pix2pix framework but is enhanced with a special ingredient: **Perceptual Loss**, which helps create much more visually convincing results.

This repository is designed to be run in a Google Colab notebook, allowing anyone to train and test the model using free GPU resources.

## The Magic Ingredient: What is Perceptual Loss?

Imagine you ask a computer to judge how "real" a generated photo of a landscape is.

A **standard loss function** (like L1 or L2 loss) acts like a meticulous checker. It compares the generated image to a real one pixel by pixel. If a pixel at coordinate (10, 20) is dark blue in the original and light blue in the generated image, it notes a small error. This method is good at getting the basic colors and placements right, but it often leads to blurry or slightly "off" textures because it doesn't understand the bigger picture.

A **perceptual loss function** acts like an art critic. Instead of looking at individual pixels, it uses a pre-trained neural network (in our case, the VGG19 network, which is already an expert at recognizing features in images) to look at both the real and generated images. It then compares the *high-level features* that the critic "sees"—things like textures, patterns, shapes, and the overall composition.

The loss is calculated based on how different these abstract features are. By minimizing this loss, we're not just telling the generator "make this pixel the right color," we're telling it "make this area *feel* like real grass" or "make this shape *look* like a real building." This results in images that are much more structurally sound and perceptually pleasing to the human eye.

---

## Key Features

- **Enhanced CycleGAN Architecture**: Based on the official `pytorch-CycleGAN-and-pix2pix` repository.
- **VGG19 Perceptual Loss**: Integrated directly into the generator's training loop to produce higher-quality, more realistic images.
- **Custom `.npy` Dataloader**: A special PyTorch dataset class is included to handle SAR data stored in the `.npy` format.
- **Quantitative & Qualitative Analysis**: Includes scripts to evaluate the model using SSIM/PSNR metrics and to visualize the input, output, and ground truth images side-by-side.

## How It Works

The project modifies the core files of the original CycleGAN framework to add the new functionality:

- **`models/networks.py`**: A `VGGPerceptualLoss` class was added. This class uses a pre-trained VGG19 network to extract feature maps and calculate the perceptual difference between images.
- **`models/cycle_gan_model.py`**: The main model file was updated to include the perceptual loss in the generator's total loss calculation, controlled by the `--lambda_perceptual` flag.
- **`data/unaligned_npy_dataset.py`**: A new dataloader was created to read `.npy` files, which is common for scientific imaging data like SAR.

## Setup and Installation (Google Colab)

1.  **Mount Your Google Drive**:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2.  **Clone the Repository**: Navigate to your project directory and clone the official CycleGAN repository.
    ```bash
    %cd /path/to/your/project_folder/
    !git clone [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git)
    ```

3.  **Install Dependencies**:
    ```bash
    %cd pytorch-CycleGAN-and-pix2pix/
    !pip install -r requirements.txt
    ```

4.  **Apply Code Modifications**: Run the cells in the `cycle_gans_perceptual_loss.ipynb` notebook that use the `%%writefile` magic command. This will automatically update the cloned code with the necessary changes for the perceptual loss and the `.npy` dataloader.

## How to Use the Model

### Training

To train the model, run the `train.py` script from your notebook. The `--lambda_perceptual` flag is crucial as it "turns on" our special loss function.

```bash
!python train.py \
  --dataroot /path/to/your/preprocessed_data/ \
  --name sar2rgb_perceptual \
  --model cycle_gan \
  --dataset_mode unaligned_npy \
  --input_nc 2 \
  --output_nc 3 \
  --no_flip \
  --lambda_identity 0 \
  --lr 0.0001 \
  --lambda_perceptual 1.0
```
- To **resume training** from a previous checkpoint, add the `--continue_train` flag.

### Testing (Generating Images)

Once your model is trained, you can use it to translate new SAR images with the `test.py` script.

```bash
!python test.py \
  --dataroot /path/to/your/test_data/sar_input \
  --name sar2rgb_perceptual \
  --model cycle_gan \
  --dataset_mode single_npy \
  --input_nc 2 \
  --output_nc 3 \
  --no_flip \
  --epoch latest \
  --eval
```
- The generated images will be saved in the `./results/sar2rgb_perceptual/test_latest/images/` directory.

### Evaluation & Visualization

The repository includes scripts to help you analyze the results:
- **`evaluate.py`**: Calculates the average Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR) to quantitatively measure image quality.
- **Visualization Scripts**: The notebooks contain code to generate side-by-side plots comparing the input SAR, the generated RGB image, and the ground truth optical image.

## Results

The addition of perceptual loss leads to a significant improvement in image quality, which is reflected in the evaluation metrics.

| Epoch | Avg. SSIM (Higher is better) | Avg. PSNR (Higher is better) |
| :---: | :---: | :---: |
|   5   |      **0.5475** |         **22.2409 dB** |

*These metrics show that the model learns to produce structurally similar and less noisy images early in the training process.*

## Dependencies

- Python 3.x
- PyTorch & Torchvision
- dominate, visdom, wandb
- NumPy, Matplotlib, scikit-image

## Acknowledgments

This work is built upon the foundational [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository by Jun-Yan Zhu and Taesung Park.
