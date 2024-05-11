# Practical Optical Camera Communication Behind Unseen and Complex Backgrounds (MobiSys 2024)

This repository contains the source code associated with the paper presented at MobiSys 2024:

> **[MobiSys 2024] Practical Optical Camera Communication Behind Unseen and Complex Backgrounds**  
> Authors: Rui Xiao, Leqi Zhao, Feng Qian, Lei Yang, and Jinsong Han  
> ACM International Conference on Mobile Systems, Applications, and Services

## Tested Environment

Our code has been tested and confirmed to work in the following setup:

- **MATLAB**: Version 2023b
- **Python**: Version 3.9.13
- **PyTorch**: Version 1.11.0
- **PyTorch Lightning**: Version 1.7.5

## Getting Started

To use the code, follow these steps:

### Preparation

Ensure that your environment matches the tested setups mentioned above. If not, you might need to install the necessary versions of MATLAB, Python, and the libraries.

### Running the Code

1. **Navigate to the DNN Directory**:
   ```bash
   cd dnn
   ```

2. **Execute the Training Script**:
   ```bash
   python train.py
   ```

   This will initiate the training process using PyTorch Lightning and generate a `logs` directory containing the following:
   - **Origin**: Original images with synthetic stripes.
   - **Reconstructions**: Images with stripes removed by the DNN.
   - **stripes_gt**: Ground truth data for the stripes.
   - **stripes_predict**: Stripes as predicted by the DNN.

### Data and Model Weights

This section details the organization and contents of the `data` and `model_weights` folders:

- **Data**:
  - The `data` folder contains a dataset of 7,245 images, each sized 128x128 pixels. Please download at https://drive.google.com/drive/folders/1Sduz3lq81CXbSBtxrSunDKV7p-H0pkHE?usp=drive_link. It is structured as follows:
    - **origin_*.npy**: These files store the original images without stripes. Each file follows the format `(n_images, n_channel, H, W)`.
    - **striped_*.npy**: These files include the same images but with synthetic stripes added. The format is identical to the `origin_*.npy` files.
    - **freqs.npy**: Contains the ground truth for the 1D stripes with a shape of `(n_images, 512)`.

- **Model Weights**:
  - The `model_weights` folder contains pretrained model weights that are compatible with images of three different sizes: 128x128, 256x256, and 512x512 pixels. These weights are crucial for deploying the trained models on your dataset without the need for retraining from scratch.

This structure ensures that users can easily navigate the provided resources, understand the


## Citing Our Work

If our work contributes to your research, please consider citing it using the following Bibtex entry:

```bibtex
@inproceedings{winklink-mobisys24,
  title     = {Practical Optical Camera Communication Behind Unseen and Complex Backgrounds},
  author    = {Rui Xiao and Leqi Zhao and Feng Qian and Lei Yang and Jinsong Han},
  booktitle = {Proceedings of the 22nd Annual International Conference on Mobile Systems, Applications, and Services (MobiSys 2024)},
  pages     = {xxx--xxx},
  year      = {2024},
  publisher = {ACM},
  address   = {Tokyo, Japan},
  date      = {June 3-7, 2024}
}
```
