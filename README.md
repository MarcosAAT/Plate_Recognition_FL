# Florida Plate Recognition Model

A simple machine learning model capable of recognizing Florida license plates. This project utilizes the ResNet18 architecture and has been trained with images of license plates from Florida as well as from other states.

## Features

- **Dual Output:** The model outputs "Florida" for license plates from the state of Florida and "Non-Florida" for plates from other states.
- **ResNet18 Architecture:** Utilizes the ResNet18 model for image recognition, providing a solid foundation for accuracy and performance.
- **Training Data:** Trained using a curated dataset of license plate images from various states, with a focus on distinguishing Florida plates.

## Key Observations

- **Data Limitations:** The model requires further training with a larger and more robust dataset to improve its accuracy and reliability for this task. 
- **Default Behavior:** In cases where the license plate is not recognized or is absent from the training data, the model defaults to guessing the plate as "Florida." This behavior is a placeholder and highlights the need for expanded dataset coverage.

## Model Background

This project builds upon the ResNet18 architecture, a pre-trained model widely used for image recognition tasks. The ResNet18 model, developed by researchers at Microsoft, has been trained on the ImageNet dataset, which includes millions of images across a wide range of categories.

## Usage

This project includes a `test.py` script for recognizing license plates in images. To use this script, you will need to specify the path to the image you want to analyze. Here's how to do it:

1. **Locate the `test.py` File:** Open the `test.py` file in your preferred code editor. This file is part of the project's repository.

2. **Modify the Image Path Variable:** Within `test.py`, look for the variable that specifies the image path. It will be named `image_path`. Replace the default path with the path to your desired image file. Or if they are in the same file, the name of the image.

### Acknowledgments

- ResNet18 was developed by the AI research team at Microsoft. More details about the model and its training can be found in their published paper, "Deep Residual Learning for Image Recognition."
- The dataset used for fine-tuning the model includes images sourced from publicly available datasets and contributions from the community.
