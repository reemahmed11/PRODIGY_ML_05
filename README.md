# PRODIGY_ML_05  

**Food Image Classification with VGG16 and CNN**

## Overview

This project focuses on classifying food images using Convolutional Neural Networks (CNN) and a pre-trained VGG16 model. The goal is to accurately recognize food items and estimate their calorie content from images. The dataset used is the Food-101 dataset, which includes various types of food images.

## Dataset

The dataset used for this project is the Food-101 dataset, which contains 101 food categories with 101,000 images. Each category has 1,000 images of food items.

- **Dataset Link:** [Food-101 Dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)](https://www.kaggle.com/datasets/dansbecker/food-101)

## Model Implementation

### Convolutional Neural Network (CNN)

#### Steps Involved:

1. **Image Preprocessing:**
    - Images are resized to a uniform size of 128x128 pixels.
    - Pixel values are normalized to be between 0 and 1.

2. **Model Architecture:**
    - A CNN model is built from scratch with several convolutional layers, activation functions, pooling layers, and fully connected layers.

3. **Model Training:**
    - The CNN model is trained on the preprocessed images. Training includes data augmentation to improve generalization.

4. **Evaluation:**
    - The model’s performance is evaluated using accuracy, confusion matrix, and classification report metrics.

### VGG16 Implementation

#### Steps Involved:

1. **Using Pre-trained VGG16:**
    - VGG16 is used as a pre-trained model with weights from ImageNet.
    - The top layer is removed, and a new classification head is added for the Food-101 categories.

2. **Model Fine-Tuning:**
    - The VGG16 model is fine-tuned on the Food-101 dataset, with careful management of computational resources due to its high requirements.

3. **Evaluation:**
    - The performance of the fine-tuned VGG16 model is evaluated with accuracy and other metrics.

### Challenges

- **Resource Constraints:**
    - Training the VGG16 model requires significant computational resources, which posed challenges in completing the training. The model was partially trained due to these constraints.

## Project Implementation

#### Data Preparation

- **Load and Preprocess the Data:**
    - Images from the dataset are resized, normalized, and organized for training and validation.

- **Train-Test Split:**
    - The dataset is split into training and validation sets to evaluate the model performance.

#### Model Training

- **CNN Model:**
    - A custom CNN model is trained and evaluated with data augmentation.

- **VGG16 Model:**
    - The pre-trained VGG16 model is adapted for food classification and fine-tuned.

#### Model Evaluation

- **Accuracy:**
    - The models’ accuracies are calculated on the validation set.

- **Confusion Matrix:**
    - A confusion matrix is plotted to visualize classification performance.

- **Classification Report:**
    - Detailed classification reports are generated, showing precision, recall, and F1-score.

#### Unseen Image Prediction

- **Prediction on Unseen Data:**
    - The trained models are used to predict the class of unseen images, with results visualized along with the input images.

## Results and Insights

- **Model Performance:**
    - The CNN model demonstrated effectiveness in classifying food images. The VGG16 model showed potential but was limited by computational resources.

- **Visualizations:**
    - Confusion matrix and sample predictions illustrate the performance of both models.

## Contact

For any questions or further information, please feel free to reach out:

- **Email:** reemahmedm501@gmail.com

## Contributing

If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

