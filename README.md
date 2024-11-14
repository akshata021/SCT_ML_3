This code involves building an SVM-based image classification model to differentiate between images of cats and dogs.
steps involved :
1. Dataset Download and Preparation: The code begins by downloading a dataset from Kaggle, specifically a "Dogs vs. Cats" dataset. Images are unzipped and organized into directories for each category.

2. Libraries and Constants Setup: Essential libraries such as OpenCV, scikit-learn, Matplotlib, and TensorFlow are imported. The image size is set to 128x128 pixels for compatibility with the VGG16 model.

3. Data Loading and Preprocessing: Images of cats and dogs are loaded into memory with labels. Images are resized to the defined dimensions, converted to arrays, and preprocessed to match the input format of the VGG16 model.

4. Feature Extraction Using VGG16: A pre-trained VGG16 model (up to the last convolutional block) is used to extract high-level features from the images. The features are then flattened to create feature vectors for each image.

5. Data Splitting and Scaling: The dataset is split into training and test sets, and features are standardized.

6. SVM Training: An SVM model with an RBF kernel is trained on the training set using optimized hyperparameters (C=100, gamma=0.001). The model is then evaluated on the test set, and accuracy is printed.

7. Confusion Matrix: A confusion matrix visualizes the model’s performance on the test set, showing correct and incorrect classifications.

8. Cross-Validation: Stratified K-Fold cross-validation (5 splits) is applied to evaluate the model's performance on different data splits. Training and validation accuracies across the splits are plotted.
