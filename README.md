# Sentiment Analysis on IMDB Dataset Using RNN
This project implements a sentiment analysis model using a Recurrent Neural Network (RNN) on the IMDB movie review dataset. The model predicts whether a review expresses a positive or negative sentiment. It includes hyperparameter tuning with Optuna, techniques to reduce overfitting, and visualizations of training and test losses.

## Features
Preprocessing of text data using tokenization and padding.
RNN architecture with LSTM layers for text classification.
Hyperparameter optimization using Optuna.
Regularization techniques to reduce overfitting.
Evaluation of model performance with loss curves.
Customizable for sentiment prediction on new reviews.

## Dataset
The project uses the IMDB Dataset.csv, which contains 50,000 movie reviews labeled as positive or negative. The dataset was preprocessed to prepare it for training the model.

## Model Architecture
Embedding Layer: Converts words to dense vectors of fixed size.
LSTM Layers: Captures sequential dependencies in text.
Dropout Layers: Prevents overfitting.
Dense Layer: Outputs the final prediction (positive or negative sentiment).

## Key Results
Accuracy: Achieved 88% accuracy on the test dataset.
Overfitting Reduction: Applied dropout layers and regularization techniques to improve generalization.
Loss Curves: Visualized the training and validation losses to monitor performance.

## Model Deployment
The trained model was saved using Keras and can be loaded for making predictions:
from tensorflow.keras.models import load_model
model = load_model('sentiment_analysis_model.h5')

## Test a new review:
new_review = ["The movie was absolutely fantastic!"]
# Preprocess the review and make predictions using the model.

## Visualizations
Loss Curve: Shows the training and validation loss during the training process.
Hyperparameter Tuning: Fine-tuned the learning rate, dropout rates, and LSTM units using Optuna.

## Technologies Used
Python: Programming language.
TensorFlow/Keras: Deep learning framework.
Optuna: Hyperparameter optimization library.
Matplotlib/Seaborn: Visualization libraries.

## Future Scope
Explore transformer-based models like BERT for better performance.
Build a web app for real-time sentiment analysis using Streamlit or Flask.
Deploy the model using AWS, Heroku, or Google Cloud Platform.
