# Sarcasm Detector: Classifying the Unspoken

## Overview
This project focuses on building a sarcasm detection model using Natural Language Processing (NLP) techniques and deep learning. By training on a dataset of news headlines, the model can accurately identify sarcastic statements from non-sarcastic ones. The goal is to achieve at least 80% accuracy and validation, making the classifier effective for real-world applications like content moderation, sentiment analysis, or social media monitoring.

## Features
- **Binary Classification**: Detects whether a given text is sarcastic or not.
- **Bidirectional LSTM**: Utilizes Bidirectional Long Short-Term Memory layers for better context understanding.
- **Efficient Tokenization and Padding**: Preprocesses text data to improve model performance.
- **High Accuracy**: Designed to meet a minimum validation accuracy of 80%.

## Dataset Source
The model uses the [Sarcasm Detection Dataset](http://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json), which contains headlines from news articles labeled as sarcastic or not. Each data point includes:
- `headline`: The news headline.
- `is_sarcastic`: Label (1 if sarcastic, 0 if not).
- `article_link`: The original link to the news article (not used in training).

## Model Architecture
1. **Embedding Layer**: Converts words into numerical vectors.
2. **Bidirectional LSTM Layers**: Two layers for capturing both forward and backward dependencies.
3. **Dense Layer**: Adds non-linearity with ReLU activation.
4. **Dropout Layer**: Prevents overfitting by randomly disabling neurons during training.
5. **Output Layer**: A single neuron activated by sigmoid for binary classification.

## Installation & Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/allanbil214/nn_SarcasmDetector.git
    cd nn_SarcasmDetector
    ```

2. Install required libraries:
    ```bash
    pip install tensorflow numpy
    ```

3. Run the model:
    ```bash
    python nn_SarcasmDetector.py
    ```

## Usage
The model will download the dataset automatically and start training. After training, the model will be saved as `model.h5`, which can be loaded for further inference tasks.

## Results
The model consistently achieves over 80% validation accuracy, ensuring reliable sarcasm detection on unseen text.

## Contributing
Feel free to fork this project, suggest improvements, or report issues. Contributions are welcome!

## License
This project is licensed under the MIT License.

## Acknowledgements
- TensorFlow for providing the dataset.
- Open-source contributors for their invaluable resources and tools.

---
