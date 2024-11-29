# Food Hazard Detection

This repository contains code for building machine learning models to detect food-related hazards and categorize them into predefined categories. It demonstrates the progression from a basic Support Vector Machine (SVM) model to an advanced BioBERT-based classification system.

## Basic Model Method: SVM
Support Vector Machines (SVM) were used as the baseline model for this task:

- **Text Encoding**: The `text` column was transformed using TF-IDF (with a maximum of 3,500 features).
- **Model Training**: Separate SVM classifiers with a linear kernel were trained for each target (`hazard-category`, `product-category`, `hazard`, and `product`).
- **Performance**: The model achieved reasonable results but was limited by the simplicity of the feature representation and the lack of contextual embeddings.

### Why SVM?
SVM was chosen as the basic model due to its simplicity and efficiency in handling smaller datasets. While effective for basic classification tasks, its performance is constrained when compared to modern transformer-based methods like BioBERT.

## Advanced Model Method: BioBERT
The advanced model leverages BioBERT for sequence classification:

- **Deep Learning Framework**: Built with PyTorch and HuggingFace Transformers.
- **Pre-trained Model**: BioBERT (`dmis-lab/biobert-base-cased-v1.1`) was fine-tuned for this task.
- **Features**:
  - Uses BioBERT embeddings for text encoding.
  - Supports multi-target prediction across hazard and product categories.
  - Saves predictions for both primary tasks.

## About
This repository was created to participate in [SemEval 2025 Task 9: The Food Hazard Detection Challenge](https://food-hazard-detection-semeval-2025.github.io/). The challenge evaluates explainable classification systems for titles of food-incident reports collected from the web. These algorithms aim to assist automated crawlers in identifying and extracting food safety issues from web sources, including social media. Transparency is a crucial aspect of this task due to the potential high economic impact of food hazards.

## Features
- Uses BioBERT for sequence classification tasks.
- Trains on labeled food hazard data and validates on unlabeled data.
- Implements custom text preprocessing and label encoding.
- Supports multi-target prediction across hazard categories and product categories.
- Saves predictions for both primary tasks.

## Requirements
- Python 3.7+
- Google Colab environment (with Google Drive integration)
- Libraries:
  - `torch`: For building and training the deep learning model.
  - `transformers`: For pre-trained BioBERT models and tokenization.
  - `pandas`: For loading and preprocessing structured data.
  - `scikit-learn`: For label encoding (`LabelEncoder`) and evaluation metrics (`f1_score`, `classification_report`).
  - `tqdm`: For progress bars during training.
  - `re`: For text preprocessing (removing special characters, cleaning text).

## Configuration
The model is configured with the following settings:
- **Maximum Sequence Length:** 256
- **Batch Size:** 16
- **Learning Rate:** 0.00005
- **Epochs:** 50
- **Early Stopping Patience:** 6
- **Model Name:** `dmis-lab/biobert-base-cased-v1.1`

## Setup and Execution

1. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Install Required Libraries:**
   ```bash
   pip install torch transformers pandas scikit-learn tqdm
   ```

3. **Load and Preprocess Data:**
   - Download the labeled training data and load it into a DataFrame.
   - Clean the text using a custom function.
   - Apply similar preprocessing to the unlabeled validation data.

4. **Train the Model:**
   - Train separate models for each target (`hazard-category`, `product-category`, `hazard`, `product`).
   - Use early stopping to avoid overfitting.
   - Save the best-performing model for each target.

5. **Generate Predictions:**
   - Load the saved models.
   - Generate predictions on the unlabeled validation data.
   - Save the predictions to CSV files.

## Data
- **Note:** For training, the `text` column was used instead of the `title` column, as it provided better predictions. This improvement is likely due to the additional context and detailed information available in the `text` column.
- **Labeled Training Data:** Contains food-related hazard information and associated categories.
  - URL: `https://raw.githubusercontent.com/food-hazard-detection-semeval-2025/food-hazard-detection-semeval-2025.github.io/refs/heads/main/data/incidents_train.csv`
  - Alternatively, located in the folder `data` as the file `incidents_train.csv`.
- **Unlabeled Validation Data:** This dataset is used by the competition to evaluate the results.
  - Located in the `data` folder as a zip file named `public_data.zip`, which contains the file `incidents.csv`.

## Output
- **Predictions:**
  - `ST1_predictions.csv`: Contains text, predicted hazard and product categories.
  - `ST2_predictions.csv`: Contains text, predicted hazard and product names.
- **Processed Predictions:**
  - `ST1_predictions_cleaned.csv`: Contains predicted hazard and product categories.
  - `ST2_predictions_cleaned.csv`: Contains predicted hazard and product names.

## Results

The performance of the model is evaluated using the **F1 Macro** score on two tasks:

### Task 1: Hazard and Product Categories
- **F1 Macro:** 0.75

### Task 2: Hazard and Product Names
- **F1 Macro:** 0.45

### Observations
- Task 1 generally performs better due to the well-defined categorical nature of hazard and product categories.
- Task 2 is more challenging, as it involves predicting specific product and hazard names, which may require finer-grained understanding.

### Scoring Function
The F1 Macro score is computed as follows:

```python
from sklearn.metrics import f1_score

def compute_score(hazards_true, products_true, hazards_pred, products_pred):
  # Compute F1 for hazards
  f1_hazards = f1_score(
    hazards_true,
    hazards_pred,
    average='macro'
  )

  # Compute F1 for products
  f1_products = f1_score(
    products_true[hazards_pred == hazards_true],
    products_pred[hazards_pred == hazards_true],
    average='macro'
  )

  return (f1_hazards + f1_products) / 2.
```

This metric emphasizes the importance of correctly predicting both hazards and products. A perfect score of 1.0 indicates that both hazards and products are entirely accurate, while hazards alone being correct results in a score of 0.5.

## How to Run
1. Clone this repository and upload it to Google Colab.
2. Ensure your Google Drive is mounted.
3. Update file paths and configuration settings if needed.
4. Run the script to train models and generate predictions.

## Model Architecture
- **Tokenizer:** HuggingFace's `AutoTokenizer`
- **Model:** HuggingFace's `AutoModelForSequenceClassification`
- **Loss Function:** Cross-Entropy Loss
- **Optimizer:** Adam
- **Learning Rate Scheduler:** ReduceLROnPlateau

## File Structure
```
|-- incidents_train.csv    # Labeled training data
|-- incidents.csv          # Unlabeled validation data
|-- ST1_predictions.csv    # Predictions for ST1
|-- ST2_predictions.csv    # Predictions for ST2
|-- ST1_predictions_cleaned.csv # Processed ST1 predictions
|-- ST2_predictions_cleaned.csv # Processed ST2 predictions
```

## License
This project is licensed under the GNU General Public License Version 3, 29 June 2007. See the LICENSE file for details.

## Acknowledgments
- [HuggingFace Transformers](https://huggingface.co/transformers/): For providing pre-trained models and tokenizer utilities that formed the backbone of this project.
- [BioBERT](https://github.com/dmis-lab/biobert): For offering a domain-specific transformer model optimized for biomedical and text classification tasks.
- [PyTorch](https://pytorch.org/): For being the core deep learning framework used to implement and train the neural network.
- [ChatGPT](https://openai.com/chatgpt): For providing insights and suggestions during the development of this project.
- [Stack Overflow](https://stackoverflow.com/): For being an invaluable resource for resolving coding challenges and gaining technical insights.

