# Food Hazard Detection with BioBERT

This repository contains code for a sequence classification model built using BioBERT to detect food-related hazards and categorize them into predefined categories. 
The solution is built in Python using PyTorch and HuggingFace Transformers.

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
  - torch
  - transformers
  - pandas
  - scikit-learn
  - tqdm

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
   !pip install torch transformers pandas scikit-learn tqdm
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
- **Labeled Training Data:** Contains food-related hazard information and associated categories.
  - URL: `https://raw.githubusercontent.com/food-hazard-detection-semeval-2025/food-hazard-detection-semeval-2025.github.io/refs/heads/main/data/incidents_train.csv`
- **Unlabeled Validation Data:** Located in your Google Drive.

## Output
- **Predictions:**
  - `ST1_predictions.csv`: Contains predicted hazard and product categories.
  - `ST2_predictions.csv`: Contains predicted hazard and product names.
- **Processed Predictions:**
  - `ST1_predictions_cleaned.csv`
  - `ST2_predictions_cleaned.csv`

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
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [BioBERT](https://github.com/dmis-lab/biobert)
- [PyTorch](https://pytorch.org/)
- [ChatGPT](https://openai.com/chatgpt) for assisting with code and documentation development.


