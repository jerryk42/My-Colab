# Food Hazard Detection with BioBERT

This repository contains code for a sequence classification model built using BioBERT to detect food-related hazards and categorize them into predefined categories. 
The solution is built in Python using PyTorch and HuggingFace Transformers.

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


