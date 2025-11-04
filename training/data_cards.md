# Data Card — SMS Spam Collection (Kaggle)

- **Source**: https://www.kaggle.com/datasets/tinu10kumar/sms-spam-dataset and https://www.kaggle.com/datasets/gevabriel/indonesian-sms-spam
- **License**: CC BY-NC-SA 4.0
- **Content**: ~11k SMS across English and Indonesian (spam/ham).
- **Preprocessing**:
  - Lowercasing
  - Basic URL/number normalization
  - Preserve punctuation tokens; normalise URLs/numbers/currency; lowercase text
- **Intended use**: Educational demo for spam detection UI and API.
- **Risks**:
  - Class imbalance
  - Non-commercial redistribution prohibited for raw dataset
- **Provenance**: Original UCI repository (via Kaggle mirror).

- **Modeling**: TF-IDF + Logistic Regression with 5-fold stratified hyperparameter search (balanced class weights).
