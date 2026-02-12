Evacuation Readiness: Predictive Modeling for Crisis Response

This repository contains a high-performance solution for the Evacuation Readiness binary classification challenge. The project focuses on leveraging advanced boosting techniques and rigorous validation strategies to predict individual evacuation outcomes.

🚀 Performance SummaryTop Model: 
  CatBoost Classifier.Peak Accuracy: 80.996% (achieved via 5-fold Cross-Validation).
  Optimization: Fine-tuned probability thresholds to maximize classification accuracy.
  
🛠️ Technical Implementation

  1. Feature Engineering & Preprocessing
     To improve model convergence and prevent overfitting, the following transformations were implemented:
       Social Context: Engineered FamilySize and a binary IsAlone flag from raw dependent data.
       Categorical Cleaning: Extracted passenger titles using regex, grouping low-frequency titles (e.g., "Dr", "Rev", "Sir") into a "Rare" category to reduce noise.
       Imputation: Utilized group-based median imputation for missing values, grouping by ServiceTier and Gender to maintain data distribution integrity.
       Log Normalization: Applied np.log1p transformations to financial features like TransactionValue to normalize skewed distributions.
  2. Model Selection & Comparison
     While several ensemble methods were tested, CatBoost outperformed others due to its specialized handling of high-cardinality categorical features.
     Model      Peak Accuracy
     CatBoost      80.996%
     Random Forest  80.373%
     XGBoost        79.127%
     LightGBM       77.880%
  3. Validation Strategy
     Stratified K-Fold: Implemented a 5-split strategy to ensure consistent class representation across all folds.
     Early Stopping: Used a 100-round early stopping criteria to halt training once the validation Log-Loss plateaued, preventing memorization of training noise.
     Probability Ensembling: Captured Out-of-Fold (OOF) probabilities to allow for precise threshold tuning rather than relying on default 0.5 hard labels.
     
📂 Project Structure
  CatBoostIntras.ipynb: Full end-to-end pipeline from EDA to submission.
  train.csv / test.csv: Competition datasets.
  submission.csv: Final output file with predicted evacuation outcomes
