# Salary Prediction using Linear Regression

Predicts an employee's salary from department, worked hours, certifications and years of experience, using multiple linear regression.

## Dataset
- `salary_data.csv`: 30 records with 5 columns: `Department`, `WorkedHours`, `Certification`, `YearsExperience`, `Salary`
- Source: *(Dataset provided as part of a course assignment)*
- The dataset is very small, so the results are indicative rather than robust.

## Method
1. Loaded the data with pandas
2. Encoded the categorical column `Department`
3. Split the data into 70% training and 30% testing
4. Scaled the features with `StandardScaler`, fitted on the training set only to avoid data leakage
5. Trained a `LinearRegression` model
6. Evaluated it on the test set

## Results (test set, 9 samples)
| Metric | Value |
|--------|-------|
| MAE | ≈ 5,592 |
| RMSE | ≈ 6,766 |

`YearsExperience` has by far the largest influence on predicted salary.

## Limitations and next steps
- Only 30 rows, so the 9-sample test set gives noisy metrics
- Add cross-validation
- Use one-hot encoding for `Department`
- Compare against a model that uses `YearsExperience` alone

## How to run
```bash
pip install pandas scikit-learn
```
Open `salary_prediction_regression.ipynb` in Jupyter or Google Colab and run all cells.

## Tools
Python, pandas, scikit-learn
