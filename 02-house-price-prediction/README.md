# House Price Prediction using Linear Regression

Predicts house prices from area, number of bedrooms, bathrooms, stories, parking, and amenity features (main road, guest room, basement, hot water heating, air conditioning, preferred area, furnishing status), using multiple linear regression.

## Dataset
- `house_price.csv`: 545 records, 13 columns (`price` is the target)
- No missing values or duplicate rows
- Source: *(Dataset provided as part of a course assignment)*

## Method
1. Loaded the data and checked for missing values and duplicates
2. Encoded the categorical columns with `LabelEncoder`
3. Split the data into 80% training and 20% testing
4. Scaled the features with `StandardScaler`, fitted on the training set only to avoid data leakage
5. Trained a `LinearRegression` model
6. Evaluated it on the test set

## Results (test set, 109 samples)
| Metric | Value |
|--------|-------|
| MAE | ≈ 716,583 |
| RMSE | ≈ 992,996 |
| R² | ≈ 0.658 |

The average price in the dataset is about 4.77 million, so the MAE is roughly 15% of a typical price. The model explains about two-thirds of the variation in price.

## Limitations and next steps
- Linear regression assumes a straight-line relationship, and house prices are usually right-skewed
- Use one-hot encoding for `furnishingstatus`
- Try a log transform of `price`
- Compare with Ridge, Lasso or a random forest
- Add cross-validation

## How to run
```bash
pip install pandas scikit-learn
```
Open `house_price_prediction.ipynb` in Jupyter or Google Colab and run all cells.

## Tools
Python, pandas, scikit-learn
