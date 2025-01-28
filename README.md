# Crop Recommendation Model

A neural network model for recommending suitable crops based on various soil and environmental factors.

## Dataset
The dataset contains the following features:
- **N (Nitrogen)**
- **P (Phosphorus)**
- **K (Potassium)**
- **Temperature**
- **Humidity**
- **pH**
- **Rainfall**
- **Class (Crop type)**

## Preprocessing
Before training, the dataset undergoes preprocessing steps, including label encoding, scaling, and oversampling.

### Label Encoding
The categorical crop labels are converted into numerical format using `LabelEncoder` from `sklearn`.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['class_encoded'] = le.fit_transform(df['class'])
class_mapping = {i: class_name for i, class_name in enumerate(le.classes_)}
df = df.drop(columns=['class']).rename(columns={'class_encoded': 'class'})
```

### Data Scaling
Feature scaling is applied using `StandardScaler` to normalize the dataset.

### Oversampling
To balance the dataset, oversampling is applied to the training set using `RandomOverSampler`.

```python
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

def scale_dataset(X, y, oversample=False):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if oversample:
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_scaled, y)
        return X_resampled, y_resampled
    
    return X_scaled, y
```

## Neural Network Model
A feedforward neural network is implemented using TensorFlow/Keras with the following characteristics:
- **Input Layer:** 7 features (N, P, K, Temperature, Humidity, pH, Rainfall)
- **Two Hidden Layers:**
  - 128 neurons with ReLU activation and 30% dropout
  - 64 neurons with ReLU activation and 20% dropout
- **Output Layer:** Softmax activation for multi-class classification
- **Loss Function:** Sparse categorical cross-entropy
- **Optimizer:** Adam (learning rate = 0.0005)

## Performance
The model is trained and evaluated on the dataset, achieving high classification accuracy.

## Future Improvements
- Experiment with additional machine learning models such as Random Forest and SVM.
- Optimize hyperparameters for improved performance.
- Expand the dataset for better generalization.

## License
This project is licensed under the MIT License.

