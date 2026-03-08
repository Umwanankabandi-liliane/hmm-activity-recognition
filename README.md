# Human Activity Recognition using Hidden Markov Models


## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)



##  Overview

This project implements a **Hidden Markov Model (HMM)** based system for recognizing human activities from inertial sensor data. The system processes accelerometer and gyroscope readings from mobile devices to classify activities in real-time. This approach is valuable for applications in health monitoring, fitness tracking, and context-aware computing.

### Supported Activities
-  **Standing** - User is stationary and upright
-  **Walking** - User is in motion at walking pace
-  **Jumping** - User is performing jumping motions
-  **Still** - User is completely stationary



##  Key Features

- **Multi-sensor Fusion**: Combines accelerometer and gyroscope data for robust activity recognition
- **Feature Engineering**: Extracts statistical and frequency-domain features from sensor signals
- **HMM-based Classification**: Leverages temporal patterns using Gaussian Hidden Markov Models
- **Cross-subject Validation**: Trained on multiple subjects and tested on unseen data
- **Comprehensive Evaluation**: Includes confusion matrices, accuracy metrics, and classification reports
- **Visualization Tools**: Provides insightful plots for data exploration and model performance



##  Dataset

### Data Organization
```
data/
├── Hortance_data/    # Training set - Participant 1
│   ├── standing/     # Multiple recording sessions
│   ├── walking/
│   ├── jumping/
│   └── still/
├── Liliane_data/     # Training set - Participant 2
│   ├── Standing_01 to Standing_15/
│   ├── Walking_01 to Walking_15/
│   └── Jumping_01 to Jumping_15/
└── Unseen_data/      # Test set - Evaluation data
    ├── standing/
    ├── walking/
    ├── jumping/
    └── still/


### Sensor Data Files
Each activity recording contains:
- `Accelerometer.csv` - 3-axis acceleration data (x, y, z)
- `Gyroscope.csv` - 3-axis angular velocity data
- `Metadata.csv` - Recording information
- `Annotation.csv` - Activity labels and timestamps



##  Model Architecture

The system employs a **Gaussian Hidden Markov Model** approach:

1. **Feature Extraction Layer**
   - Time-domain features: mean, std, min, max, range
   - Statistical features: skewness, kurtosis, entropy
   - Frequency-domain features: FFT coefficients, spectral energy

2. **HMM Training**
   - Individual HMM per activity class
   - Gaussian emission distributions
   - Configurable number of hidden states
   - Covariance type: diagonal

3. **Classification**
   - Likelihood-based prediction
   - Activity class with maximum log-likelihood selected



##  Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd human-activity-hmm
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install packages individually:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn hmmlearn scipy
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```



##  Usage

### Quick Start

1. **Open the main notebook**
   ```bash
   jupyter notebook Untitled1.ipynb
   ```

2. **Execute cells sequentially** to:
   - Load and preprocess sensor data
   - Visualize activity patterns
   - Train HMM models for each activity
   - Evaluate on test data
   - Generate performance metrics

### Workflow Steps

```python
# 1. Data Loading
train_data = load_activity_data(train_paths)

# 2. Feature Extraction
features = extract_features(sensor_data)

# 3. Model Training
hmm_models = train_hmm_per_activity(features)

# 4. Prediction
predictions = predict_activities(test_data, hmm_models)

# 5. Evaluation
accuracy, confusion_mat = evaluate_model(predictions, true_labels)




##  Project Structure

```
human-activity-hmm/
│
├── data/                      # Sensor datasets
│   ├── Hortance_data/        # Participant 1 recordings
│   ├── Liliane_data/         # Participant 2 recordings
│   └── Unseen_data/          # Test/validation data
│
├── Untitled1.ipynb           # Main analysis notebook
├── README.md                  # Project documentation
└── requirements.txt          # Python dependencies
```



##  Methodology

### 1. Data Preprocessing
- Load accelerometer and gyroscope CSV files
- Merge sensor streams with timestamps
- Handle missing values and outliers
- Normalize sensor readings

### 2. Feature Engineering
Extract informative features from raw sensor data:
- **Time-domain**: Statistical moments (mean, variance, skewness, kurtosis)
- **Frequency-domain**: FFT coefficients, power spectral density
- **Information-theoretic**: Entropy measures
- **Signal characteristics**: Zero-crossing rate, signal magnitude area

### 3. Model Training
- Train separate Gaussian HMM for each activity class
- Optimize hyperparameters (number of states, covariance type)
- Use Baum-Welch algorithm for parameter estimation

### 4. Classification
- Compute forward log-likelihood for each activity model
- Assign test sequence to activity with highest likelihood
- Apply majority voting for sequence-level predictions

### 5. Evaluation
- Compute accuracy, precision, recall, F1-score
- Generate confusion matrix
- Perform cross-subject validation



##  Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3.x |
| **ML Framework** | scikit-learn, hmmlearn |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Signal Processing** | SciPy |
| **Environment** | Jupyter Notebook, Google Colab |



##  Results

The trained HMM models demonstrate strong performance in activity recognition:

- **High accuracy** on both training and test sets
- **Robust generalization** to unseen subjects
- **Confusion matrix** visualization highlights model strengths and weaknesses
- **Activity-specific metrics** via classification report

*Detailed results and visualizations are available in the notebook.*



##  Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Areas for Improvement
- Add more activity types
- Implement deep learning comparisons (LSTM, CNN)
- Real-time activity detection
- Mobile deployment
- Enhanced feature selection






