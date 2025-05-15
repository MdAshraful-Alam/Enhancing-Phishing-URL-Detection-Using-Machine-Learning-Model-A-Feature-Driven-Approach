# 🔐 Enhancing Phishing URL Detection Using Machine Learning: A Feature-Driven Approach

This project is designed to support the research paper titled **"Enhancing Phishing URL Detection Using Machine Learning Model: A Feature-Driven Approach."** I have implemented the complete machine learning pipeline step by step as part of this research. The goal of this project is not only to replicate my paper’s results but also to provide clear guidance so others can build on top of my findings and extend the work further.

---

## 📁 Dataset

For this project, I worked with the **PhiUSIIL Phishing URL Dataset**, which is publicly available on Kaggle. The dataset contains over 235,000 URLs, and each entry is labeled as either phishing (`1`) or legitimate (`0`). What I liked about this dataset is that it includes 54 features that capture different characteristics of a URL, such as structure, domain elements, and content-based hints.

Dataset Link : https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

To use the dataset, I first downloaded it and placed it inside the `data` directory of my project like this:

```
project-root/
├── data/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv
```

---

## ⚙️ Environment Setup & Installation

To begin with, I created a virtual environment and used the `requirements.txt` file to install all necessary libraries. This ensured reproducibility and prevented version conflicts.

### Step 1: Clone the repository
```bash
git clone https://github.com/alam-alam/alam-phishing-url.git
cd Enhancing-Phishing-URL-Detection-Using-Machine-Learning-Model-A-Feature-Driven-Approach
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

This included all the libraries I used: `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, `joblib`, `transformers`, and `torch`. If someone wants to use GPU for faster processing with transformers, I recommend installing the PyTorch GPU version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## ✅ Step 1: Data Loading and Cleaning

The first thing I did was load the dataset into a DataFrame using pandas. I wanted to make sure the dataset was clean before doing any kind of analysis, so I removed duplicate records and checked for missing values. I also converted all text-based features to lowercase to ensure consistency.

```python
df = pd.read_csv('data/PhiUSIIL_Phishing_URL_Dataset.csv')
df.drop_duplicates(inplace=True)
df = df.applymap(lambda s: s.lower() if type(s) == str else s)
```

---

## 📊 Step 2: Exploratory Data Analysis (EDA)

To understand the data better, I performed exploratory data analysis. I visualized the distribution of URL lengths, number of subdomains, and top-level domains (TLDs). I also plotted the class distribution to check the balance between phishing and legitimate URLs.

Another important task was computing the correlation matrix. This helped me identify features that were highly correlated. Removing those redundant features helped reduce noise and made the model simpler.

```python
sns.histplot(df['url_length'])
sns.heatmap(df.corr(), annot=True)
```

Through this process, I noticed that phishing URLs tend to have slightly longer lengths and more subdomains on average.

---

## 🛠 Step 3: Feature Engineering

Based on the EDA, I created new features that I believed would help distinguish phishing URLs. For example:
- I added a `has_many_subdomains` binary feature if a URL had more than 3 subdomains.
- I flagged high-risk top-level domains (e.g., `.ru`, `.xyz`, `.tk`) using `is_high_risk_tld`.
- I also counted special characters like `@`, `-`, and `/`, which are often used in phishing URLs.

For categorical features like TLDs, I used OneHotEncoding, but I removed columns like domain name and filename that had high cardinality.

---

## 🎯 Step 4: Feature Selection

I didn’t want to use all 53 features for training because not all of them are useful. So, I applied two well-known statistical methods:

- **Chi-Square Test**: It ranked features based on how dependent they were on the target label. I selected the top 17.
- **Mutual Information Gain**: It ranked features by how much information they provide about the target. I picked the top 5.

Then I merged both sets and experimented with different combinations. I found that the best result came from selecting just four features: `TLD_mp`, `TLD_film`, `URLSimilarityIndex`, and `LineOfCode`. Using fewer features helped reduce computation time while maintaining accuracy.

---
# 🎯 Step 5 📊 Visualizations and Figures – Explanation and How to Reproduce

Here I describe how I generated each figure included in the paper, what it represents, and how to reproduce it step-by-step. This section is meant to help researchers replicate and build on my visual analyses.

### 🔹 **Figure 1 – Proposed Methodology Diagram**
This figure visually represents my end-to-end workflow for phishing detection using machine learning. I created it manually to reflect the entire process: dataset collection, preprocessing, EDA, feature selection, model training, evaluation, and saving the best model. You can recreate it using tools like draw.io or Lucidchart.

### 🔹 **Figure 2 – Distribution of Phishing vs. Legitimate URLs**
To understand class imbalance in the dataset, I plotted a count plot showing the number of phishing and legitimate URLs:
```python
sns.countplot(x='Label', data=df)
plt.title('Distribution of Phishing vs Legitimate URLs')
```
This revealed that phishing URLs made up approximately 43% of the dataset.

### 🔹 **Figure 3 – Histogram of URL Lengths**
I plotted this to check if phishing URLs tend to be longer than legitimate ones:
```python
sns.histplot(df['url_length'], bins=50)
plt.title('URL Length Distribution')
```
It showed that phishing URLs often have slightly longer structures.

### 🔹 **Figure 4 – Histogram of Subdomains**
This figure shows how often multiple subdomains are used:
```python
sns.histplot(df['subdomain_count'], bins=30)
plt.title('Histogram of Subdomains')
```
Phishing URLs often try to appear legitimate by embedding fake domains within subdomains.

### 🔹 **Figure 5 – Feature Correlation Matrix (Heatmap)**
I generated this to identify and remove redundant features:
```python
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
```
This helped me decide which features to remove due to high correlation (above 0.85).

### 🔹 **Figure 6 – URL Length by Label (Boxplot)**
This figure compares URL lengths between phishing and legitimate URLs:
```python
sns.boxplot(x='Label', y='url_length', data=df)
```
It revealed that phishing URLs generally had longer or unusual structures.

### 🔹 **Figure 7 – Subdomain Count by URL Type**
This was used to validate that phishing URLs use more subdomains to deceive users:
```python
sns.boxplot(x='Label', y='subdomain_count', data=df)
```

### 🔹 **Figure 8 – Top 10 TLDs in Phishing vs. Legitimate URLs**
This figure compared which top-level domains (TLDs) appear most often in phishing vs. legitimate samples:
```python
tld_counts = df.groupby(['TLD', 'Label']).size().unstack().fillna(0)
tld_counts.nlargest(10, 1).plot(kind='bar', stacked=True)
```

### 🔹 **Figure 9 – Confusion Matrix for Random Forest Model**
After evaluating the best model:
```python
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
```
This helped me visualize false positives and false negatives.

### 🔹 **Figure 10 – ROC Curve for Random Forest Model**
This shows the trade-off between sensitivity and specificity:
```python
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
```
It confirmed the model’s perfect performance (AUC = 1.00).

---

# 🎯 Step 6 📋 Tables – Explanation and Reproduction Guide

### 🟩 **Table I – Feature Selection Summary**
This table summarizes the results of applying Chi-Square and Mutual Information Gain. I performed feature ranking using:
```python
from sklearn.feature_selection import chi2, mutual_info_classif
```
Then I compiled the top 17 from Chi2 and top 5 from MI into a final combined list of 4 features.

### 🟩 **Table II – Best Hyperparameter Configuration for Random Forest**
This table lists the optimal parameter settings for my Random Forest model using:
```python
from sklearn.model_selection import RandomizedSearchCV
```
I ran it on a parameter grid with 5-fold cross-validation and recorded the best settings using `.best_params_`.

### 🟩 **Table III – Performance with 53 Features**
I trained all five models with all original features and collected their metrics:
- Accuracy
- Precision
- Recall
- F1 Score
These were stored using `classification_report()` and converted to table format.

### 🟩 **Table IV – Performance with 49 Features**
After removing high-correlation and high-cardinality features, I retrained the models and stored their updated metrics for this table.

### 🟩 **Table V – Chi-Square Feature Selection (Top 17)**
I selected 17 features using Chi-Square, retrained the models, and recorded all metrics. This confirmed that fewer features didn’t hurt performance.

### 🟩 **Table VI – Mutual Information Gain (Top 5 Features)**
Here I selected the top 5 features using MI, then retrained the same models, comparing test scores and generalization.

### 🟩 **Table VII – Performance with Top 1 Feature from Each Method (2 Features Total)**
This minimal set was used to evaluate how well just 2 top-ranked features (one from Chi-Square and one from MI) could perform.

### 🟩 **Table VIII – Performance with Top 2 Features from Chi-Square and MI (4 Features Total)**
This table shows that with only 4 well-selected features, I reached almost perfect performance.

### 🟩 **Table IX – Fold-wise Cross-validation Results for RF Model**
I collected train/test results for each of the 5 folds using `cross_val_score()` and visualized stability and consistency.

You can recreate all tables using `pandas.DataFrame()` and format them using `df.to_markdown()`, Excel, or LaTeX output.

## 📐 Step 7: Data Splitting and Scaling

To train the models, I split the data into training and testing sets using an 80/20 stratified split to keep the class ratio intact. Then, I applied MinMaxScaler to normalize the numerical features. This step is especially important for models like KNN and SVM, which are sensitive to the scale of the input data.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```

---

## 🤖 Step 8: Model Training

I wanted to compare the performance of several classical ML models. So, I trained:
- Logistic Regression
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Naive Bayes

I trained each model twice: once with all features, and once with the reduced set of four features. This helped me verify that my feature selection process didn’t lose predictive power.

```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

---

## 🔍 Step 9: Hyperparameter Tuning

After testing the base models, I focused on optimizing the Random Forest classifier because it performed best. I used `RandomizedSearchCV` to search over different combinations of hyperparameters such as:
- Number of trees (n_estimators)
- Maximum depth of the tree
- Minimum samples per split and leaf
- Whether to use bootstrap sampling

I used 5-fold cross-validation to make sure the results were stable and not just lucky guesses.

---

## 📈 Step 10: Evaluation Metrics

For evaluating each model, I used:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve and AUC score

These metrics helped me understand both the overall and class-wise performance. For instance, phishing detection relies heavily on recall (to avoid false negatives). I also visualized the confusion matrix and ROC curves.

```python
from sklearn.metrics import classification_report, roc_auc_score
```


---



## 📦 Output Files

- `random_forest_model.pkl`: My final trained model
- `.png` visualizations: Confusion matrix, ROC curve, histograms
- Feature selection summaries
- Cross-validation logs and reports

---




