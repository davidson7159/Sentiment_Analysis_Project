# 📱 SMS Spam Classification (NLP Project)

A complete end-to-end machine learning project to classify SMS messages as **Spam** or **Ham** using NLP techniques and classic ML models.

This project includes:
- Exploratory Data Analysis (EDA)
- Text preprocessing
- Feature extraction (TF-IDF)
- Model training and evaluation
- Model selection
- Export of final model + vectorizer
- CLI prediction script (`predict.py`)

---

## 🚀 Project Structure
sms-spam-nlp/
│
├── data/
│   ├── raw/                 # raw dataset
│   └── processed/           # cleaned dataset
│
├── notebooks/
│   ├── 01_EDA_SMS_Spam.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Model_Selection_SMS.ipynb
│
├── models/
│   ├── best_model.pkl       # saved model
│   └── tfidf_vectorizer.pkl # saved TF-IDF vectorizer
│
├── src/
│   ├── predict.py           # CLI prediction script
│
├── requirements.txt
└── README.md

---

## 📊 Dataset

The dataset used is the **SMS Spam Collection Dataset**, containing  
> 5,574 SMS messages labeled as `spam` or `ham`.

Source: Kaggle.

Columns used:
- **v1** → label (`ham` / `spam`)
- **v2** → message text

---

## 🔍 Exploratory Data Analysis

Key findings:
- Only **13%** of messages are spam → imbalanced dataset  
- Spam messages are generally **longer** than ham messages  
- Frequent spam keywords include: *free, call, now, claim, prize*  
- Frequent ham keywords include: *ok, tomorrow, home, thanks*

Visualizations include:
- Label distribution  
- Message length distribution  
- Most common spam vs ham words  

All results are in `01_EDA_SMS_Spam.ipynb`.

---

## 🧹 Text Preprocessing

Steps applied:
- Lowercasing  
- Removing punctuation  
- Removing stopwords  
- Tokenization  
- Lemmatization (optional)  
- Joining tokens back into processed text  

Final cleaned dataset saved as:
data/processed/sms_preprocessed.csv
Notebook: `02_Preprocessing.ipynb`

---

## 🧠 Modeling

Models trained:
- Logistic Regression  
- Multinomial Naive Bayes  
- Random Forest
- XGBoost  

Feature extraction:  
TfidfVectorizer(ngram_range=(1,2))

Evaluation metric:
- **F1-score** (important due to class imbalance)

Notebook: `03_Modeling_SMS_Spam.ipynb`

---

## 🏆 Best Model

The best performing model is:

👉 **XGBoost**

It provided the highest F1-score on the test set.

Saved files:
models/best_model.pkl
models/tfidf_vectorizer.pkl

---

## 🔮 How to Use the Prediction Script

You can predict whether a message is spam directly from the terminal:

### 1️⃣ Run the script
python src/predict.py “Congratulations! You’ve won a free prize”
### 2️⃣ Output
SPAM

Example ham:
python src/predict.py “Hey David, are we still meeting tonight?”
HAM

---

## 📦 Installation

Install dependencies:
pip install -r requirements.txt

---

## 📌 Next Steps (Future Work)

- Deploy a **Streamlit web app**
- Add a **FastAPI REST API**
- Improve preprocessing (stemming, lemmatization)
- Train transformer-based models (BERT, DistilBERT)
- Handle dataset imbalance with SMOTE
- Add hyperparameter tuning (GridSearch)

---

## 👨‍💻 Author

Davidson ADRIEN — Data Scientist & Machine Learning Enthusiast  
Project created for educational and portfolio purposes.

---