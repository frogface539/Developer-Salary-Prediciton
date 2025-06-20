# 💼 Salary Prediction App

A machine learning project to predict developer salaries based on their background using a regression model and an interactive Streamlit frontend.

## 🔍 Project Overview

This app uses real-world survey data to predict annual developer salaries. It asks users for:
- Country
- Years of professional experience
- Education level
- Remote work type
- Company size
- Employment combinations (full-time, freelance, student, etc.)

Then it returns a **salary prediction** using a trained regression model.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** for data wrangling
- **Scikit-learn** for regression modeling
- **Streamlit** for web UI
- **Joblib** for model persistence

---

## ⚙️ File Structure

```bash
salary-predictor/
│
├── app.py              # Streamlit web app
├── utils.py            # Preprocessing logic (feature encoding)
├── Models/
│   └── reg_model.pkl   # Trained regression model
└── requirements.txt    # All dependencies

📦 Features
Handles complex multi-label employment combinations.

Ensures prediction input features match exactly with the model training features.

Clean modular design: all preprocessing is abstracted in utils.py.

🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/salary-predictor.git
cd salary-predictor
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
🧠 Model Details
Trained using Linear Regression on a curated dataset

All categorical variables were one-hot encoded with full column alignment

Saved using joblib for fast loading in the app

📈 Future Improvements
Add visual salary insights by country or job type

Include more advanced models like XGBoost or RandomForest

Deploy to public cloud (Streamlit Cloud or HuggingFace Spaces)

## 📊 Dataset

The original dataset (`survey_results_public.csv`) is not included in this repository due to GitHub's 100MB file limit.

You can download it manually from the following official source:

🔗 [2024 Stack Overflow Developer Survey Dataset] https://survey.stackoverflow.co/datasets/stack-overflow-developer-survey-2024.zip

> Once downloaded, place the file in the following folder:
    Notebooks/survey_results_public.csv
📬 Contact
Feel free to connect or contribute!

Author: [Lakshay Jain]
LinkedIn: [https://www.linkedin.com/in/lakshay-jain-a48979289/]
GitHub: [https://github.com/frogface539]