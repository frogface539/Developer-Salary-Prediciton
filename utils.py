import pandas as pd
import joblib
import re 

model = joblib.load('Models/reg_model.pkl')

COLUMNS = [
    'YearsCodePro',
    'Country_Brazil',
    'Country_Canada',
    'Country_France',
    'Country_Germany',
    'Country_India',
    'Country_Italy',
    'Country_Netherlands',
    'Country_Other',
    'Country_Poland',
    'Country_Spain',
    'Country_Sweden',
    'Country_Switzerland',
    'Country_Ukraine',
    'Country_United Kingdom of Great Britain and Northern Ireland',
    'Country_United States of America',
    'EdLevel_Other',
    'EdLevel_Professional',
    'Employment_Employed, full-time;Employed, part-time',
    'Employment_Employed, full-time;Independent contractor, freelancer, or self-employed',
    'Employment_Employed, full-time;Independent contractor, freelancer, or self-employed;Employed, part-time',
    'Employment_Employed, full-time;Independent contractor, freelancer, or self-employed;Not employed, and not looking for work',
    'Employment_Employed, full-time;Independent contractor, freelancer, or self-employed;Student, part-time',
    'Employment_Employed, full-time;Independent contractor, freelancer, or self-employed;Student, part-time;Employed, part-time',
    'Employment_Employed, full-time;Independent contractor, freelancer, or self-employed;Student, part-time;Retired',
    'Employment_Employed, full-time;Not employed, and not looking for work',
    'Employment_Employed, full-time;Not employed, but looking for work',
    'Employment_Employed, full-time;Not employed, but looking for work;Employed, part-time',
    'Employment_Employed, full-time;Not employed, but looking for work;Independent contractor, freelancer, or self-employed',
    'Employment_Employed, full-time;Not employed, but looking for work;Independent contractor, freelancer, or self-employed;Employed, part-time',
    'Employment_Employed, full-time;Not employed, but looking for work;Not employed, and not looking for work;Employed, part-time',
    'Employment_Employed, full-time;Retired',
    'Employment_Employed, full-time;Student, full-time',
    'Employment_Employed, full-time;Student, full-time;Employed, part-time',
    'Employment_Employed, full-time;Student, full-time;Independent contractor, freelancer, or self-employed',
    'Employment_Employed, full-time;Student, full-time;Independent contractor, freelancer, or self-employed;Employed, part-time',
    'Employment_Employed, full-time;Student, full-time;Not employed, but looking for work',
    'Employment_Employed, full-time;Student, full-time;Not employed, but looking for work;Independent contractor, freelancer, or self-employed;Not employed, and not looking for work;Student, part-time;Employed, part-time;Retired',
    'Employment_Employed, full-time;Student, full-time;Not employed, but looking for work;Independent contractor, freelancer, or self-employed;Student, part-time;Employed, part-time',
    'Employment_Employed, full-time;Student, full-time;Student, part-time',
    'Employment_Employed, full-time;Student, full-time;Student, part-time;Employed, part-time',
    'Employment_Employed, full-time;Student, part-time',
    'Employment_Employed, full-time;Student, part-time;Employed, part-time',
    'Employment_Employed, part-time',
    'Employment_Employed, part-time;Retired',
    'Employment_Independent contractor, freelancer, or self-employed',
    'Employment_Independent contractor, freelancer, or self-employed;Employed, part-time',
    'Employment_Independent contractor, freelancer, or self-employed;Employed, part-time;Retired',
    'Employment_Independent contractor, freelancer, or self-employed;Not employed, and not looking for work',
    'Employment_Independent contractor, freelancer, or self-employed;Not employed, and not looking for work;Retired',
    'Employment_Independent contractor, freelancer, or self-employed;Not employed, and not looking for work;Student, part-time',
    'Employment_Independent contractor, freelancer, or self-employed;Retired',
    'Employment_Independent contractor, freelancer, or self-employed;Student, part-time',
    'Employment_Independent contractor, freelancer, or self-employed;Student, part-time;Employed, part-time',
    'Employment_Not employed, but looking for work',
    'Employment_Not employed, but looking for work;Employed, part-time',
    'Employment_Not employed, but looking for work;Independent contractor, freelancer, or self-employed',
    'Employment_Not employed, but looking for work;Independent contractor, freelancer, or self-employed;Employed, part-time',
    'Employment_Not employed, but looking for work;Independent contractor, freelancer, or self-employed;Not employed, and not looking for work',
    'Employment_Not employed, but looking for work;Independent contractor, freelancer, or self-employed;Retired',
    'Employment_Not employed, but looking for work;Independent contractor, freelancer, or self-employed;Student, part-time',
    'Employment_Not employed, but looking for work;Independent contractor, freelancer, or self-employed;Student, part-time;Employed, part-time',
    'Employment_Not employed, but looking for work;Student, part-time;Employed, part-time',
    'Employment_Retired',
    'Employment_Student, full-time;Employed, part-time',
    'Employment_Student, full-time;Independent contractor, freelancer, or self-employed',
    'Employment_Student, full-time;Independent contractor, freelancer, or self-employed;Employed, part-time',
    'Employment_Student, full-time;Independent contractor, freelancer, or self-employed;Not employed, and not looking for work',
    'Employment_Student, full-time;Independent contractor, freelancer, or self-employed;Student, part-time;Employed, part-time',
    'Employment_Student, full-time;Not employed, but looking for work;Employed, part-time',
    'Employment_Student, full-time;Not employed, but looking for work;Independent contractor, freelancer, or self-employed',
    'Employment_Student, part-time;Employed, part-time',
    'RemoteWork_In-person',
    'RemoteWork_Other',
    'RemoteWork_Remote',
    'OrgSize_10 to 19 employees',
    'OrgSize_10,000 or more employees',
    'OrgSize_100 to 499 employees',
    'OrgSize_2 to 9 employees',
    'OrgSize_20 to 99 employees',
    'OrgSize_5,000 to 9,999 employees',
    'OrgSize_500 to 999 employees',
    'OrgSize_I don’t know',
    'OrgSize_Just me - I am a freelancer, sole proprietor, etc.',
    'OrgSize_Other'
]

def preprocess_input(input_dict):
    row = {col: 0 for col in COLUMNS}

    try:
        row['YearsCodePro'] = float(input_dict.get('YearsCodePro', 0))
    except ValueError:
        row['YearsCodePro'] = 0.0

    def normalize(value):
        return re.sub(r'\s*;\s*', ';', value.strip())

    # One-hot encode the categorical fields
    for prefix in ['Country', 'EdLevel', 'Employment', 'RemoteWork', 'OrgSize']:
        raw_value = input_dict.get(prefix, '')
        if not raw_value:
            continue

        key = f"{prefix}_{normalize(raw_value)}"

        if prefix == "OrgSize" and key == "OrgSize_I dont know":
            key = "OrgSize_I don’t know"

        if key in row:
            row[key] = 1

    return pd.DataFrame([row])

def predict_salary(input_dict):
    df = preprocess_input(input_dict)
    prediction = model.predict(df)
    return round(float(prediction[0]), 2)
