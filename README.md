# Stack-Overflow-Based-Predictive-Analysis-of-Developer-Success
This project explores key factors influencing developer success by analyzing data from the 2024 Stack Overflow Developer Survey. Success is measured in terms of annual salary. The analysis uses machine learning model to derive insights and predict outcomes based on variables such as education, employment type, remote work setup, and years of experience.

## Project Overview
The study follows the CRISP-DM methodology to guide the workflow:

- Conducted exploratory data analysis (EDA) to identify patterns and data quality issues.

- Cleaned and preprocessed the data, handling missing values and converting text features.

- Built a predictive model for salary estimation (regression).

- Evaluated model performance and summarized findings.

## Technologies Used
- Python 3.10+

- Google Colab

- Libraries:
    - pandas, numpy – Data processing
      
    - matplotlib, seaborn – Visualization
      
    - scikit-learn – Modeling and evaluation

## Dataset

- Source: [Stack Overflow 2024 Developer Survey](https://survey.stackoverflow.co/2024)

- File: survey_results_public.csv

- Size: Over 65,000 responses (note: exceeds GitHub's 25MB upload limit)

- Features used: Education level, employment type, remote work status, years of coding experience, job satisfaction, and salary

## Modeling Process

1. Salary Prediction Model

- Target Variable: ConvertedCompYearly

- Type: Regression

- Model Used: Linear Regression

- Features: EdLevel, Employment, RemoteWork, YearsCodePro

## Evaluation Metrics

Salary Prediction:

- Mean Absolute Error (MAE): ~$50,500

- R² Score: ~0.06

## Key Findings

- Remote work and professional experience significantly impact salary.

- Predicting salary is difficult due to high variance and missing key variables (like geography, company size).

## Future Scope

- Integrate additional features such as country, job title, and technologies worked with.

- Apply advanced models (e.g., XGBoost, SHAP) for improved interpretability.

- Extend the analysis across multiple survey years to track trends over time.

## Conclusion

This project demonstrates how publicly available developer survey data can be used to derive valuable insights and build predictive models. Although salary prediction remains complex, the insights gathered here can inform hiring practices, career planning, and workforce analytics.


