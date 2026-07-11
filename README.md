# Campus Placement Predictor + Skill Gap Analyzer

This project predicts the likelihood of campus placement for a student using academic and profile-based features, and also provides a skill-gap analysis by comparing the user profile with the profile of successfully placed students.

## Project Overview

The system helps students understand:
- their placement probability
- which factors are affecting their chances
- which areas they should improve

It uses a machine learning model trained on campus placement data and a Streamlit-based web app for interaction.

## Features

- Placement probability prediction
- Skill gap analysis
- Comparison against placed-student averages
- Easy-to-use web interface
- Deployed using Streamlit

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Jupyter / Python scripts

## Project Structure

- `app/` - Streamlit application
- `src/` - Model training and prediction logic
- `notebooks/` - EDA and model training scripts
- `data/` - Dataset
- `model/` - Trained model artifacts
- `tests/` - Basic test cases

## How to Run Locally

1. Create and activate a virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
## Deployment
This project is designed to be deployed on Streamlit Cloud using the entry file:

streamlit_app.py
Dataset
The project uses a campus placement dataset with features like:

academic percentages
specialization
work experience
placement status
Future Improvements
Add more features and better model tuning
Improve the gap analysis with more detailed recommendations
Add visual charts and dashboards
