# Campus Placement Predictor + Skill Gap Analyzer

This project predicts campus placement probability from academic and profile features and explains which factors are holding a candidate back relative to the placed cohort.

## Run locally

1. Create and activate a virtual environment.
2. Install requirements with `pip install -r requirements.txt`.
3. Place the Kaggle dataset at `data/Placement_Data_Full_Class.csv`.
4. Run `python notebooks/train_model.py`.
5. Start the app with `streamlit run app/app.py`.
