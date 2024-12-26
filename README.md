# Case Study Submission

This repository contains my submission for the Customer Service Insight Analyst case study. Below is a detailed overview of the steps and tools I used to conduct the analysis, along with the purpose of each script and notebook.

## Overview of the Workflow

### 1. Data Preparation
- **`build_db.py`**:
  - Generates a local SQLite database from the two provided Parquet files, performing all the necessary initial data transformations (e.g., converting the base36 column).
  - SQLite was chosen for this step to leverage its built-in data quality features (e.g., foreign key checks) and facilitate my quick data exploration from the console before starting Python coding.

- **`build_ml_data.py`**:
  - Prepares two Parquet files with dataframes optimized for training machine learning models.
  - Steps include:
    - Handling timestamps (e.g., extracting day of the week and time slot information, which manual exploration suggested to be informative).
    - Preparing categorical variables for one-hot encoding and scaling.
  - While these prepared dataframes were not fully utilized in this submission due to time constraints, they provide a strong foundation for future machine learning work.

### 2. Exploratory Data Analysis (EDA)
- **Jupyter Notebooks**:
  - The notebooks (`eda1.ipynb`, `eda2.ipynb`, `eda3.ipynb`, `eda4.ipynb`) and auxiliary modules (`analysis.py`, `plots.py`, and `common.py`) were used to perform detailed exploratory data analysis. The code is modular and organized to be production-ready if interesting findings need further development.
  - **Notebook Breakdown**:
    - **`eda1.ipynb`**: Focuses on initial data exploration, including the study of individual columns and their values.
    - **`eda2.ipynb`**: Analyzes features related to the number of errands and the presence/absence of errands in customer orders.
    - **`eda3.ipynb`**: Explores initial machine learning models, primarily for identifying feature importance.
    - **`eda4.ipynb`**: Conducts clustering and pattern identification to uncover underlying trends in customer service interactions.
  - Each notebook contains detailed comments and notes made during development, which served as the basis for the slides submitted as part of this case study.

### 3. Presentation Preparation
- **`presentation.ipynb`**:
  - Used to prepare the data, tables, and plots for the final slides in the same order they appear in the presentation.
  - Ensures consistency between the analysis and visualizations shared in the submission.

## Summary
This submission demonstrates a systematic approach to analyzing customer service data, including:
- Ensuring data integrity and quality through SQLite integration.
- Preparing machine learning-ready datasets for future predictive and prescriptive analytics.
- Conducting exploratory data analysis to uncover actionable insights.
- Developing visualizations and summaries tailored for mixed technical and non-technical audiences.
