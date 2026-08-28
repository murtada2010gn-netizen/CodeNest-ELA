# ELA

**A CodeNest Project**

#### Video Demo: [https://youtu.be/lLtXW3cZu0U?si=UxWRM7EhqxCkLrSl](https://youtu.be/lLtXW3cZu0U?si=UxWRM7EhqxCkLrSl)

#### Project Date: August 28, 2026

## Description

ELA is a no-code web application built to make machine learning easier to use and easier to understand.

The idea started from a simple problem: training a machine learning model usually requires writing code, preparing the dataset, choosing the right features, selecting a model, tuning its parameters, and then interpreting the results. For someone who is learning AI and machine learning, these steps can quickly become confusing.

ELA tries to simplify that process through a graphical interface. The user uploads a dataset, chooses the target column, selects the features they want to use, chooses between classification and regression, and starts the training process without having to write machine learning code.

The application is designed with two goals in mind. The first is to make model training faster and more accessible. The second is to make the process educational by helping the user understand the data, the settings they selected, and the results they obtained.

## How It Works

The workflow starts when the user uploads a CSV dataset.

ELA reads the dataset and displays its available columns. The user can then choose the column they want the model to predict and select the features that should be used during training. This also gives the user a simple way to exclude columns that should not be used, such as identifiers or other irrelevant information.

After selecting the data, the user chooses the problem type:

* Classification
* Regression

The application then prepares the selected data and starts the training process.

The current version performs several preprocessing steps before training. Missing rows and duplicate rows are removed, and categorical features are converted into numerical values using one-hot encoding. The dataset is then divided into training and testing sets so that the final model can be evaluated on data that was not directly used for training.

## Model Training and Hyperparameter Search

One of the main parts of ELA is model optimization.

Instead of relying on a single manually chosen configuration, the application uses `GridSearchCV` to test multiple combinations of model hyperparameters. Cross-validation is used during this process to compare the different configurations and select a better-performing one.

The current implementation explores parameters such as:

* Number of estimators
* Maximum tree depth

This makes the training process more useful than simply creating one model with a fixed configuration.

## Results & Automated PDF Reporting

After training, ELA displays the results through the web interface and provides an automated PDF report generation system.

For classification problems, the current version reports:

* Accuracy
* F1 Score
* Confusion Matrix

For regression problems, it reports:

* Mean Absolute Error (MAE)
* R² Score

These metrics give the user a basic view of how well the trained model performs on the test data.

Additionally, users can download a complete, structured PDF engineering report summarizing the dataset parameters, selected features, model performance metrics, and automated AI model audit findings for archival or documentation purposes.

The purpose is not only to produce a score, but also to give the user information that can be used to evaluate the quality of the model and the choices made during the training process.

## Educational Goal

ELA is intended to be more than a model-training interface.

A beginner can easily get an accuracy value without understanding why the model produced that result. ELA is designed around the idea that training a model should also be a learning experience.

The project is intended to help users identify issues in their datasets and understand how different choices can affect a model. Examples include selecting irrelevant columns, using unsuitable features, having missing data, or choosing settings that may not produce good results.

This educational side is an important part of the project's purpose: the user should not only receive a result, but also have a clearer idea of what happened during the process.

## Project Structure

The project is divided into two main parts:

### Backend

The backend is built with Python and Flask. It handles:

* Receiving uploaded datasets
* Receiving user settings
* Reading and preparing the data
* Splitting the dataset
* Training the machine learning models
* Hyperparameter search
* Calculating evaluation metrics
* Generating downloadable PDF engineering reports
* Returning the results to the frontend

### Frontend

The frontend provides the graphical interface used by the user to:

* Upload a dataset
* Select the target column
* Select features
* Choose the problem type
* Start model training
* View the results and download PDF reports

Bootstrap is used to speed up interface development and keep the application clean and responsive, allowing more time to be spent on the machine learning workflow and the functionality of the application.

## Technologies Used

ELA was built using:

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* HTML
* CSS
* JavaScript
* Bootstrap

Scikit-learn is used for preprocessing, dataset splitting, model training, hyperparameter search, and evaluation.

## Design Decisions

One of the main design decisions was to keep the machine learning workflow behind a graphical interface. The user should be able to experiment with a dataset without first learning how to write the complete training pipeline manually.

Another important decision was to keep a separate testing set. The model is trained on the training data, while the test data is kept aside for the final evaluation. This gives a more realistic indication of how the model performs on data it has not seen during training.

Grid search and cross-validation were also chosen because manually guessing hyperparameters would make the process less useful for a no-code tool. By testing several combinations, the application can compare configurations instead of depending on a single fixed setting.

For categorical columns, one-hot encoding is used to convert non-numeric values into a format that machine learning algorithms can process.

## Academic Integrity & Development Attribution

This project strictly adheres to academic integrity, transparency, and software engineering ethics.

### Core Architecture & Backend Ownership (100% Original Work)
The primary technical focus and personal contribution in this project reside in the backend engineering and machine learning workflows. All pipeline logic, automated preprocessing (`ColumnTransformer`), feature handling, hyperparameter search (`GridSearchCV`), evaluation metrics calculation, automated PDF report generation engine, and backend API endpoints were independently architected and developed from scratch. 

My full-stack development capability was fully demonstrated in **Version 1 (V1)** of this project, where the functional end-to-end prototype—including both backend and initial frontend—was hand-crafted entirely from scratch.

### Frontend Refinement (Tool-Assisted Styling)
Following the completion of the functional V1 prototype, **Codex** was utilized as an auxiliary tool specifically for **UI layout refinement and visual CSS polishing**. This allowed optimization of engineering time, enabling deeper focus on building the backend dataset cleaning engine, model accuracy, PDF exporting, and core system functionality. 

All underlying system architecture, data flow logic, and Machine Learning algorithms remain 100% original work.

## Current Version

The current version of ELA is **v0.5**.

The first working version focuses on the core workflow:

**Dataset → Target → Features → Model Type → Training → Evaluation & Report Generation**

The project is intentionally being developed in stages so that new features can be added after the core training pipeline is stable.

## Future Development

Future versions of ELA are planned to expand the project beyond the current training workflow.

Possible additions include:

* Downloading the trained model for reuse in other projects
* Providing clearer explanations of errors and model performance
* Adding NLP-based analysis of datasets and training results
* Using an 8B language model for more detailed explanations
* Supporting additional machine learning algorithms
* Providing more visualizations and model comparisons

These features are part of the future direction of the project and are separate from the functionality implemented in the current v0.5 release.

## Why I Built ELA

I built ELA because I wanted to combine two things I am interested in: building software and learning artificial intelligence.

Machine learning can feel complicated when every experiment requires writing another training script. I wanted to build something that removes some of that repetitive work while still keeping the learning process visible to the user.

The project is also an experiment for me as a developer. Instead of building a simple application around a fixed dataset, I wanted to create something that can accept different datasets and allow the user to make the important decisions themselves.

## Acknowledgment

ELA is part of the **CodeNest** project series.

CodeNest is the shared name I use across my projects, while **ELA** is the name of this specific application.

© 2026 ELA — A CodeNest Project