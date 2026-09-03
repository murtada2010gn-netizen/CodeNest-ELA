import json
from pathlib import Path
import uuid
import io
from flask import Flask, render_template, request, send_file, session, make_response
import numpy as np
import ollama
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

app = Flask(
    __name__, template_folder="../frontend", static_folder="../frontend"
)

models = {}
app.secret_key = "1903876473eeel222344pp"


@app.route("/")
def index():
  return render_template("ELA.html")


@app.route("/train", methods=["POST"])
def receive_settings():
  task_id = uuid.uuid4().hex

  uploaded_file = request.files["file"]

  settings_text = request.form.get("settings")
  settings = json.loads(settings_text) if settings_text else {}


  parms = {
      "model__max_depth": list(np.arange(3, 12, 3)),
      "model__n_estimators": list(np.arange(50, 201, 50)),
  }

  raw_tts = settings.get("TTS", 20)

  if raw_tts <= 0:
    raw_tts = 10
  elif raw_tts >= 100:
    raw_tts = 90

  test_size = raw_tts / 100.0

  selected_feature = settings.get("fe", [])
  selected_target = settings.get("ta")

  data = pd.read_csv(uploaded_file)

  data = data.dropna(subset=[selected_target])

  x = data[selected_feature]
  y = data[selected_target].values

  x_train, x_test, y_train, y_test = train_test_split(
      x, y, test_size=test_size, random_state=42
  )

  processor = cleaner(x_data=x_train)


  if settings["model"] == "rfr":

    pip = Pipeline([
      ("processor", processor),
      ("model", RandomForestRegressor(random_state=42))
    ])

    model = GridSearchCV(
        estimator=pip,
        param_grid=parms,
        scoring="r2",
        cv=5,
        n_jobs=-1
    )

    model.fit(x_train, y_train)

    pred = model.predict(x_test)

    mae, r2 = m1(y_test, pred)

    response = {
        "status": "success",
        "mae": float(mae),
        "r2": float(r2)
    }


  elif settings["model"] == "rfc":

    pip = Pipeline([
        ("processor", processor),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced"
            ),
        )

    ])

    model = GridSearchCV(
        estimator=pip,
        param_grid=parms,
        scoring="recall",
        cv=5,
        n_jobs=-1
    )

    model.fit(x_train, y_train)

    pred = model.predict(x_test)

    f1, accuracy, confusion_matrx = m2(y_test, pred)

    response = {
        "status": "success",
        "f1": float(f1),
        "accuracy": float(accuracy),
        "confusion_matrix": confusion_matrx.tolist(),
    }


  summuray = {
      "len_df": len(data),
      "columns": list(data.columns),
      "discribtion": data.describe().to_dict(),
      "sample": data.head(3).to_dict(orient="records"),
  }


  report = generate_report(
      settings=settings,
      metrics=response,
      summary=summuray
  )


  Path("settings.json").unlink(missing_ok=True)


  models[task_id] = {
      "model": model,
      "alldata": {
          "setting": settings,
          "test_size": test_size,
          "metrics": response,
          "aireport": report,
      }
  }


  response["task_id"] = task_id

  return response


@app.route("/download_report", methods=["GET"])
def download_report():

  return render_template("download_report.html")


@app.route("/download_report/<task_id>", methods=["GET"])
def download_report_file(task_id):

  data = models.get(task_id)

  if not data:
    return "No trained model found.", 404

  alldata = data["alldata"]

  settings = alldata.get("setting", {})
  metrics = alldata.get("metrics", {})
  report = alldata.get("aireport", "No AI report available.")
  test_size = alldata.get("test_size", 0.2)


  PdfName = f"{uuid.uuid4()}.pdf"

  doc = SimpleDocTemplate(
      PdfName,
      pagesize=A4,
      rightMargin=30,
      leftMargin=30,
      topMargin=30,
      bottomMargin=30,
  )

  story = []

  style = getSampleStyleSheet()

  Title = style["Title"]
  Heading2 = style["Heading2"]
  Normal = style["Normal"]


  story.append(Paragraph("CodeNest - ELA", Title))

  story.append(Spacer(1, 15))

  story.append(
      Paragraph(
          "Machine Learning Model Analysis Report – Created by ELA Platform",
          Heading2,
      )
  )

  story.append(Spacer(1, 15))


  if settings.get("model") == "rfr":

    tab = [
        ["Metric / Setting", "Value"],
        ["Model Type", "Random Forest Regressor"],
        ["Train Test Split", f"{int(test_size * 100)}%"],
        ["Mean Absolute Error (MAE)", f"{metrics.get('mae', 0):.4f}"],
        ["R2 Score", f"{metrics.get('r2', 0):.4f}"],
    ]

  else:

    tab = [
        ["Metric / Setting", "Value"],
        ["Model Type", "Random Forest Classifier"],
        ["Train Test Split", f"{int(test_size * 100)}%"],
        ["Accuracy", f"{metrics.get('accuracy', 0):.2%}"],
        ["F1 Score", f"{metrics.get('f1', 0):.2%}"],
    ]


  metrics_table = Table(
      tab,
      colWidths=[200, 220]
  )

  metrics_table.setStyle(
      TableStyle([
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
          ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
          ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
          ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
          ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
          ("PADDING", (0, 0), (-1, -1), 6),
      ])
  )


  story.append(metrics_table)

  story.append(Spacer(1, 25))

  story.append(
      Paragraph(
          "AI Audit Report:",
          style["Heading3"]
      )
  )

  story.append(Spacer(1, 10))


  for line in report.split("\n"):

    if line.strip():

      story.append(
          Paragraph(line, Normal)
      )

      story.append(
          Spacer(1, 6)
      )


  doc.build(story)

  response = send_file(
      PdfName,
      as_attachment=True
  )


  @response.call_on_close
  def delete_pdf():

    Path(PdfName).unlink(
        missing_ok=True
    )


  return response


@app.route("/download_model/<task_id>")
def download_model(task_id):
  data = models.get(task_id)
  if not data:
    return "No trained model found.", 404
  model = data["model"]


  buffer = io.BytesIO()

  pickle.dump(
      model,
      buffer
  )

  buffer.seek(0)


  return send_file(
      buffer,
      as_attachment=True,
      download_name=f"model_{task_id}.pkl",
      mimetype="application/octet-stream"
  )


def m1(yt, yp):

  mae = mean_absolute_error(
      y_true=yt,
      y_pred=yp
  )

  r2 = r2_score(
      y_true=yt,
      y_pred=yp
  )

  return mae, r2


def m2(yt, yp):

  f1 = f1_score(
      y_true=yt,
      y_pred=yp,
      average="weighted"
  )

  accuracy = accuracy_score(
      y_true=yt,
      y_pred=yp
  )

  confusion_matrx = confusion_matrix(
      y_true=yt,
      y_pred=yp
  )

  return f1, accuracy, confusion_matrx


def generate_report(settings, metrics, summary):

  system_prompt = """
  ROLE:
  You are ELA's Machine Learning Audit Engine.
  You are NOT a chatbot, assistant, tutor, or conversational agent.

  Your only task is to generate a machine learning audit report from the provided experiment data.

  INPUT:
  You will receive:
  1. Model configuration
  2. Evaluation metrics
  3. Confusion matrix when applicable
  4. Dataset summary

  ANALYSIS TASKS:

  1. PERFORMANCE ANALYSIS
  Analyze the actual model performance using only the supplied metrics and confusion matrix.

  For classification, analyze:
  Accuracy, F1 Score, confusion matrix, class prediction behavior, and whether the performance appears strong, moderate, or weak.

  For regression, analyze:
  R2 Score and MAE and whether the performance appears strong, moderate, or weak.

  2. CONFIGURATION AUDIT
  Detect actual problems in the experiment configuration.

  Check for:
  Target/model mismatch.
  Possible data leakage.
  Identifiers or obviously irrelevant columns.
  Invalid train/test split.
  Metric and confusion-matrix inconsistencies.
  Other clear configuration errors that can be determined from the supplied information.

  Do not invent problems.
  If no clear configuration problem exists, explicitly state that no clear configuration error was detected.

  3. DATASET AUDIT
  Analyze only observable properties contained in the supplied dataset summary.

  Do not assume information that is not provided.

  STRICT OUTPUT BEHAVIOR:

  This is an automated audit engine.

  DO NOT ask questions.
  DO NOT answer questions.
  DO NOT offer help.
  DO NOT give recommendations.
  DO NOT give suggestions.
  DO NOT tell the user what they should do.
  DO NOT propose feature engineering.
  DO NOT propose hyperparameter tuning.
  DO NOT propose resampling.
  DO NOT propose different models.
  DO NOT start a conversation.
  DO NOT address the user directly.
  DO NOT use conversational phrases such as:
  "Let me know"
  "You can"
  "You should"
  "I recommend"
  "If you want"
  "Would you like"

  The report must be self-contained and final.

  STRICT FORMAT:

  Output PLAIN TEXT ONLY.

  Markdown is completely forbidden.

  Do not use:
  #
  *
  **
  _
  ~
  -
  bullet lists
  numbered lists
  tables
  Markdown headings
  Markdown emphasis
  Markdown code blocks
  LaTeX
  HTML

  Use CAPITALIZED SECTION TITLES followed by normal paragraphs.

  Use this exact structure:

  PERFORMANCE ANALYSIS

  [analysis]

  CONFIGURATION AUDIT

  [analysis]

  DATASET AUDIT

  [analysis]

  FINAL ASSESSMENT

  [final assessment]

  Do not add any other sections.

  Do not include questions anywhere in the output.
  Do not include recommendations anywhere in the output.
  Do not use Markdown formatting anywhere in the output.
  """

  user_prompt = f"Settings: {settings}\nMetrics: {metrics}\nSummary: {summary}"


  response = ollama.chat(
      model="qwen3:8b",
      messages=[
          {
              "role": "system",
              "content": system_prompt
          },
          {
              "role": "user",
              "content": user_prompt
          },
      ],
      options={
          "temperature": 0.1
      },
  )


  report = response["message"]["content"]

  report = report.replace("**", "")
  report = report.replace("*", "")
  report = report.replace("#", "")
  report = report.replace("`", "")
  report = report.replace("~", "")

  return report


def cleaner(x_data):

  binary_column = []
  multi_int_cat_col = []
  int_column = []
  float_column = []
  cat_colums = []
  binary_str_column = []


  for i in x_data.columns:

    is_text = pd.api.types.is_object_dtype(
        x_data[i]
    ) or pd.api.types.is_string_dtype(
        x_data[i]
    )


    if pd.api.types.is_numeric_dtype(x_data[i]):

      diffs = x_data[i].diff().dropna()

      if (
          pd.api.types.is_numeric_dtype(x_data[i])
          and
          (x_data[i].nunique() / len(x_data[i])) > 0.95
      ):
        continue


    if set(x_data[i].dropna().unique()) <= {0, 1}:

      binary_column.append(i)

    elif pd.api.types.is_integer_dtype(x_data[i]):

      if x_data[i].nunique() <= 15:

        multi_int_cat_col.append(i)

      else:

        int_column.append(i)


    elif is_text and len(x_data[i].dropna().unique()) <= 2:

      binary_str_column.append(i)


    elif pd.api.types.is_float_dtype(x_data[i]):

      float_column.append(i)


    elif is_text and len(x_data[i].dropna().unique()) <= 15:

      cat_colums.append(i)


  numeric_cols = int_column + float_column

  categorical_cols = cat_colums + multi_int_cat_col


  numpip = Pipeline([
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler()),
  ])


  bpip = Pipeline([
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("scaler", StandardScaler()),
  ])


  catpip = Pipeline([
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("encoder", OneHotEncoder(
          handle_unknown="ignore",
          sparse_output=False
      )),
  ])


  bin_strpip = Pipeline([
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("encoding", OneHotEncoder(
          handle_unknown="ignore",
          sparse_output=False
      )),
  ])


  processor = ColumnTransformer(
      transformers=[
          ("num", numpip, numeric_cols),
          ("bin", bpip, binary_column),
          ("cat", catpip, categorical_cols),
          ("binstr", bin_strpip, binary_str_column),
      ]
  )


  return processor


if __name__ == "__main__":
  app.run(debug=True)