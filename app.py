import os
import io
import base64
import numpy as np
import pandas as pd

from flask import (
    Flask,
    request,
    jsonify,
    render_template
)

from PIL import Image

# =========================================================
# IMPORT PREDICTION FUNCTION
# =========================================================

from PREDICTION import predict

# =========================================================
# LOAD MODEL METRICS
# =========================================================

results_df = pd.read_csv("results.csv")

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

model_metrics = results_df.to_dict(orient="records")

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# HOME
# =========================================================

@app.route('/')
def index():

    return render_template(

        'index.html',

        model_metrics=model_metrics

    )

# =========================================================
# PREDICT ROUTE
# =========================================================

@app.route('/predict', methods=['POST'])
def predict_route():

    if 'files' not in request.files:

        return jsonify({
            "error": "No files uploaded"
        }), 400

    files = request.files.getlist('files')

    results = []

    # =====================================================
    # TEMP FOLDER
    # =====================================================

    os.makedirs("temp", exist_ok=True)

    # =====================================================
    # PROCESS MAX 3 FILES
    # =====================================================

    for file in files[:3]:

        try:

            # =================================================
            # SAVE TEMP FILE
            # =================================================

            temp_path = os.path.join(
                "temp",
                file.filename
            )

            file.save(temp_path)

            # =================================================
            # PREDICTION
            # =================================================

            result = predict(temp_path)

            image_data = result["image"]

            # =================================================
            # NORMALIZE IMAGE
            # =================================================

            img_normalized = (

                255 * (image_data - np.min(image_data))

                / (np.ptp(image_data) + 1e-6)

            ).astype(np.uint8)

            img = Image.fromarray(img_normalized)

            # =================================================
            # IMAGE TO BASE64
            # =================================================

            buffered = io.BytesIO()

            img.save(buffered, format="PNG")

            img_base64 = base64.b64encode(
                buffered.getvalue()
            ).decode("utf-8")

            # =================================================
            # STORE RESULT
            # =================================================

            results.append({

                "filename": file.filename,

                "actual": result["actual"],

                "predicted": result["prediction"],

                "image_data":
                    f"data:image/png;base64,{img_base64}",

                "probabilities": {

                    "Meningioma":
                        round(
                            result["probabilities"]["Meningioma"],
                            2
                        ),

                    "Glioma":
                        round(
                            result["probabilities"]["Glioma"],
                            2
                        ),

                    "Pituitary Tumor":
                        round(
                            result["probabilities"]["Pituitary Tumor"],
                            2
                        )
                }
            })

            # =================================================
            # DELETE TEMP FILE
            # =================================================

            if os.path.exists(temp_path):

                os.remove(temp_path)

        except Exception as e:

            results.append({

                "filename": file.filename,

                "error": str(e)

            })

    return jsonify(results)

# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':

    app.run(debug=True)