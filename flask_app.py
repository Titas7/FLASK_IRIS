from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the model named "iris"
with open("iris", "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/", methods=["GET", "POST"])
def classify():
    if request.method == "POST":
        # Get values from form and convert to float
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])
        
        # Prepare input for model
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Predict
        prediction = model.predict(features)[0]

        # Optional: Map prediction if model returns numeric label
        label_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
        predicted_label = label_map.get(prediction, prediction)

        return redirect(url_for("result", prediction=predicted_label))
    
    return render_template("pred.html")

@app.route("/result/<prediction>")
def result(prediction):
    return f"<h1>Predicted Iris Species: {prediction}</h1>"

if __name__ == '__main__':
        app.run(debug=True)
