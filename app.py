from flask import Flask, request, render_template
from predict import predict_flood, predict_earthquake

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_flood', methods=['POST'])
def flood_route():
    try:
        data = {
            'rainfall': float(request.form['rainfall']),
            'river_level': float(request.form['river_level']),
            'humidity': float(request.form['humidity']),
            'temperature': float(request.form['temperature'])
        }
        result = predict_flood(data)
    except Exception as e:
        result = f"Error: {str(e)}"
    return render_template('index.html', flood_result=result)


@app.route('/predict_earthquake', methods=['POST'])
def earthquake_route():
    try:
        data = {
            'magnitude': float(request.form['magnitude']),
            'depth': float(request.form['depth']),
            'ground_acceleration': float(request.form['ground_acceleration'])
        }
        result = predict_earthquake(data)
    except Exception as e:
        result = f"Error: {str(e)}"
    return render_template('index.html', earthquake_result=result)


if __name__ == '__main__':
    app.run(debug=True)
