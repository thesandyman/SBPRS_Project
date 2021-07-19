from flask import Flask, jsonify,  request, render_template
import model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        user = request.form['Username']
        output = model.preprocess(user)
        return render_template('index.html', prediction_text=output)
    else:
        return render_template('index.html')

# @app.route("/predict_api", methods=['POST', 'GET'])
#def predict_api():
#    print(" request.method :", request.method)
#    if (request.method == 'POST'):
#        data = request.get_json()
#        return jsonify(model.preprocess(user).tolist())
#    else:
#        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)