import random
from predict import predict

from flask import Flask
from flask import request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/pd', methods=['GET'])
def pd():
    smiles = request.args.get('sm', default='', type=str)
    ic, cc, si = predict(smiles)
    obj = {'ic': ic, 'cc': cc, 'si': si}
    return obj


if __name__ == '__main__':
    app.run(ssl_context=('certificate.crt', 'private.key'))