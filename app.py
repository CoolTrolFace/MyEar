from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import base64
app = Flask(__name__)


#to display the connection status
@app.route('/', methods=['GET'])
def handle_call():
    return "Successfully Connected"


@app.route('/getdata', methods=['POST'])
def getdata():
    data = request.form
    print("Data "+data['data'])
    return "Data received"


if __name__ == '__main__':
   app.run()