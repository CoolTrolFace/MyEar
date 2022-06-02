from pickle import load
import wave
import time
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
    json_data = request.json
    val = json_data["data"]
    msg = "0:none" #if the accuracy match rate is less than %90, we dont label the sound, it is indefinite
    return msg

if __name__ == '__main__':
   app.run()
