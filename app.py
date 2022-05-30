from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
app = Flask(__name__)


#to display the connection status
@app.route('/', methods=['GET'])
def handle_call():
    return "Successfully Connected"

#the get method. when we call this, it just return the text "Hey!! I'm the fact you got!!!"
@app.route('/getfact', methods=['GET'])
def get_fact():
    return "Hey!! I'm the fact you got!!!"

@app.route('/test', methods=['POST'])
def test():
    array = request.form.get('array')
    return "Test complete."


if __name__ == '__main__':
   app.run()