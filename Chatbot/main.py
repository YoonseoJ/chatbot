from flask import Flask, render_template, redirect, url_for, request
from static.script.chatbot import *

app = Flask(__name__)

history = []

@app.route("/")
def home():
    result = request.args.get('result')
    input = request.args.get('input')
    myhistory = request.args.get('history')
    print("888")
    print(history)
    myResult = ""
    if result is not None:
        myResult = result
    return render_template("home.html", result=myResult, input=input, history=history, len=len(history))

@app.route("/runscript")
def ScriptPage():
    myInput = request.args.get('input')
    print("=== " + myInput)
    message = myInput
    intents = pred_class(message, classes)
    result = get_response(intents, data)
    history.append(myInput)
    history.append(result)
    return redirect(url_for("home", result=result, input=myInput, history=history))


if __name__ == '__main__': 
    app.run(debug=True)