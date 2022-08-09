CSTP 2301 machine learning project
# Chatbot for COVID19 FAQ
### ✍️ Authors
Son Minh Nguyen . Yooenseo Jeong . Yifei Chen . Seyedeh Sarina 

## :robot: How to install and run the project
- Navigate into Chatbot folder
- Create your virtual python environment by running following commands in terminal
```bash
python3 -m venv venv
source venv/bin/activate
```
- We need to install some python packages in order to run the app.
- You can install packages using the following command: (You must install pip first if you doesn't have it)
```bash
python -m pip install <package-name>
```
:warning: Be sure that you are inside the folder that contains the virtual environment(the Chatbot folder) before installing any packages

**Packages that we will need are:**

- numpy
- nltk
- tensorflow
- keras
- flask 

After installing the packages we need to config some variables for Flask
```bash
export FLASK_APP=main
export FLASK_ENV=development
```
You can then run the app by entering
```bash
flask run 
```
Open http://127.0.0.1:5000 on your browser to see the result.