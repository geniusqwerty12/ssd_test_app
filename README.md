# tflite_model_tester
 This simple python app lets you test tflite models on your computer, before deploying them to flutter apps

# Running the app

Jupyter Notebook
You can either copy the codes, open the source code on Jupyter and run the application.

VSCode
Before running, you must setup the Python environment for the application.


Note: If you have a newer version onf Python (3.11 above), you need to either downgrade to 3.11 due to compatability issues with tensorflow.

You can install multiple versions of Python (3.9, 3.10, 3.11 and make sure they are added on the environment path)

Setup virtualenv to use a specific version of Python
1. pip install virtualenv
2. virtualenv venv --python=python3.11
3. venv\Scripts\activate
4. pip install -r requirements.txt 

venv refers to the folder created

If Python version 3.11 is installed in your computer
Run the following commands on the terminal with VSCode
1. python -m venv env
2. env\Scripts\activate
3. pip install -r requirements.txt

Run step 2 whenever you are opening/reopening VSCode

Once the environment and modules are setup, you can run the app using the command: python main.py

# MODEL directory
The model and labels are stored inside the model folder. For testing your models, paste it there and update the following lines of code in main.py

* interpreter = Interpreter(model_path="model/YOUR MODEL NAME.tflite") line 8 for the model
* with open("model/YOUR LABEL FILE NAME.txt", 'r') as f: line 27 for the label