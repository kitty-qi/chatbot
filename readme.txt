Youtube Demo:
https://www.youtube.com/watch?v=B2kbY2a1Y3w

Create a virtual environment:

pip install virtualenv

# dsci560 is the virtual environment name
python3 -m venv dsci560

# choose from one of below
# on Mac: source dsci560/bin/activate
# on windows: .\dsci560\Scripts\activate

#install package_name
pip install -r requirements.txt


streamlit run KKA.py

# Deactivate the virtual environment
deactivate


The pre-trained Keras model for food image recognition is from https://github.com/MaharshSuryawala/Food-Image-Recognition 
Copyright (c) 2020 Maharsh Suryawala
