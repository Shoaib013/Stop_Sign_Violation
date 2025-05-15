This project is aimed to detect stop signs and the text boxes below it and then takes decision whether there was stop sign violation happened or not, on the basis of video input provided and speed values provided in the form of jsons.
Following steps were followed for developing it:
  1. Annotated Stop Sign Images and the text boxes that are below stop signs and data was taken from kaggle and roboflow.
  2. Created four classes as stop_sign, except_right_turn,right_turn_only and arrow_right_turn_only.
  3. Trained it on YoloV8n.
  4. Used tessaract for OCR.
  5. Deployed using streamlit.
You can run this in two ways:
  1. First way is to run it using streamlit in which you would be using app.py file.
  2. The second way is to use batch_process.py file in which you have to provide path of videos and json files and then it will run and return a csv with results and videos as well which will have visualizations.
