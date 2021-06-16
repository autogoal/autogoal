# =====================
# GPU base image
# ---------------------

FROM autogoal/server

RUN sudo apt-get update 
RUN sudo apt-get install tesseract-ocr -y 
RUN sudo apt-get install tesseract-ocr-spa -y 
RUN sudo apt-get clean 
RUN sudo apt-get autoremove

ADD . /home/coder/autogoal

WORKDIR /home/coder/autogoal

RUN pip3 install pytesseract
