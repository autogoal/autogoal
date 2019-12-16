FROM python:3.6

ADD * /gpto/

RUN make install
