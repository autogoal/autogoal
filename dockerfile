FROM python:3.6

ADD . /autogoal

WORKDIR /autogoal

RUN make install
