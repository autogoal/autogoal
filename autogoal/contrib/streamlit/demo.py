import sys
import streamlit as st


class Demo:
    def __init__(self):
        self.main_sections = {
            "Intro": self.intro,
        }

    def intro(self):
        st.write("# AutoGOAL Demos")

        st.write(
            """
            Welcome to the AutoGOAL Demos. In the left sidebar
            you will find all the available demos and additional
            controls specific to each of them.
            """
        )

        st.write(
            """
            This is the introductory demo. Here we will show the basic usage
            of AutoGOAL.
            """
        )

        st.write("## Basic usage")
        st.write(
            """
            AutoGOAL is first and foremost a framework for Automatic Machine Learning.
            With a few simple lines of code, you can quickly find a close to optimal
            solution for classic machine learning problems.
            """
        )

        st.write(
            """
            Let's start by (down)loading the classic [Cars dataset for the UCI repository](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation).
            This is a low-dimensionality supervised problem with 21 one-hot encoded features.
            """
        )

        with st.echo():
            from autogoal.datasets import cars
            X, y = cars.load()

        if st.checkbox("Preview data"):
            head = st.slider("Preview N first rows", 0, len(X), 5)
            st.show(X[:head, :])
            st.show(y[:head])

        st.write("The next step is to instantiate an AutoML solver and run it on this problem.")

        with st.echo():
            from autogoal.contrib.streamlit import StreamlitLogger
            from autogoal.ml import AutoML

            automl = AutoML(errors='ignore')

        st.write("And run!")

        st.code("automl.fit(X, y, logger=StreamlitLogger())", language="Python")

        if st.button("Run it!"):
            automl.fit(X, y, logger=StreamlitLogger())


    def run(self):
        main_section = st.sidebar.selectbox("Demo", list(self.main_sections))
        self.main_sections[main_section]()


demo = Demo()
demo.run()
