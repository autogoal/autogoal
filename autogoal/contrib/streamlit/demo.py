import sys
import streamlit as st


class Demo:
    def __init__(self):
        self.main_sections = {
            "Intro": self.intro,
            "Automatic pipelines": self.build_pipelines,
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

        st.write(
            "The next step is to instantiate an AutoML solver and run it on this problem."
        )

        iterations = st.number_input("Number of iterations", 1, 100, 10)

        with st.echo():
            from autogoal.contrib.streamlit import StreamlitLogger
            from autogoal.ml import AutoML

            automl = AutoML(
                random_state=0,  # fixed seed fo reproducibility
                errors="ignore",  # ignore exceptions
                search_iterations=iterations,  # total iterations
                search_kwargs=dict(search_timeout=5),  # max time per pipeline
            )

        st.write("And run!")

        st.code("automl.fit(X, y, logger=StreamlitLogger())", language="Python")

        if st.button("Run it!"):
            automl.fit(X, y, logger=StreamlitLogger())

        st.write(
            """
            ## Next steps

            Take a look at the remaining examples in the sidebar.
            """
        )

    def build_pipelines(self):
        st.write("# Building automatic pipelines")

        st.write(
            "This example illustrate how AutoGOAL automatically builds "
            "a graph of pipelines for different problems settings."
        )

        with st.echo():
            from autogoal.kb import CategoricalVector, Document, Tuple
            from autogoal.kb import build_pipelines
            from autogoal.contrib import find_classes

            space = build_pipelines(
                input=Tuple(Document(), CategoricalVector()),
                output=CategoricalVector(),
                registry=find_classes(),
            )

        if st.button("Sample"):
            st.code(space.sample())

    def run(self):
        main_section = st.sidebar.selectbox("Demo", list(self.main_sections))
        self.main_sections[main_section]()


demo = Demo()
demo.run()
