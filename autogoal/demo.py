import sys

try:
    import streamlit as st
except ImportError:
    print("(!) Too run the demo you need streamlit installed.")
    print("(!) Fix it by running `pip install streamlit`.")
    sys.exit(1)


class Demo:
    def __init__(self):
        self.main_sections = {
            "Intro": self.intro,
        }

    def intro(self):
        st.write("# AutoGOAL Demos")

        st.write("""
        Welcome to AutoGOAL Demos. In the left sidebar
        you will find all the available demos and additional
        controls specific to each of them.
        """)

        st.write("""
        This is the introductory demo. Here we will show the basic usage
        of AutoGOAL.
        """)

        st.write("## Basic usage")
        st.write("AutoGOAL is first and foremost a framework for Automatic Machine Learning.")

        with st.echo():
            from autogoal.ml import AutoML

    def run(self):
        main_section = st.sidebar.selectbox("Demo", list(self.main_sections))
        self.main_sections[main_section]()


demo = Demo()
demo.run()
