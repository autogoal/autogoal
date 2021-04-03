import sys
import streamlit as st
import textwrap
import networkx as nx
import pandas as pd
import altair as alt
import nx_altair as nxa
import inspect

from autogoal.kb import build_pipeline_graph


@st.cache(allow_output_mutation=True)
def eval_code(code, *variables):
    locals_dict = {}
    exec(code, globals(), locals_dict)

    if len(variables) == 0:
        return None
    if len(variables) == 1:
        return locals_dict[variables[0]]
    else:
        return [locals_dict[var] for var in variables]


class Demo:
    def __init__(self):
        self.main_sections = {
            "Intro": self.intro,
            "High-Level API": self.high_level,
            "Pipelines": self.build_pipelines,
        }

    def intro(self):
        st.write("# AutoGOAL Demos")

        st.write(
            """
            Welcome to the AutoGOAL Demo. In the left sidebar
            you will find all the available demos and additional
            controls specific to each of them.

            AutoGOAL is a framework in Python for automatically finding the best way to solve a given task.
            It has been designed mainly for automatic machine learning~(AutoML)
            but it can be used in any scenario where several possible strategies are available 
            to solve a given computational task.
            """
        )

        st.write(
            """
        ## About this demo

        The purpose of this demo application is to showcase the main use cases of AutoGOAL.
        Keep in mind that AutoGOAL is a software library, i.e., meant to be used from source code.
        This demo serves as an interactive and user-friendly introduction to the library, but it
        is in no case a full-featured AutoML application.

        There are two sections to showcase different components of AutoGOAL.
        You can switch sections in the left sidebar.
        
        * The **High-Level API** section presents the public interface of AutoGOAL
        in several datasets.
        * The **Pipelines** section shows the internal components of AutoGOAL and allows
        to explore the possible pipelines.
        """
        )

        st.write(
            """
        ## Running the code

        To execute this demo on your own infrastructure, you need AutoGOAL's docker image.
        There are two images available, without and without GPU support.

        Download the corresponding Docker image:

            docker pull autogoal/autogoal

        """
        )

        st.write(
            """
            Launch a Docker container.

                docker run --rm -p 8501:8501 autogoal/autogoal

            Navigate to <http://localhost:8501>.
            """
        )

    def high_level(self):
        st.write("# High-Level API")
        st.write(
            """
            AutoGOAL is first and foremost a framework for Automatic Machine Learning.
            With a few simple lines of code, you can quickly find a close to optimal
            solution for classic machine learning problems.
            """
        )

        from autogoal import datasets

        dataset_descriptions = {
            "cars": """
                [Cars](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
                is a low-dimensionality supervised problem with 21 one-hot encoded features.
                """,
            "german_credit": """
                [German Credit](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29)
                is a low-dimensionality supervised problem with 20 categorical or numerical features.
                """,
            "abalone": """
                [Abalone](https://archive.ics.uci.edu/ml/datasets/Abalone)
                is a low-dimensionality supervised problem with 8 categorical or numerical features.
                """,
            "shuttle": """
                [Shuttle](https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle))
                is a low-dimensionality supervised problem with 9 numerical features.
                """,
            "yeast": """
                [Yeast](https://archive.ics.uci.edu/ml/datasets/Yeast)
                is a low-dimensionality supervised problem with 9 numerical features.
                """,
            "dorothea": """
                [Dorothea](https://archive.ics.uci.edu/ml/datasets/dorothea)
                is a high-dimensionality sparse supervised problem with 100,000 numerical features.
                """,
            "gisette": """
                [Gisette](https://archive.ics.uci.edu/ml/datasets/Gisette)
                is a high-dimensionality sparse supervised problem with 5,000 numerical features.
                """,
            "haha": """
                [HAHA 2019](https://www.fing.edu.uy/inco/grupos/pln/haha/index.html#data) is
                a text classification problem with binary classes in Spanish.
                """,
            "meddocan": """
                [MEDDOCAN 2019](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN) is
                an entity recognition problem in Spanish medical documents.
                """,
        }

        override_types = {
            "german_credit": ("MatrixContinuousDense()", "CategoricalVector()"),
            "dorothea": ("MatrixContinuousSparse()", "CategoricalVector()"),
            "gisette": ("MatrixContinuousSparse()", "CategoricalVector()"),
            "haha": ("List(Sentence())", "CategoricalVector()"),
            "meddocan": ("List(List(Word()))", "List(List(Postag()))"),
        }

        st.write(
            """Let's start by selecting one of the example datasets.
            These are sample datasets which are automatically downloaded by AutoGOAL,
            and can be used to benchmark new algorithms and showcase AutoML tools.
            """
        )

        dataset = st.selectbox("Select a dataset", list(dataset_descriptions))

        st.write(
            dataset_descriptions[dataset] + "Here is the code to load this dataset."
        )

        code = textwrap.dedent(
            f"""
            from autogoal.datasets import {dataset}

            X, y, *_ = {dataset}.load()
        """
        )
        st.code(code)

        X, y = eval_code(code, "X", "y")

        if st.checkbox("Preview data"):
            try:
                l = len(X)
            except:
                l = X.shape[0]

            head = st.slider("Preview N first items", 0, l, 5)
            if isinstance(X, list):
                st.write(X[:head])
            else:
                st.write(X[:head, :])

            st.write(y[:head])

        st.write(
            """
            The next step is to instantiate an AutoML solver and run it on this problem.
            The `AutoML` class provides a black-box interface to AutoGOAL.
            You can tweak the most important parameters at the left sidebar, even though
            sensible defaults are provided for all the parameters.
            """
        )

        st.sidebar.markdown("### AutoML parameters")
        iterations = st.sidebar.number_input("Number of iterations", 1, 10000, 100)
        global_timeout = st.sidebar.number_input(
            "Global timeout (seconds)", 1, 1000, 60
        )
        pipeline_timeout = st.sidebar.number_input(
            "Timeout per pipeline (seconds)", 1, 1000, 5
        )

        from autogoal.contrib.streamlit import StreamlitLogger

        if dataset in override_types:
            input_type, output_type = override_types[dataset]
            types_code = f"""
                input={input_type},
                output={output_type},
            """

            st.info(
                f"""
            In most cases AutoGOAL can automatically infer the input and output type
            from the dataset. Sometimes, such as with `{dataset}`, the user will need to provide them
            explicitely.
            """
            )
        else:
            types_code = ""

        code = textwrap.dedent(
            f"""
            from autogoal.kb import *
            from autogoal.ml import AutoML

            automl = AutoML(
                errors="ignore",  # ignore exceptions (e.g., timeouts)
                search_iterations={iterations}, # total iterations
                search_kwargs=dict(
                    search_timeout={global_timeout}, # max time in total (approximate)
                    evaluation_timeout={pipeline_timeout}, # max time per pipeline (approximate)
                ), {types_code}
            )
            """
        )

        st.code(code)
        automl = eval_code(code, "automl")

        st.write(
            """
            Click run to call the `fit` method. Keep in mind that many of these pipelines can be 
            quite computationally heavy and both the hyperparameter configuration as well as the
            infrastructure where this demo is running might not allow for the best pipelines to execute.
            """
        )

        st.code("automl.fit(X, y)", language="Python")

        if st.button("Run it!"):
            automl.fit(X, y, logger=StreamlitLogger())

        st.write(
            """
            ## Next steps

            Take a look at the remaining examples in the sidebar.
            """
        )

    def build_pipelines(self):
        st.write("# Pipelines")

        st.write(
            "This example illustrates how AutoGOAL automatically builds "
            "a graph of pipelines for different problems settings."
        )

        from autogoal.kb._data import DATA_TYPES

        types_str = [cls.__name__ for cls in DATA_TYPES]

        st.write(
            """
            AutoGOAL pipeline discovery is based on a hierarchy of semantic datatypes.
            Each type represents a semantic datum that can be used in a machine learning algorithm,
            from matrices and vectors to sentences, entities and and images.

            The following picture shows all available semantic data types.
            You can click the top right corner to enlarge.
            """
        )

        st.image("/code/docs/guide/datatypes.png", use_column_width=True)

        from autogoal.contrib import find_classes

        all_classes = {k.__name__: k for k in find_classes()}

        st.write(
            f"""
            ## Algorithm Library

            AutoGOAL automatically builds pipelines by selecting from a wide range of algorithms
            implemented in `contrib` modules.
            The list of all available algorithms is shown here.

            There are a total of **{len(all_classes)}** algorithms implemented.
            Select one to display some information.
            """
        )

        class_name = st.selectbox("Select an algorithm", list(all_classes))
        class_type = all_classes[class_name]

        st.write(f"### {class_type.__module__}.{class_name}")

        run_signature = inspect.signature(class_type.run)
        st.write(f"**Input type**: {run_signature.parameters['input'].annotation}")
        st.write(f"**Output type**: {run_signature.return_annotation}")

        st.write("#### Parameters")
        params = []
        for name, param in inspect.signature(class_type.__init__).parameters.items():
            if name == "self":
                continue

            params.append(f"* **{name}**: {param.annotation}")
        st.write("\n".join(params))

        st.write("## Pipeline Builder")

        st.write(
            """
            AutoGOAL can automatically build pipelines given a desired input and output
            value. It uses the annotations of the `run` method of each algorithm to detect
            which algorithms can be connected.

            In the following section, you can select a desired input and output types and 
            explore the pipelines that AutoGOAL discovers.
            In the left sidebar you can fine-tune the input value, e.g., make it a list
            of elements instead of a single element.
            """
        )

        st.sidebar.markdown("### Configure input and output types")
        list_input = st.sidebar.number_input("Input list (level)", 0, 3, 1)
        list_output = st.sidebar.number_input("Output list (level)", 0, 3, 0)
        tuples = st.sidebar.checkbox("Is supervised (use Tuple in input)", True)

        input_type = st.selectbox(
            "Select an input type", types_str, types_str.index("Sentence")
        )

        output_type = st.selectbox(
            "Select and output type", types_str, types_str.index("CategoricalVector")
        )

        input_type = input_type + "()"
        for i in range(list_input):
            input_type = f"List({input_type})"

        output_type = output_type + "()"
        for i in range(list_output):
            input_type = f"List({output_type})"

        if tuples:
            input_type = f"Tuple({input_type}, {output_type})"

        st.write(f"#### Defined input type:  `{input_type}`")
        st.write(f"#### Defined output type: `{output_type}`")

        st.write(
            """
            The following code uses explicitely AutoGOAL's pipeline discovery engine
            to find all the pipelines that can be constructed from the desired
            input to the desired output.
            """
        )

        code = textwrap.dedent(
            f"""
            from autogoal.kb import *
            from autogoal.kb import build_pipelines
            from autogoal.contrib import find_classes

            # explicitly build the graph of pipelines
            space = build_pipelines(
                input={input_type},
                output={output_type},
                registry=find_classes(),
            )
            """
        )

        st.code(code)

        try:
            space = eval_code(code, "space")
        except Exception as e:
            if "No pipelines can be constructed" in str(e):
                st.error(str(e))
                st.info(
                    "Try changing the input and output type or select **Is supervised** in the left sidebar."
                )
                return

            raise

        st.write(
            """
            ### The Pipelines Graph
            
            This is the graph that represents all the posible pipelines find by AutoGOAL.
            Each node in this graph is an algorithm from the _Algorithm Library_ that is
            compatible with the input and output types of its neighbors.
            Any path from the top to the bottom of the graph represents a valid pipeline.
            """
        )

        graph = nx.DiGraph()

        def get_node_repr(node):
            try:
                return get_node_repr(node.inner)
            except:
                return dict(
                    label=str(node).split(".")[-1], module=node.__module__.split("_")[0]
                )

        for node in space.graph.nodes:
            attrs = get_node_repr(node)
            graph.add_node(attrs["label"], **attrs)

        for u, v in space.graph.edges:
            graph.add_edge(get_node_repr(u)["label"], get_node_repr(v)["label"])

        pos = nx.nx_pydot.pydot_layout(graph, prog="dot", root=space.Start)
        chart = (
            nxa.draw_networkx(graph, pos=pos, node_color="module", node_tooltip="label")
            .properties(height=500)
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)

        st.write(
            """
            ### Example Pipeline
            
            Here is an example pipeline that has been randomly sampled from the previous graph.
            You can try different samples. Notice how not only the nodes (algorithms) that participate
            in the pipeline are different each time, but also their internal hyperparameters change.
            
            When sampling a pipeline from the graph AutoGOAL samples all the internal
            hyperparameters as defined by the constructor.
            When these hyperparameters have complex values (e.g., an algorithm per-se), AutoGOAL
            recursively samples instances of the internal algorithms, and so on.
            """
        )

        st.code(space.sample())

        st.button("Sample another pipeline")

    def run(self):
        main_section = st.sidebar.selectbox("Section", list(self.main_sections))
        self.main_sections[main_section]()


demo = Demo()
demo.run()
