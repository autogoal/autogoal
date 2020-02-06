try:
    import streamlit as st
except ImportError:
    print("(!) Too run the demo you need streamlit installed.")
    print("(!) Fix it by running `pip install streamlit`.")
    sys.exit(1)


from autogoal.search import Logger


class StreamlitLogger(Logger):
    def __init__(self):
        self.evaluations = 0
        self.current = 0
        self.status = st.info("Waiting for evaluation start.")
        self.progress = st.progress(0)
        self.error_log = st.empty()
        self.best_fn = 0
        self.chart = st.line_chart([dict(current=0.0, best=0.0)])
        self.current_pipeline = st.code("")

    def begin(self, evaluations):
        self.status.info(f"Starting evaluation for {evaluations} iterations.")
        self.progress.progress(0)
        self.evaluations = evaluations

    def update_best(self, new_best, new_fn, previous_best, previous_fn):
        self.best_fn = new_fn

    def sample_solution(self, solution):
        self.current += 1
        self.status.info(
            f"""
            Current evaluation: {self.current}/{self.evaluations}.

            Best found={self.best_fn}
            """)
        self.progress.progress(self.current / self.evaluations)
        self.current_pipeline.code(repr(solution))

    def eval_solution(self, solution, fitness):
        self.chart.add_rows([dict(current=fitness, best=self.best_fn)])

    def end(self, best, best_fn):
        self.status.success(
            f"""
            **Evaluation completed!**

            Best solution={best_fn}
            """
        )
