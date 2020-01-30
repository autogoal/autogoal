import time
import textwrap

from autogoal.search import Logger
from logbot import Client


class TelegramBotLogger(Logger):
    def __init__(self, token, name=""):
        self.name = name
        self.last_time = time.time()
        self.client = Client(token=token, host="http://127.0.0.1", port=6778)
        self.progress = 0
        self.evaluations = 1
        self.best = 0.0
        self.current = ""

    def begin(self, evaluations):
        self.evaluations = evaluations
        self._send()

    def update_best(self, new_best, new_fn, *args):
        self.best = new_fn
        self._send()

    def end(self, best, best_fn):
        self.best = best_fn
        self._send()

    def eval_solution(self, solution, fitness):
        self.progress += 1
        self.current = repr(solution)
        self._send()

    def _send(self):
        try:
            self.client.send(
                textwrap.dedent(
                    f"""
                    **{self.name}**
                    Best: `{float(self.best):0.3}`
                    Iterations: `{self.progress}/{self.evaluations}`
                    """
                ),
                progress=self.progress / self.evaluations,
                edit=True,
            )
        except:
            pass
