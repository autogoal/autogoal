import time
import textwrap

from autogoal.search import Logger
from telegram.ext import Updater, Dispatcher, CommandHandler
from telegram import ParseMode
from telegram.utils.helpers import escape_markdown


class TelegramLogger(Logger):
    def __init__(self, token, channel: str = None, name=""):
        self.name = name
        self.channel = int(channel) if channel and channel.isdigit() else channel
        self.last_time = time.time()
        self.updater = Updater(token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.progress = 0
        self.generations = 1
        self.best = 0.0
        self.current = ""
        self.message = self.dispatcher.bot.send_message(
            chat_id=self.channel,
            text=f"**{self.name}**\nStarting...",
            parse_mode=ParseMode.MARKDOWN,
        )

    def begin(self, generations, pop_size):
        self.generations = generations
        self.init_time = time.time()
        self._send_state()

    def update_best(self, new_best, new_fn, *args):
        message = f"**New Best:** {new_fn}"
        try:
            self.dispatcher.bot.send_message(
                chat_id=self.channel, text=message, parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            print(e)

    def end(self, best, best_fn):
        self.best = best_fn
        self._send_state()

    def eval_solution(self, solution, fitness):
        message = f"""
        **Evaluation:**\n
        - Solution: {escape_markdown(repr(solution))}
        - Fitness: {fitness}
        """
        try:
            self.dispatcher.bot.send_message(
                chat_id=self.channel, text=message, parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            print(e)

    def error(self, e: Exception, solution):
        message = f"**ERROR:**\n{escape_markdown(str(e))}"
        try:
            self.dispatcher.bot.send_message(
                chat_id=self.channel, text=message, parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            print(e)

    def start_generation(self, generations, best_fn):
        self.progress = self.generations
        self.best = best_fn
        self._send_state()

    def _send_state(self):
        if not self.channel:
            return

        if time.time() - self.last_time < 10:
            return

        self.last_time = time.time()

        text = textwrap.dedent(
            f"""
            **{self.name}**
            Best: `{float(self.best):0.3}`
            Iterations: `{self.progress}/{self.generations}`
            Total Time: {time.time() - self.init_time}
            """
        )
        try:
            self.dispatcher.bot.send_message(
                chat_id=self.channel, text=text, parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            print(e)
