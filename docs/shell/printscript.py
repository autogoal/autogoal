import sys
from sys import argv
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

import time

console = Console()
console.print("")

with open(sys.argv[1]) as fp:
    text = fp.read()
    styled_text = Syntax(
        "", lexer_name="Python", theme="monokai", line_numbers=True
    ).highlight(text)

time.sleep(0.5)

for i, c in enumerate(text):
    console.print(c, end="", style=styled_text.get_style_at_offset(console, i))

    if c == " ":
        time.sleep(0.2)
    elif c == "\n":
        time.sleep(0.5)
    else:
        time.sleep(0.1)

console.print("\n")
