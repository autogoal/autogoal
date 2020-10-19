import sys
from sys import argv
from rich.console import Console
import time

console = Console()
console.print("")

command = " ".join(sys.argv[1:])
console.print("[bold blue]$[/] ", end="")

time.sleep(0.5)

for c in command:
    console.print(c, end="", style="bold")

    if c == " ":
        time.sleep(0.3)
    else:
        time.sleep(0.1)

console.print("\n")
