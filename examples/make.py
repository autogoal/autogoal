# Convert examples in this folder to their corresponding .md files in docs/examples

from pathlib import Path


def main():
    current = Path(__file__)
    folder = current.parent

    for fname in folder.iterdir():
        if fname.name.startswith("_"):
            continue
        if not fname.name.endswith(".py"):
            continue
        if fname.name == current.name:
            continue

        process(fname)


class Markdown:
    def __init__(self, content):
        self.content = content

    def print(self, fp):
        fp.write("\n")

        for line in self.content:
            if line.startswith("# "):
                fp.write(line[2:])
            else:
                fp.write("\n")

        fp.write("\n")

class Python(Markdown):
    def print(self, fp):
        fp.write("```python\n")

        for line in self.content:
            fp.write(line)

        fp.write("```\n")


def process(fname: Path):
    content = []

    with fname.open("r") as fp:
        current = []
        state = 'markdown'

        for line in fp:
            if line.startswith("#"):
                if state == 'python':
                    if current:
                        content.append(Python(current))
                        current = []
                    state = 'markdown'
                current.append(line)
            else:
                if state == 'markdown':
                    if current:
                        content.append(Markdown(current))
                        current = []
                    state = 'python'
                current.append(line)

        if current:
            if state == 'markdown':
                content.append(Markdown(current))
            else:
                content.append(Python(current))

    output = fname.parent.parent / "docs" / "examples" / (fname.name[:-3] + ".md")

    with output.open("w") as fp:
        for c in content:
            c.print(fp)


if __name__ == "__main__":
    main()
