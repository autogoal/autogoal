from illiterate.core import Parser, Docstring, Markdown, Python

from pathlib import Path
import sys
import black

output = Path(sys.argv[2])
output.mkdir(exist_ok=True)

for py in Path(sys.argv[1]).rglob("*.py"):
    print(py)

    content = Parser(py.open(), inline=False, module_name=py.name)
    output_file = output / f"test_{py.name}"
    with output_file.open("w") as fp:
        for i, block in enumerate(content.parse().content):
            if isinstance(block, Python):
                fp.write(str(block))
                fp.write("\n\n")

            if isinstance(block, Docstring):
                fp.write(f"def test_block_{i:000d}():\n")

                for line in block.content:
                    fp.write(" " * 4 + line)

                fp.write('    """\n    pass\n\n')

    black.format_file_in_place(
        output_file, fast=True, mode=black.FileMode(), write_back=black.WriteBack.YES
    )
