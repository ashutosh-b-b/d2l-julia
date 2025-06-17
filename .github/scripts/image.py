import re
import sys
from pathlib import Path

# Match Markdown image + label below it (specifically fig_*)
FIGURE_BLOCK_RE = re.compile(
    r"!\[([^\]]+)\]\((.*?)\)\s*\n[ \t]*:label:`(fig_[\w-]+)`",
    re.MULTILINE
)

# Match references to figures
NUMREF_FIG_RE = re.compile(r":numref:`(fig_[\w-]+)`")

def convert_figure_blocks(text: str) -> str:
    def repl(m):
        caption = m.group(1).strip()
        src = m.group(2).strip()
        fig_id = m.group(3).strip()
        return (
            f"```@raw html\n"
            f'<img src="{src}" id="{fig_id}" />\n'
            f"```\n\n"
            f"*{caption}*"
        )
    return FIGURE_BLOCK_RE.sub(repl, text)

def convert_figure_refs(text: str) -> str:
    return NUMREF_FIG_RE.sub(r"[Figure](#\1)", text)

def process_file(path: Path):
    text = path.read_text(encoding="utf-8")
    original = text

    text = convert_figure_blocks(text)
    text = convert_figure_refs(text)

    if text != original:
        print(f"Modified: {path}")
        path.write_text(text, encoding="utf-8")

def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_figures.py <path_to_folder>")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.is_dir():
        print(f"Error: '{base_dir}' is not a directory.")
        sys.exit(1)

    for md_file in base_dir.rglob("*.md"):
        process_file(md_file)

if __name__ == "__main__":
    main()
