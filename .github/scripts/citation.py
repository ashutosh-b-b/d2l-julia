import re
import sys
from pathlib import Path

# Match :citet:`ref1, ref2` or :cite:`ref1`
CITE_STYLE_RE = re.compile(r":(cite|citet):`([^`]+)`")

def convert_citation_syntax(text: str) -> str:
    def repl(m):
        tag = m.group(1)  # 'cite' or 'citet'
        refs = [f"[{ref.strip()}](@{tag})" for ref in m.group(2).split(",")]
        return ", ".join(refs)
    return CITE_STYLE_RE.sub(repl, text)

def process_file(path: Path):
    text = path.read_text(encoding="utf-8")
    new_text = convert_citation_syntax(text)
    if new_text != text:
        print(f"Modified: {path}")
        path.write_text(new_text, encoding="utf-8")

def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_cites.py <path_to_folder>")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.is_dir():
        print(f"Error: '{base_dir}' is not a directory.")
        sys.exit(1)

    for md_file in base_dir.rglob("*.md"):
        process_file(md_file)

if __name__ == "__main__":
    main()
