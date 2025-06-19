import re
import sys
from pathlib import Path

# Only match label types: sec_, subsec_, chap_
HEADING_LABEL_RE = re.compile(
    r"^(#{1,6}) (.+?)\n[ \t]*:label:`((?:sec_|subsec_|chap_)[-\w]+)`",
    re.MULTILINE,
)

def convert_heading_labels(text: str) -> str:
    def repl(m):
        hashes = m.group(1)
        title = m.group(2).strip()
        label = m.group(3).strip()
        return f"{hashes} [{title}](@id {label})"
    return HEADING_LABEL_RE.sub(repl, text)

def process_file(path: Path):
    text = path.read_text(encoding="utf-8")
    new_text = convert_heading_labels(text)
    if new_text != text:
        print(f"Modified: {path}")
        path.write_text(new_text, encoding="utf-8")

def main():
    if len(sys.argv) != 2:
        print("Usage: python heading.py <path_to_folder>")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.is_dir():
        print(f"Error: '{base_dir}' is not a directory.")
        sys.exit(1)

    for md_file in base_dir.rglob("*.md"):
        process_file(md_file)

if __name__ == "__main__":
    main()
