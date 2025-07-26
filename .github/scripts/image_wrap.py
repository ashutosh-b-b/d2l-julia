import re
import sys
from pathlib import Path

# Match <img ...> tags
IMG_TAG_RE = re.compile(r"<img\s[^>]*>", re.IGNORECASE)

# Check for the start of a raw html block
RAW_BLOCK_START_RE = re.compile(r"^\s*```@raw html\s*$")
RAW_BLOCK_END_RE = re.compile(r"^\s*```\s*$")

def convert_img_tags(text: str) -> str:
    lines = text.splitlines(keepends=True)
    new_lines = []
    in_raw_block = False

    for line in lines:
        stripped = line.strip()

        if RAW_BLOCK_START_RE.match(stripped):
            in_raw_block = True
            new_lines.append(line)
            continue
        elif RAW_BLOCK_END_RE.match(stripped) and in_raw_block:
            in_raw_block = False
            new_lines.append(line)
            continue

        if not in_raw_block and IMG_TAG_RE.search(line):
            def wrap_match(m):
                return f"```@raw html\n{m.group(0)}\n```\n"
            line = IMG_TAG_RE.sub(wrap_match, line)

        new_lines.append(line)

    return ''.join(new_lines)

def process_file(path: Path):
    text = path.read_text(encoding="utf-8")
    new_text = convert_img_tags(text)
    if new_text != text:
        print(f"Modified: {path}")
        path.write_text(new_text, encoding="utf-8")

def main():
    if len(sys.argv) != 2:
        print("Usage: python img_wrap.py <path_to_folder>")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.is_dir():
        print(f"Error: '{base_dir}' is not a directory.")
        sys.exit(1)

    for md_file in base_dir.rglob("*.md"):
        process_file(md_file)

if __name__ == "__main__":
    main()
