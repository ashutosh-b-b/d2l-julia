import re
import sys
from pathlib import Path

# Regex to match from <?xml ...?> to </svg>
SVG_BLOCK_RE = re.compile(
    r"(?s)(<\?xml\b[^>]*?\?>\s*<svg\b.*?</svg>)"
)

def process_file(path: Path):
    text = path.read_text(encoding="utf-8")
    new_text, count = SVG_BLOCK_RE.subn(r"```@raw html\n\1\n```", text)
    if count > 0:
        print(f"Modified: {path} ({count} block{'s' if count > 1 else ''})")
        path.write_text(new_text, encoding="utf-8")

def main():
    if len(sys.argv) != 2:
        print("Usage: python wrap_svg_blocks.py <path_to_folder>")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.is_dir():
        print(f"Error: '{base_dir}' is not a directory.")
        sys.exit(1)

    for md_file in base_dir.rglob("*.md"):
        process_file(md_file)

if __name__ == "__main__":
    main()
