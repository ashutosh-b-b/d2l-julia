import re
import sys
from pathlib import Path

# Regex to match from <?xml ...?> to </svg>
SVG_BLOCK_RE = re.compile(
    r"(?s)<\?xml\b[^>]*?\?>\s*(<svg\b.*?</svg>)"
)

# Regex to match ANSI escape sequences (e.g., \x1b[36m, \x1b[1m, etc.)
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")

# Regex to match indented log blocks with lines starting with "[ Info:"
LOG_BLOCK_RE = re.compile(
    r"((?:^[ \t]+\[ Info:.*\n?)+)", re.MULTILINE
)

def wrap_log_blocks(text: str) -> str:
    def replacer(match):
        log_block = match.group(1).rstrip()
        return (
            "```@raw html\n"
            '<div style="max-height:300px; overflow-y:auto; background:#111; color:#eee; padding:1em; border-radius:5px;">\n'
            f"<pre>{log_block}</pre>\n"
            "</div>\n"
            "```"
        )
    return LOG_BLOCK_RE.sub(replacer, text)

def process_file(path: Path):
    text = path.read_text(encoding="utf-8")

    # Wrap SVG blocks
    new_text, svg_count = SVG_BLOCK_RE.subn(r"```@raw html\n\1\n```", text)

    # Remove ANSI escape sequences
    new_text, ansi_count = ANSI_ESCAPE_RE.subn("", new_text)

    # Wrap indented log blocks
    wrapped_text = wrap_log_blocks(new_text)

    if svg_count > 0 or ansi_count > 0 or wrapped_text != new_text:
        print(f"Modified: {path} ({svg_count} SVG block{'s' if svg_count != 1 else ''}, "
              f"{ansi_count} ANSI sequence{'s' if ansi_count != 1 else ''})")
        path.write_text(wrapped_text, encoding="utf-8")

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