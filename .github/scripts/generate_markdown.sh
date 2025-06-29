#!/bin/bash

# Set the target directory (can be passed as an argument or hardcoded)
TARGET_DIR="${1:-.}"  # Default to current directory if no argument provided
echo $TARGET_DIR
# Ensure the directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: '$TARGET_DIR' is not a valid directory."
    exit 1
fi

# Iterate over each subdirectory
PARENT_DIR="$(dirname "$TARGET_DIR")"
MARKDOWN_DIR="$PARENT_DIR/Julia_Markdown/src"
mkdir "$MARKDOWN_DIR/img"

for dir in "$TARGET_DIR"/*/; do
    # Check if there are any matches
    [ -d "$dir" ] || continue
    OUTPUT_DIR="$MARKDOWN_DIR/$(basename "$dir")"
    mkdir -p "$MARKDOWN_DIR/$(basename "$dir")"
    jupyter nbconvert "$dir/*.ipynb" --to markdown --output-dir="$OUTPUT_DIR" 
    # Your custom logic here
done

python3 .github/scripts/formatter.py "$MARKDOWN_DIR"
python3 .github/scripts/heading.py "$MARKDOWN_DIR"
python3 .github/scripts/image.py "$MARKDOWN_DIR"
python3 .github/scripts/citation.py "$MARKDOWN_DIR"