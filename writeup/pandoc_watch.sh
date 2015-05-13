# This will watch the "raw" directory for any file changes,
# then compile SVGs to PDFs and compile the markdown documents to a PDF with pandoc.
# - Francis

# Requirements:
# brew install librsvg

if [ "$#" -ne 2 ]; then
    echo "Usage: pandoc_watch <source directory> <output filename>"
    echo "Example: pandoc_watch raw book.pdf"
    exit
fi

while ./sleep_until_modified.py $1 ||
sleep 1
do
    echo "Compiling..."

    # Compile SVGs to PDFs so pandoc can handle them.
    cd $1/assets
    for svg in *.svg; do
        rsvg-convert "$svg" -f pdf -o "${svg%.*}.pdf"
    done
    cd $OLDPWD

    # Compile the notes.
    # Replaces references to svg files with pdf references.
    cd $1

    # Insert pagebreaks between documents.
    for f in *.md; do
        sed '$a\'$'\n''\\pagebreak'$'\n' "$f"
    done | sed "s/\(!\[.*\](.*.\)svg\()\)/\1pdf\2/g" | pandoc -s --mathjax -o $OLDPWD/$2
    cd $OLDPWD

    echo "Compiled."
done

# sed command breakdown:
# First matching group:     \(!\[.*\](.*.\)
#   matches ![.*](.*
# Replacement target:       svg
# Second matching group:    \()\)
#   just matches the last parenthesis
# Replacement:
#   \1  => preserves first matching group
#   pdf => replaces svg with pdf
#   \2  => preserves second matching group