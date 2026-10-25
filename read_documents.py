from pathlib import Path
from docx import Document


# Document location
file_path = Path("documents/lausuntopyyntolomake-20.10.2025.docx")

# Open Word document
document = Document(file_path)

# Initialise list
texts = []

# Read paragraphs
for block in document.iter_inner_content():

    if hasattr(block, "text"):
        text = block.text.strip()

        if not texts or texts[-1] != text:
            texts.append(text)

    else:
        for row in block.rows:
            for cell in row.cells:
                text = cell.text.strip()

                if not texts or texts[-1] != text:
                    texts.append(text)

output_path = Path("documents/lausuntopyynto.txt")

output_path.write_text(
    "\n".join(texts),
    encoding="utf-8"
)

print(f"Saved {len(texts)} text elements to {output_path}")