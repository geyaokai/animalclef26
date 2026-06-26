# AnimalCLEF 2026 Working Note

Draft LaTeX working note for the AnimalCLEF 2026 submission, using the
CEUR-WS single-column `ceurart` template.

## Build

```bash
cd /home/hechen/gyk/animalclef/working_note
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The CEUR template support files are vendored in this folder:

- `ceurart.cls`
- `elsarticle-num-names.bst`
- `cc-by.pdf`
- `cc-by.png`
- `pdfa.xmpi`

## Items to Confirm

- Author name, affiliation, email, and ORCID if needed.
- Replace the temporary VIPL-VSU team author block with the official author list.
- Confirm the final CLEF/CEUR venue metadata if the organizers update the proceedings instructions.
