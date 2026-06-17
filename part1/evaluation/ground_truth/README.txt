Ground-truth JSON files for offline accuracy evaluation
=======================================================

Each JSON file in this directory corresponds to one of the filled BL283
sample forms in phase1_data/.  Filenames match the PDF stem:

  283_ex1.json  →  phase1_data/283_ex1.pdf
  283_ex2.json  →  phase1_data/283_ex2.pdf
  283_ex3.json  →  phase1_data/283_ex3.pdf

HOW TO LABEL
------------
1. Open the PDF and the corresponding JSON stub side-by-side.
2. Fill in the exact values that appear on the form for each field.
3. Leave a field as "" if it is genuinely blank on the form.
4. Run the evaluation:
     python -m part1.evaluation.evaluate

The harness skips any file whose ground-truth JSON is entirely empty, so
you can label incrementally — label one file, run, see results.

NOTES
-----
- Phone numbers: digits only (no dashes/spaces), matching extractor output.
- Dates: day/month/year as numeric strings ("05", "11", "2023").
- Gender: use "זכר" or "נקבה" (the normalised output the extractor produces).
- Signature: "קיימת" if a signature is present, "" if not.
