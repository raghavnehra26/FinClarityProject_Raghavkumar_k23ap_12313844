# LendingClub source files

The supplied files are large CSVs. Keep them outside Git and do not load both fully into memory:

- `accepted_2007_to_2018Q4.csv`: 2,260,701 rows, 151 columns. Includes 1,076,751 `Fully Paid`, 878,317 `Current`, 268,559 `Charged Off`, and other/missing statuses.
- `rejected_2007_to_2018Q4.csv`: 27,648,741 rows, 9 columns. Contains application records and fields such as amount requested, DTI, employment length, and loan title.

The local training command defaults to the files at:

```text
C:\Users\ragha\AppData\Local\Temp\accepted_2007_to_2018Q4.csv
C:\Users\ragha\AppData\Local\Temp\rejected_2007_to_2018Q4.csv
```

Temporary files may be removed by Windows. If that happens, extract both CSVs from the ZIP into a stable local folder and pass their actual paths to `train_application_model.py --accepted ... --rejected ...`. A CSV path cannot refer to a file still inside an unextracted ZIP.

Crucially, these are **unmatched** historical source datasets. The accepted file consists of issued loans; its statuses are repayment outcomes, not application decisions. The rejected file contains applications but has no matching record for each accepted loan. This project therefore trains only a clearly labeled source-file-membership demonstration—not a real approval, eligibility, or default model.

Training streams the files in chunks, validates numeric fields, and reservoir-samples at most 40,000 rows from each. The rejected DTI values include extreme outliers; values outside 0–100 are treated as missing for modeling. The dataset report lists selected-column missing/invalid counts. It does not scan for full-file duplicate rows.

The Kaggle page lists CC0: Public Domain. The data are derived from LendingClub records; confirm applicable source terms and attribution before redistribution or deployment. This is not legal advice. Do not commit either CSV, applicant data, or cloud credentials.
