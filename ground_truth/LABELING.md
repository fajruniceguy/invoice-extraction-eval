# Labeling guide

Ground truth is what the DOCUMENT says, not what it should say and not what a model returned.
One JSON file per labeled document: `ground_truth/<doc_id>.json`. Every change is logged in
[`CHANGELOG.md`](CHANGELOG.md) (the labeling app does it for you).

## One document per input

A label describes ONE business document. The extractor is asked for one document per input, so an
input that holds several cannot be labeled or scored:

- **Several documents on one page** (two invoices side by side): black one out with a redaction box
  (`needs_redaction`, `data/redactions.yaml`) so the stored page holds one document, or leave the
  input out of the dataset.
- **A bundle** (a PDF of many invoices or companies): slice it with `page_range`, one manifest row per
  document, or leave it out. Do not label 20 pages of unrelated documents as one.

An input left out is removed from the manifest (and its stored pages and draft), with a line in
`CHANGELOG.md` saying why. `doc_004`, `doc_017` and `doc_010` were left out this way.

## The three states of every field

| In the JSON | Meaning | Scored? |
|---|---|---|
| a value (`"ACME"`, `"1250000.00"`) | the document prints this | yes |
| explicit `null` | the field is **absent** on the document | yes: the right answer is status `absent` |
| `"__unlabeled__"` | nobody labeled this field | **skipped** |

An omitted key means `"__unlabeled__"`, never "absent". Use `null` only when you looked and the
document does not print the field. Use `"__unlabeled__"` when you cannot or did not decide
(an unreadable photo, a field you chose not to label). Never guess to avoid `"__unlabeled__"`.

`null` maps onto the extractor's `absent` status. How an `abstained` prediction is scored against
`null` is decided in the metrics code, not here.

## Rule 1: record values AS PRINTED, never computed

The label is what is printed. Do not add up, subtract, multiply or "fix" anything.

- A faktur that prints no grand total gets `total = null` (note: `total not printed`), even if
  you could compute it. Checking that numbers are consistent is the **validator's** job
  (`docextract/validate`), not the label's. A document whose printed numbers do not add up is
  labeled with the printed numbers.
- If a printed value looks wrong (a typo on the document), label the typo and add a note.

## Rule 2: normalization (the only transformation allowed)

| Kind | Label as | Example |
|---|---|---|
| dates | ISO-8601 `YYYY-MM-DD` | `5 Maret 2024` -> `"2024-03-05"` |
| money | a decimal string, no symbols, no thousands separators, `.` as the decimal mark, **printed scale kept** | `Rp 1.250.000,00` -> `"1250000.00"`; `Rp 1.250.000` -> `"1250000"` |
| quantities | the same decimal string | `0,352` -> `"0.352"` |
| currency | the ISO 4217 code | `"IDR"` |

Indonesian number format uses `.` for thousands and `,` for decimals: `1.250.000,00` ->
`1250000.00`, `9.116,8` -> `9116.8`. International format is the mirror image:
`1,250,000.00` -> `1250000.00`. The shared normalizer (`docextract/normalize.py`) does this and
the labeling app runs it on what you type.

**Ambiguous numbers are not guessed.** `1.250` is 1250 in rupiah but 1.25 in dollars; `05/03/2024`
is 5 March or 3 May. The labeling app refuses to guess: type the decimal string or the ISO date
yourself, after checking the document.

**Currency.** Label `currency` as `IDR` only if the document prints it (`Rp`, `IDR`, `Rupiah`) or it
is unambiguous (an Indonesian faktur with rupiah amounts and no other currency). Otherwise `null`
if truly absent, or `"__unlabeled__"` if you cannot tell.

## Rule 3: what goes in each field

| Field | Label |
|---|---|
| `vendor_name` | the seller / merchant as printed (for a faktur: the seller, "Pengusaha Kena Pajak") |
| `vendor_address` | the seller's address as printed, one string |
| `vendor_tax_id` | the seller's tax id as printed (a faktur: the **seller's NPWP**), digits and punctuation as printed |
| `buyer_name` | the customer / buyer as printed (`null` on most receipts) |
| `document_number` | the document's own number: invoice no., receipt no., booking id, or for a faktur pajak the **nomor seri faktur** |
| `document_date` | the date the document was issued / the transaction took place. A hotel check-in date is not the document date; use the confirmation/booking date if one is printed, else `null` |
| `due_date` | the payment due date, if printed |
| `payment_method` | as printed (`VISA ending 1234`, `Transfer`, `Tunai`) |
| `subtotal` | the amount before tax as printed. **A faktur: the DPP** (Dasar Pengenaan Pajak) |
| `tax` | the tax amount as printed. **A faktur: the PPN** |
| `discount` | the total discount if printed as one amount (`Potongan Harga`), else `null` |
| `total` | the grand total / amount to pay, only if printed as such. Otherwise `null` |
| `line_items` | the printed rows, see Rule 4; `null` if the document has no itemization |

**Fakturs.** DPP -> `subtotal`, PPN -> `tax`, nomor seri faktur -> `document_number`, seller NPWP ->
`vendor_tax_id`. *Harga Jual / Penggantian / Uang Muka / Termin* is the gross price before the
discount: it is **not** the document total, so do not put it in `total` (mention it in `notes`).
Beware **DPP Nilai Lain** (the 11/12 base): PPN is then 12% of the *DPP Nilai Lain*, i.e. effectively
11% of the gross price, so **PPN is not 12% of the gross price**. Label the PPN as printed; never
recompute it.

## Rule 4: line items

One entry per printed row, in printed order, with `description`, `quantity`, `unit_price`, `amount`.
Each cell is a value, `null` (the row does not show it: the usual case for a missing quantity) or
`"__unlabeled__"`.

- **Service charge, delivery fee, admin fee, rounding** printed as a row -> a line item, with the
  description exactly as printed (`Service Charge 5%`, `Ongkos Kirim`). They are not folded into
  `tax` or `discount`.
- A discount printed as its own negative row is a line item with a negative amount.
- A multi-line description is one string. A table that continues over several pages is one list.
- `line_items: null` means "labeled: this document has no itemization"; omit the key (or use
  `"__unlabeled__"`) when you have not labeled the rows.

### Long tables: how much to label

Labeling a table of 20 or more rows by hand is slow, and you may skip it. **Label every printed
row, or mark the whole table `unlabeled`: never label part of it.** The extractor's rows that
match nothing you labeled are scored as spurious, so a half-labeled table counts the model's
correct rows against it. A table marked `unlabeled` is simply skipped in scoring, and the document's
other fields still count. To enter a long table quickly, type or paste it into a spreadsheet and
use "Paste rows from a spreadsheet" in the labeling app (columns: description, quantity,
unit_price, amount; verifying the rows still normalizes and checks every number).

The same leniency applies to a free-text field you cannot read or cannot delimit (an address that
runs into other text, a payment method that is half cut off): mark it `unlabeled`. Do not guess
and do not use `null` for "I am not sure": `null` claims the page does not print it.

## Rule 5: redacted fields

Pages are redacted (black boxes) before they are stored. A field you cannot read because a box
covers it is `null` with the note `redacted`. The extractor sees the same redacted page, so
`null` is the correct target.

## Notes

`field_notes` is an optional `{field: note}` mapping for the humans reading the label. The key must
be one of the label's fields (`vendor_name` ... `total`, or `line_items` for the rows as a whole;
a typo is an error) and the note must not be empty. Notes are not scored. Keep to this vocabulary
where it applies:

| Note | Use |
|---|---|
| `redacted` | under a redaction box |
| `total not printed` | no printed grand total |
| `printed typo` | the document itself looks wrong |
| `illegible` | present but unreadable (use `"__unlabeled__"` as the value) |
| `ambiguous number` | printed in a way that could be read two ways; say which you chose |

Free text in the document-level `notes` field is for anything else (a bundle page, a repeated
header, an unusual layout).

## File format

```json
{
  "doc_id": "doc_007",
  "label_mode": "blind",
  "labeled_at": "2026-10-06T09:15:00Z",
  "vendor_name": "PT Contoh Makmur",
  "vendor_tax_id": "01.234.567.8-901.000",
  "document_number": "010.000-24.12345678",
  "document_date": "2024-03-05",
  "due_date": null,
  "subtotal": "1126126.00",
  "tax": "123874.00",
  "total": null,
  "discount": "__unlabeled__",
  "line_items": [
    {"description": "Kertas A4 80gr", "quantity": "10", "unit_price": "45000", "amount": "450000"}
  ],
  "field_notes": {"total": "total not printed"},
  "notes": ""
}
```

Rules the loader enforces (`evalkit/gt_store.py`):

- Unknown keys are rejected (a typo such as `vendor_nme` must fail, not silently become unlabeled).
- The file is named after the document: `doc_007.json` holds `"doc_id": "doc_007"`.
- `label_mode` is `prelabel` or `blind` and must be the one in `data/manifest.csv`;
  `labeled_at` is an ISO timestamp WITH a timezone (`2026-10-06T09:15:00Z`, `...+07:00`).
- Money and quantities are strings, so the printed scale survives (`"1250000.00"` is not
  `"1250000"`); dates are `YYYY-MM-DD`.
- The labeling app writes every field explicitly, `"__unlabeled__"` included, so a saved file shows
  exactly the decisions that were made.

**Documents that share content** (`gt_from` in the manifest: a second photo of the same invoice, a
degraded variant) use the labels of the document they point at: label once, never copy. They must
not have a file of their own; asking for their label returns the owner's under their own id.

**Constructed invoices** (`cons_001`..`cons_003`) have generated labels, written by
`scripts/construct_invoices.py` from the same data the PDF is drawn from. They carry no
`label_mode` (nobody labeled them) and are never edited by hand: change the generator and
regenerate (`--force` replaces a label and logs it in `CHANGELOG.md`).

`python -m uv run python scripts/check_ground_truth.py` checks all of this, plus that every blind
document is free of drafts, and prints how much of each split is labeled.

## Blind and pre-labeled documents

`label_mode` is recorded in every label.

- `prelabel`: a model drafts the label (`scripts/prelabel.py` -> `ground_truth/drafts/<doc_id>.json`)
  and a human verifies **every** field against the page. The prelabeling model is deliberately a
  different model from the extractor.
- `blind`: a human labels from the page alone; **no draft exists and none is ever shown**. These
  documents measure how good the labels are without any model influence.

The labeling app only lets you save when every field is `verified`, `absent` or `unlabeled`.
