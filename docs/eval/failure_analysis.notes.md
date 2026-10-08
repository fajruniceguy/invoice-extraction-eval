**The accuracy gap is mostly one guard, not the model.** Of 122 labeled fields, 14 were returned
as abstained. 13 of those carry the HARD code `evidence_not_grounded`; none was withheld for low
confidence (`ABSTAIN_BELOW` is 0.3, and a capped confidence never goes below 0.5, so that
threshold never fires); one was the model's own abstention. Of the 13, **11 withheld a value the
model had right** (the `cost` rows above), and only 2 withheld a wrong one (doc_015 `tax`, doc_028
`document_date`, where the model invented a date the label says is absent: the guard did its job
there). Had the 11 been kept, dev-final would be 110/122 (90.2%) instead of 99/122 (81.1%).

**Why they were withheld (verified, not guessed).** Ten of the eleven are money fields whose
quote prints a number like `208,179`, `84.000`, `135.400`, `13.992` and the receipt prints no
currency marker, so the document currency is `absent`. `normalize.parse_number` refuses to guess
those (a three-digit group is thousands in one locale and decimals in another) and returns
`ambiguous` with both candidates; in every one of these cases the model's value is one of the two
candidates. The grounding rule says an ambiguous span "cannot confirm a value" and hard-abstains.
Not confirmed is not the same as contradicted: that is the design flaw, and the normalizer is
behaving as specified. The eleventh is doc_028 `vendor_address`, where the model's value says
`Jl.` and its verbatim quote says `J1.`: a one-character OCR slip separates a correct value from
its own quote.

**The line items are the same story.** As returned, 26 of 81 labeled rows are missed and 23 are
spurious (row F1 0.69). Scored on the model's own rows, 78 are matched, 3 missed and none
spurious (F1 0.98). The difference is 33 rows returned as abstained: 12 of 13 on doc_030 and all
13 on doc_022. Every one of the 33 carries the same `evidence_not_grounded` code. An abstained row
pairs with nothing, so it counts once as missed and once as spurious. The extractor reads these
tables fine.

**What is genuinely wrong** (9 fields, not abstained): 4 `vendor_address`, 3 `payment_method`,
`vendor_name` and 1 missed `discount`. All but two are string mismatches on free-text fields (a
partial address, `KFC` for `KFC Soekarno Hatta Malang`); they are about how much of a printed
block counts as the value, a labeling and comparator granularity question as much as a model error.

**What this does not show.** One run of 10 documents; the noise study shows these abstentions
flip between runs (doc_022's four money fields are withheld in three of four identical runs and
correct in the fourth), so the 11 is a typical value, not a constant.

**Options for the grounding flaw, none applied (the validators are out of this milestone's scope).**
(a) Treat an ambiguous reading that contains the model's value as consistent: keep the value and
cap its confidence (a soft code) rather than abstain; keep the HARD code for a value that matches
no candidate. On dev this would restore about 10 correct money fields (99 to 109 of 122 correct,
89%) and return doc_015 `tax` as a wrong answer instead of a withheld one. (b) Give the grounder the document's
locale when no currency is printed, e.g. a configured default currency for the deployment.
(c) Infer the thousands convention from the document's other amounts. (d) Leave it and report the
guard's price as above. (a) is the smallest change and keeps the no-guessing rule intact.
