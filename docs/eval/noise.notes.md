**Two runs of the same setting are not the same run.** Four runs of the identical configuration
(same prompt, model, effort, pages; the last three with a fresh sample each) differ in accuracy by
up to 3.3 points (0.795 to 0.828), flip between 2 and 12 of 122 fields between any two runs (1.6%
to 9.8%), and 15 fields are right in some runs and wrong or withheld in others. Output tokens
vary by 39% across runs, and one document by almost 3x (doc_022: 6,987 to 19,917 tokens, 28 s to
78 s). The run-level p95 latency, which with 10 documents is the single slowest call, moves from
33 s to 78 s.

**What it means for the rest of this project.**
- A difference between two runs is a finding only if it is larger than this. An accuracy
  difference under about 3 points, or a flip count under about 12, is inside the noise. The M8
  sweep's 2-point tolerance and the regression gate's 2-point materiality threshold are below the
  run-to-run range. The gate is protected from false alarms by also requiring the exact McNemar
  p-value: three of the six pairs above flip 10 or more fields, and none has a one-sided p below
  0.19. The sweep has no such protection: its choice of `effort-low` over the baseline is not
  distinguishable from noise.
- The only effects in the M8 sweep that exceed the noise are large ones: `effort-off` uses roughly
  a quarter to a third of the output tokens (14.7k against 45k to 76k) and is about 2.5x faster,
  and the switch from GLM 5.3 Flash to DeepSeek V4
  Flash Vision cut median latency about 7x at the same accuracy within noise.
- Quote medians and ranges, not a p95 from one run of 10 documents.
- Ten documents give an accuracy standard error of several points. To detect a 2-point change
  reliably the labeled set must be much larger, or each configuration must be run several times
  and judged on the mean; this study used four runs of one configuration.
- The 10 documents are the independent sample. Repeating runs measures sampling noise, not
  document-to-document variation, so pooling runs narrows nothing about generalisation.
