**The model's confidence is informative; the validators' caps make it worse.** Across four runs,
the 137 governed fields the model scored 0.95 or above were right every time (accuracy 1.000),
and the area under the risk-coverage curve (0.030) is far below the 0.079 a confidence that
ranked nothing would give. So the model's own number sorts right from wrong. But the confidence
the system returns, after the caps, is less calibrated than the model's: ECE 0.116 against 0.071
and Brier 0.093 against 0.073 on the same fields. The caps (arithmetic 0.5, shape 0.7, normalized
value 0.85) move 57 fields into the `[0.50, 0.70)` bin, where 88% are in fact right. They are
placeholders set before any data; these numbers say they are too aggressive, not that they should
be removed.

**Confidence-based abstention is effectively off.** `ABSTAIN_BELOW` is 0.3 and no capped
confidence is below 0.5, so every abstention in these runs comes from a HARD code or from the
model itself. The threshold the rule chooses, 0.92, would raise selective accuracy from 0.921 to
0.961 but cut coverage from 0.881 to 0.574 and headline accuracy (abstentions count as errors)
from 0.811 to 0.551: about three in ten more fields withheld to remove a few wrong ones, because
the wrong answers sit at every confidence level below 0.95 rather than in a low tail. I would not
adopt it. Whether a wrong answer costs more than a withheld one is a product decision; on this
evidence the better lever is the grounding rule (see the failure analysis), which withholds
correct values whose raw confidence is mostly 0.85 to 0.96.

**Limits.** Ten documents, four runs of them: 488 field instances but 10 independent documents,
so the bins are rough (a bin of 15 fields moves by 7 points per field) and the chosen threshold is
a fit to dev. It is a recommendation to be reported on the test split once that is labeled, not a
result. The correctness target is the model's own value, which the validators never change; the
fixed abstentions (58 instances) are excluded from the reliability tables because no threshold
can move them.
