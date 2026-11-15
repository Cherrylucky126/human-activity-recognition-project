# Known limitations

Written for the report and the viva. Each entry is a thing an examiner could
reasonably ask about, with what we know and what we would do differently.
Stating these is deliberate: every one of them is easier to defend when it is
raised by us first.

Last updated 15 November 2026.

---

## 1. There is no "running" class

**What the models do.** They recognise six activities: downstairs, laying,
sitting, standing, upstairs, walking. Running is not among them.

**Why.** Running was recorded and then removed from the training set by
`data-preprocessing/remove-running`. It was a deliberate choice, not an
oversight.

**The consequence, stated plainly.** A classifier is forced to answer with
one of the classes it knows. Someone running is therefore not reported as
"unknown" - they are reported as walking or upstairs, and with high
confidence, because the model has no way to express "this is none of my
classes". Confidence is relative to the six, not absolute.

**What we would do with more time.** Either keep running as a seventh class,
or add a rejection threshold so that a window no class fits well is shown as
"uncertain" rather than being forced into the nearest one.

### Suggested wording for the report

> The classifier distinguishes six activities: downstairs, laying, sitting,
> standing, upstairs and walking. Running was recorded but removed from the
> training set. As the system has no rejection class, running is reported as
> whichever of the six the model finds closest - usually walking - and with
> high confidence, since confidence is distributed across the six known
> classes rather than measured against the possibility of an unknown one.
> A rejection threshold would be the first addition to the model.

---

## 2. Validation is a random split, not leave-one-subject-out

**What we did.** `train_test_split` with a random split across all windows.

**Why that overstates accuracy.** Windows overlap by 50%, so consecutive
windows share half their samples. A random split puts near-identical windows
on both sides of the train/test boundary. On top of that, windows from the
same participant appear in both sets, so the model is partly rewarded for
recognising a person's particular gait rather than the activity itself. The
reported figure answers "can it recognise this activity in someone it has
already seen?" - which is not the question a wearable has to answer.

**What the standard is.** Leave-one-subject-out: train on all participants
but one, test on the held-out participant, repeat. The HAR literature treats
this as the meaningful protocol, and the project's own literature review
recommends it.

**What to expect.** LOSO accuracy is reliably lower. That is not the method
being worse - it is the honest number, and the random-split figure was
flattering.

### Suggested wording for the report

> Validation used a random train/test split. With 50% window overlap and all
> participants present in both sets, this is optimistic: adjacent windows
> share half their samples, and the model may be recognising a participant's
> gait as much as the activity. Leave-one-subject-out validation, as
> recommended in our literature review, would give a lower and more
> defensible figure, and we would expect accuracy to fall. We report the
> random-split result here and flag it as an upper bound rather than an
> estimate of performance on an unseen wearer.

---

## 3. Fall detection runs below the literature threshold

**The constraint.** The collector firmware never writes MPU6050 register
`0x1C`, so the accelerometer stays on its default +/-2 g range. Every
reading is clipped at 2 g before it leaves the device.

**What that broke.** The literature threshold for a fall impact is about
2.5 g (Bagala et al. survey 2-3 g at the waist; a wrist sees higher peaks
still). On this hardware, 2.5 g can never be reported. Fall detection worked
in simulation and could not have fired on a real recording.

**The decision, 15 November 2026.** Lower the threshold to 1.8 g rather than
re-range the sensor. Setting `0x1C` to +/-8 g is the better fix in
principle, but every trained model was fitted on +/-2 g data and would have
been invalidated, requiring re-collection and re-training that the schedule
did not allow.

**The cost.** 1.8 g is inside the range of ordinary vigorous movement, so the
impact stage alone is weaker evidence than it was. The alert does not rest on
it alone: a fall must also show a 30-150 ms free-fall dip below 0.6 g before
the impact, and a second of stillness after. Walking, running, jumping, a
knock on the table and a fast sit-down are all still rejected in testing.

**What is not yet known.** The 1.8 g threshold has been tested against
synthetic signals only. Walking peaks at 1.60 g in those tests, a margin of
0.2 g. It has not been validated against a real recording.

### Suggested wording for the report

> The accelerometer operates at its default +/-2 g range, so impacts clip at
> 2 g and the 2.5 g threshold used in the literature is unreachable on this
> hardware. We lowered the impact threshold to 1.8 g rather than re-range the
> sensor, which would have invalidated models already trained on +/-2 g data.
> Detection remains a three-stage rule - free fall, impact, stillness - so the
> reduced threshold is not evaluated in isolation. We note that this reduces
> specificity and that validation to date is against synthetic signals.

---

## 4. Heart rate is indicative, not clinical

Heart rate comes from an optical sensor on a wrist strap and is presented as
an indication of pulse, not a medical measurement. The dashboard says so on
the page itself.

The live estimate also differs slightly from the offline pipeline's
`rolling_bpm`, which smooths with a centred median - that needs samples from
after the moment being described, which a live system does not have. The
dashboard estimates the same quantity from past samples only, so small
differences from the offline figures are expected and are not a defect.

---

## 5. Three of the 43 features are approximated live

The IR channel is z-scored across the entire dataset before training, which a
live system cannot reproduce: there is no "entire dataset" while samples are
still arriving. `ir_ac_variance`, `ir_p2p_amplitude` and `ibi_std` are
therefore scaled from a trailing 60 seconds instead.

The other 40 features are computed by the project's own feature code and are
exact.

This is fixable to the last decimal with two numbers: the training set's IR
mean and standard deviation, placed in `TRAINING_IR_MEAN` and
`TRAINING_IR_STD` in `dashboard/inference.py`.

---

## 6. The step count is from a wrist

Gesturing while seated adds steps; walking with a hand in a pocket loses
them; a fall impact registers as one step. Counting only while the classifier
reports walking would remove most of this, at the cost of coupling the
pedometer to the model.
