# Schedule — hard finish Tuesday 17 November 2026

Revised 2026-11-14. Supersedes the earlier day-by-day plan, which assumed some work could
land on 18–20 November. Everything now ships by the end of Tue 17 Nov.

## What changed and why

| Change | Was | Now | Reason |
|---|---|---|---|
| M8 — hand over the three `.pkl` files | 18 Nov | **Sun 15 Nov, evening** | 18 Nov is after the deadline; dashboard PR 10 and the report's Results chapter both depend on it |
| A8 — connect the live ESP32 | 20 Nov | **Mon 16 Nov** | Same reason: three days after submission is no use |
| Task 22 — PowerPoint (2 h) | Vernika | **Pranav** | Load balance, see below |
| Task 17 — backup demo video (30 min) | Vernika | **Pranav** | Load balance |
| Paper tasks renumbered | 20, 21, 22 | **23, 24, 25** | Those numbers were already used by the report section |

## Why the reassignment

Work remaining before the reshuffle:

| Person | Hours |
|---|---|
| Vernika | 20 h 15 |
| Tanish | 5 h 50 |
| Anshita | 4 h 30 |
| Pranav | 2 h 15 |
| Yashas | 2 h 05 |
| **Total** | **~35 h** |

The team total fits the time available. The distribution did not: one person held 58% of
the project. Moving the PowerPoint and the backup video to Pranav brings Vernika to
17 h 45 and Pranav to 4 h 45.

## Sat 14 Nov (today, ~4 h) — unblock everyone

The point of today is clearing the three blockers before Sunday starts.

- [ ] Yashas D1 — freeze `training_data.csv` (10 min). **Ten minutes standing in front of 5 h 50 of work. Must be today.**
- [ ] Yashas D4 — send it to Tanish (10 min), straight after D1
- [ ] Anshita — write down the serial line format and send it (10 min). Not a numbered task, but PR 13 and M4 both need it
- [ ] Anshita A3 — lock the firmware to 50 Hz, verify over 60 s (30 min)
- [ ] Pranav P1 — ask the leader the 3 open questions (10 min). Answers can still change scope, so today or never
- [ ] Pranav P2 — agree who presents and who runs the demo (30 min)
- [ ] Vernika — dashboard PRs 1–4 (2 h)
- [ ] Vernika task 13 — report skeleton on the official template (30 min)
- [ ] Vernika task 21 — verify the project proposal (20 min)

## Sun 15 Nov (~7 h) — dashboard working, models trained

- [ ] Tanish M1–M3 — train Decision Tree, Random Forest, KNN (2 h 15). Needs D1
- [ ] Tanish M5 — evaluation metrics and confusion matrices (45 min). State the split method: leave-one-subject-out if the data allows, and say which was used either way
- [ ] Tanish M4 — the three live demo scripts (1 h 30). Needs the serial format
- [ ] Tanish M8 — hand the three `.pkl` files to Vernika, Sunday evening (5 min)
- [ ] Vernika — dashboard PRs 5–12 (5 h 05). PR 10 runs on the existing `random_forest_model.pkl`, so it does not wait on M8
- [ ] Vernika task 14 — report design and implementation (1 h 30)

## Mon 16 Nov (~7 h) — report finished, hardware integrated

- [ ] Anshita A1, A2, A4, A5, A6 (3 h 20): serial dropouts, resolder with strain relief, buzzer and LED pins, battery runtime, rename the collector
- [ ] Anshita A8 — connect the live ESP32 with Vernika (30 min)
- [ ] Yashas D2, D3, D5 (1 h 45): preprocessing writeup, experiment 1, consent and anonymization check
- [ ] Tanish M6, M7 (1 h 15): classifier comparison and window-size experiment. These feed the Results chapter the same day, so they come before Vernika's evening
- [ ] Vernika PR 13 — live ESP32 source (45 min), alongside Anshita
- [ ] Vernika task 15 — report Results (45 min). Needs the M5–M7 numbers
- [ ] Vernika task 16 — report limitations and conclusion (30 min)
- [ ] Vernika task 12 — user manual (45 min). Droppable
- [ ] Vernika task 20 — requirement specification (45 min). Droppable
- [ ] Vernika tasks 23–24 — paper outline and first draft (2 h 45)

## Tue 17 Nov (~5 h) — finish and submit

- [ ] Pranav task 22 — PowerPoint, 10–12 slides (2 h), built from the report
- [ ] Pranav task 17 — backup demo video (30 min). Droppable
- [ ] Pranav P3 — collect and store the consent forms (30 min)
- [ ] Vernika task 25 — finish and proofread `paper.doc` (30 min)
- [ ] Vernika PR 14 — README and submission packaging (30 min)
- [ ] Pranav P4 — build and lock the submission ZIP (30 min): every required filename present and spelled exactly
- [ ] Vernika task 19 — final upload check (20 min)
- [ ] Whole team — rehearsal, tasks 18 and P5 (45 min)

## What breaks this plan

Worst first.

1. **Open question 1 is unanswered.** If the leader says the hardware must be an Arduino, the firmware becomes a rewrite and this schedule is void. Chase it today.
2. **D1 slipping past Saturday.** All eight of Tanish's tasks sit behind one ten-minute job.
3. **PR 9, feature extraction, overrunning.** If the feature order does not match the training CSV, every prediction is silently wrong while the app still looks fine. Budget an extra hour on Sunday.

## Droppable, in this order

1. Backup demo video — 30 min
2. Requirement specification — 45 min
3. User manual — 45 min

None of the three is on the six-item submission checklist. Of the written deliverables,
only `report.doc` and `paper.doc` are.
