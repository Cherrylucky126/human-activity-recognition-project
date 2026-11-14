# HAR Dashboard — what I understand and what I will do

Written 2026-11-14 (Sat) for Vernika Sharma, Team 2, Applied Computer Science Lab, SRH Heidelberg.

**Status: AWAITING APPROVAL. Nothing will be built until Vernika approves this file.**

---

## 1. The project, as I understand it

Team 2 is building a wrist-worn Human Activity Recognition (HAR) unit and the software around it. The hardware exists: an ESP32 DevKit with an MPU6050 IMU and a MAX30102 PPG sensor, in a 3D-printed watch-style case, streaming over USB to a PC.

Submission deadline: Tuesday 17 November 2026. Today is Saturday 14 November, which the task plan marks as a buffer day. That leaves Sun 15, Mon 16, Tue 17.

Vernika owns four deliverables: the dashboard, the project report, the user manual, and the research paper. Other roles (Anshita, Yashas, Tanish, Pranav) own the firmware, the dataset, the ML models, and coordination.

This file covers the DASHBOARD only.

## 2. What I am building

One Python program: a Dash web application. Vernika runs `python app.py`, opens `localhost:8050` in a browser, and sees a live dashboard. One app, not several.

Data path:

    wrist unit -> USB serial -> buffer -> 2 s window -> features -> Random Forest -> screen
        (or the simulator, when the ESP32 is not plugged in)

Incoming data: 50 samples per second. Accelerometer x/y/z and gyroscope x/y/z from the MPU6050, plus the MAX30102 IR value used for heart rate.

What the screen shows:

- Current activity and the model's confidence in it
- Battery, connection state, measured sampling rate, heart rate
- Live sensor values and a step count (pedometer)
- Two scrolling graphs: accelerometer and gyroscope, last 10 seconds
- One bar per activity showing how likely each is right now
- A strip of the last 60 seconds of predictions
- A red banner when a fall is detected

## 3. Two design points I want to be explicit about

**1. The simulator is not a throwaway.** It is a permanent second data source sitting behind the same interface as the real device. The dashboard runs identically with or without hardware. This is what lets the dashboard be built and demonstrated without waiting for the firmware.

**2. Fall detection does not use the machine-learning model.** It is a separate threshold rule on raw accelerometer data: free-fall below 0.6 g for 30-100 ms, then an impact peak above 2.5 g, then about a second of stillness. The Random Forest never predicts a "fall" class, because the training data contains no falls. The thresholds come from Bagala et al. (2012), cited in the literature review.

## 4. Decisions already taken

- Framework: Dash, not Tkinter. Chosen for faster graph work.
- Data: build on simulated data first, connect the real ESP32 later.
- Granularity: one pull request per task, 14 in total.
- Each PR leaves the app runnable. If time runs out at PR 8, there is still a working dashboard to demonstrate.

## 5. The 14 pull requests

| # | Task | Time | Notes |
|---|---|---|---|
| 1 | Repo scaffold + Dash skeleton | 20 min | |
| 2 | Data source interface + simulator | 40 min | |
| 3 | Rolling buffer + live tick | 30 min | |
| 4 | Layout shell | 30 min | |
| 5 | Header: activity, time, confidence, gesture | 30 min | |
| 6 | Status strip: battery, link, Hz, heart rate | 30 min | |
| 7 | Sensor panel + pedometer | 30 min | |
| 8 | Accelerometer + gyroscope graphs | 45 min | |
| 9 | Feature extraction | 45 min | highest risk |
| 10 | Model inference (load the .pkl) | 40 min | needs frozen model |
| 11 | Probability bars + prediction timeline | 45 min | |
| 12 | Fall detection + warning banner | 40 min | |
| 13 | Live ESP32 serial source | 45 min | blocked: serial format |
| 14 | README + submission packaging | 30 min | |

Time:

    PRs 1-12, all on simulated data ....... 7 h 05 min
    PR 13, needs the wrist unit ...........      45 min
    PR 14, submission day .................      30 min
    TOTAL ................................. 8 h 20 min

PRs 1-12 fit Day 1's 7-hour budget with no slack. Add roughly 5 minutes per PR for Vernika's review, about an hour in total.

PR 9 is the most likely to overrun. If the feature order does not match the training CSV, the model still runs but predicts nonsense, and nothing on screen looks broken. That is why it has its own PR and a test.

## 6. Rules I am working under

- **I will not merge anything.** Not any PR, not ever, without Cherrylucky126's approval. I prepare branches and set up the PR page; the Create pull request button and the Merge button are Vernika's and Cherrylucky126's.
- I will not push to `main` directly.
- I will not touch other people's deliverables (firmware, dataset, ML models) unless a task says "Needs:" and Vernika asks me to.

## 7. Where the code goes

The repo is `Cherrylucky126/human-activity-recognition-project`, private, on branch `main`. Current contents:

    app/
      arduino-ide-data-collection      (file, no extension)
      arduino-ide-imu-calibration      (file, no extension)
    data-preprocessing/
    data-processing/
    firmware                           (a FILE, not a folder)
    README.md

My recommendation: the dashboard goes in a new `dashboard/` folder, because `app/` already holds Arduino firmware sketches despite the README reserving `app/` for the companion app. **Needs Vernika's decision.**

Access note: the GitHub API cannot see this repo (404) even though Chrome opens it fine. Same account, `vernikas-ctrl`; the session token is scoped to selected repositories and this one is not listed. Workaround: a local clone plus Claude in Chrome. No action needed unless we want API pushes.

## 8. Problems I found that are not mine to fix

Flagged for Vernika and the team, in priority order.

1. **Two handoff dates land after the deadline.** Task M8 promises the three .pkl files on 18 Nov; task A8 targets 20 Nov. The deadline is 17 Nov. Dashboard PRs 10 and 13 and the report's Results chapter all depend on these.

2. **Nothing in the repo has a file extension.** No .py, no .ino. The submission checklist requires exact names such as `source_code_1_data_collection.py`. Affects tasks A6, D1 and P4.

3. **Task numbers 20, 21 and 22 are used twice** in the task plan: once in the report section, once in the research-paper section. The paper's should be 23, 24, 25.

4. **Model filename mismatch.** The trained file is `random_forest_model.pkl`; the submission checklist requires `randomForest.pkl`. Also, that model was trained on `extracted_features.csv`, while task D1 freezes a new `training_data.csv`. If they differ, the demo model is not the model the report describes.

5. **Validation method.** The literature review recommends leave-one-subject-out validation and warns that random splits overestimate accuracy. Task M5 does not state how the data is split. If the paper cites LOSO and the results use a random split, an examiner will notice.

6. **Model count is inconsistent.** The literature review says the brief asks for four classifiers, tables five, and the submission requires three. SVM is named the strongest performer but is not a deliverable.

7. **Both `data-preprocessing/` and `data-processing/` exist** in the repo. The README mentions only `data-processing/`.

## 9. What I need before PR 1

- Approval of this file.
- The repo cloned locally, and the path:

        cd "$HOME"
        git clone https://github.com/Cherrylucky126/human-activity-recognition-project.git

- A decision: `dashboard/` (my recommendation) or `app/`.

## 10. What happens once approved

I start at PR 1 and work down the list in order. After each PR I stop, tell Vernika what changed, and wait. Vernika pushes and clicks Create pull request; Cherrylucky126 approves and merges. I never merge.
