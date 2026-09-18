# BuzzSuite: From Recording to Data

### A Simple Step-by-Step Protocol

*No coding required — just clicking, naming things, and selecting files.*

*The Activity module used in this protocol is BuzzSuite's activity-tracking pipeline
(built on the original BuzzWatch tracking engine) — this protocol covers that path from
raw video to the activity plot, start to finish.*

---

> **The workflow at a glance**
> 1. Collect your `.h264` video data from the camera.
> 2. Open **BuzzSuite** and use the **Intake wizard** to load the video and create the
>    experiment — this also converts `.h264` to `.mp4` for you, no separate app needed.
> 3. Extract a background image and draw the cage borders.
> 4. Run tracking.
> 5. Make the activity plot.

## Step 1 — Collect your `.h264` data

Copy the `.h264` video files off the **Raspberry Pi** onto the computer. Put them
somewhere you can find them (you will point BuzzSuite at them in Step 2).

## Step 2 — Open BuzzSuite and create the experiment

### 2.1 — Launch the app

Double-click `run_buzzsuite.bat` (or its desktop shortcut) in the BuzzSuite folder.

> **If Windows shows a blue "Windows protected your PC" warning**
> Click **"More info"**, then **"Run anyway"**. This is normal and safe here.

A welcome screen appears first. Click **"New experiment →"** to jump straight into
setup (or, if you're continuing work on an experiment you already created, select it
from the **recent experiments** list, or use **"Browse for experiment…"**, then click
**"Continue →"**).

### 2.2 — The Intake wizard

*(Tab: `1` · `Setup` → sub-tab `Experiment & Video`)*

This one screen replaces what used to be a separate folder-creation step and a separate
Convert app. Work through it top to bottom:

**A — Source video folder.** Click **Browse…** and select the folder with your `.h264`
files (from Step 1), then click **Scan**. BuzzSuite lists how many video segments and
dates it found.

**B — Dates to include.** A checkbox appears for every recording date found. All are
ticked by default — untick any date you don't want included.

**C — Experiment setup.** Fill in these boxes, then click **"Create / open
experiment"**:

- **Experiment:** type `Flare`. (Empty by default.)
- **Incubator:** `Cakung` or `Bangkok` (pick from the dropdown)
- **Species:** e.g. `AedesAegypti`
- **Sex:** `F` or `M` (pick from the dropdown)
- **Cage name:** BuzzSuite fills this in for you from Species + Sex + the dates you
  ticked in Step B (e.g. `AedesAegypti_M_20260721_20260722`). **You can type over it**
  if you'd rather keep the older short style, e.g. `AedesAegyptiM_21-22`.

> **Don't mix up the two "Cage" fields**
> **Incubator** (`Cakung`/`Bangkok`) is what used to be called "Cage Name".
> **Cage name** is a different box — it's the actual experiment folder name (what used
> to be called "Batch Name"). Same idea as before, just split across two clearly-named
> boxes now.

BuzzSuite creates the experiment folder for you automatically — there is no more manual
folder creation in File Explorer. For reference, this is what gets built behind the
scenes (BuzzSuite finds your data drive automatically — you don't need to know or type
this path):

```
<your data drive>\Buzzwatch
  └── Recording
      └── Flare                      (= your "Experiment" box)
          └── Cakung                 (= your "Incubator" box)
              └── AedesAegyptiM_21-22  (= your "Cage name" box)
```

**D — Convert / move video into the experiment.** Click **"Convert / move selected
dates"**. Any `.h264` files are losslessly converted to `.mp4`; any files that are
already `.mp4` are copied straight in. A progress bar shows how far along it is. Wait
for it to finish.

> **A separate H264 → MP4 Converter tab also exists**
> If you ever want to convert video without going through the wizard (e.g. into a folder
> outside a BuzzSuite experiment), there's also a standalone **"H264 → MP4 Converter"**
> tab along the top of the app that does the same conversion. For the normal workflow,
> Step D above is all you need.

## Step 3 — Background image and cage borders

*(Tab: `1` · `Setup` → sub-tab `Background & Cage Border`)*

1. **Extract Images from Video.**
   Click the button and wait until it finishes.

2. **Get Background from Images.**
   Click the button and wait. *This one takes a while to run.*

3. **Draw the cage borders.**
   1. Click **Show background with borders** (the cage image appears on the right).
   2. Click **Draw Cage Borders** — a separate image window opens.
   3. Click the **four corners of the cage** — **any order is fine**, BuzzSuite
      automatically works out which corner is which.
   4. After the 4th click, a box is drawn joining the corners. Press any key to close
      the window — the borders are saved automatically.

```
        (*)---------------(*)
         |                 |
         |      cage       |
         |                 |
        (*)---------------(*)
```
*click the four corners in any order*

If you ever need to redo this, click **Reset Drawn Borders** first.

## Step 4 — Run tracking

There are two ways to do this — pick whichever you prefer. **4.2 (the Experiment
Dashboard) works perfectly well for a single experiment too**, not just several at once,
so don't feel you need to use 4.1 just because you're only tracking one.

### 4.1 — The Single experiment tab

*(Tab: `2` · `Analysis` → sub-tab `Single experiment`)*

1. Choose **what to track**:
   - **All untracked (batch)** — the normal choice; tracks every video segment that
     hasn't been tracked yet and skips ones already done. *(This is selected by
     default.)*
   - **Specific segments** — pick individual segments from a list.
   - **Time range** — pick a start and end time.

2. Click **"Run tracking"**. *This normally takes several hours* — you don't need to
   set anything else first; BuzzSuite manages processing power automatically.

3. **(Optional)** If you want to filter out unrealistic flight speeds, use the **Speed
   filter** box (Min/Max, or leave **"Auto Detect"** ticked — it is by default) and
   click **"Run Speed Filter"**. This is usually not needed.

That's it for this tab — once tracking finishes, BuzzSuite automatically saves and
combines the data for you. There is no separate "Concatenate" or "Load Data" step
anymore.

### 4.2 — Or use the Experiment Dashboard

*(Tab: `2` · `Analysis` → sub-tab `Multiple experiments`)*

This does the same tracking job as 4.1, with a clearer per-experiment progress view. Use
it for one experiment, or several at once — it works the same way either way:

1. Pick the experiment from the **Experiment** dropdown (it lists your 10 most recently
   opened experiments), or click **Browse…** to pick any other one.

2. Click **"Add to dashboard"**. This adds it as a row **without** starting tracking
   yet. If you want to track more than one experiment together, repeat this step for
   each of them first.

3. Click **"Start"** on that row to begin tracking it — or, if you've added more than
   one, click **"Start all queued"** to begin all of them together.

4. Each row shows its own progress bar and status (*queued* → *starting* → *tracking
   i/N segments* → *concatenating* → *done*), and a **Cancel** button if you need to
   stop it partway through.

5. On the right, a **"Live file progress"** panel shows the progress of each individual
   video segment currently being tracked.

If you do track more than one experiment at once, BuzzSuite automatically splits
processing power between whichever jobs are running — there's no need to set worker
numbers by hand or open a second copy of the app. (You still need to finish Step 3,
background & borders, for each experiment individually before tracking it, by switching
which experiment is loaded on the `1` · `Setup` tab.)

## Step 5 — Make the activity plot

*(Tab: `3` · `Plotting` → sub-tab `Activity (in-cage)`)*

1. Click **"Add experiment"** (pick it from the recent-experiments dropdown, or
   **Browse…**) to load its tracked data in.

2. In the **dates table** below, tick which dates to include, and mark each one **LD**
   (normal light:dark cycle) or **DD** (constant dark) — **LD** is the default for every
   date. Use **"All LD"** / **"All DD"** to set them all at once. Click **"Save LD/DD
   tags"** to keep this for next time.

3. Under **Plot**, leave **Variable** on its default (`total_pixels_moved` — overall
   movement per minute), or pick a different one from the dropdown.

4. Choose how you want it laid out (**Output**):
   - **Individual** — one panel per date *(default)*
   - **Overlay** — every date on one 24-hour axis, useful for comparing days
   - **Continuous** — one panel with real calendar time running along the bottom

5. Click **"Plot"** — this is the **activity plot**. Click **"Save PNG"** to save it as
   an image file.
