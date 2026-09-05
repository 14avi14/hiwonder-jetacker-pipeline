# HiWonder JetAcker LLM Navigation  Pipeline

A modular large language model based navigation pipeline for a HiWonder JetAcker (Ackermann-steering
car, Jetson Orin Nano). Starting milestone: "drive to the [color] cube/ball."

## File structure

```
.
├── target_finder.py          # The "brain": turns instructions into detector prompts
│                              # (Call #1) and detections into a drive delta (Call #2)
├── vision.py                  # Runs YOLOE/YOLO-World detection + real color verification
├── path_planner.py            # Continuous closed-loop motion controller (outputs
│                              # linear_x/angular_z, matching the real JetAcker's
│                              # /controller/cmd_vel topic)
│
├── run_pipeline.py            # Main real end-to-end runner (real detection, real/
│                              # deterministic LLM calls, simulated motion controller)
├── run_pipeline_manual.py     # Same, but Call #2 goes through copy/paste into any
│                              # chat GUI instead of an API key
├── test_pipeline.py           # Fully mocked smoke test, no API key or real photo needed
│
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt

# YOLOE specifically also needs:
pip install git+https://github.com/ultralytics/CLIP.git
```

Note for Mac users: skip MPS. YOLOE's CLIP text encoder crashes on Apple's MPS
backend (unsupported float64 op) -- `vision.py` / the run scripts default to CPU.

### About Call #2's LLM path

`target_finder.py` has a `USE_LLM_FOR_CALL2` flag (currently `True` by default) that
decides whether Call #2 goes through a real LLM or a deterministic fallback. The real
path (`call_llm()`) is currently wired to the Cerebras API, but the provider choice
isn't finalized yet -- `cerebras_cloud_sdk` is deliberately left out of `requirements.txt` for
that reason. If you want to actually exercise the LLM path right now, install it
yourself (`pip install cerebras_cloud_sdk`) and set `CEREBRAS_API_KEY`; otherwise, set
`USE_LLM_FOR_CALL2 = False` to use the deterministic fallback instead, which needs no
extra install or API key.

## How to run

**Quickest sanity check (no API key, no real photo, no dependencies beyond the
base install):**
```bash
python3 test_pipeline.py
```

**Real end-to-end run** (real detection on a real photo, real or deterministic LLM
calls depending on the `USE_LLM_FOR_CALLx` flags in `target_finder.py`):
```bash
python3 run_pipeline.py
```

**Same, but without an API key** (Call #2 via copy/paste into any chat GUI):
```bash
python3 run_pipeline_manual.py
```

## Current state / what's real vs. mocked

- **Real:** object detection (YOLOE or YOLO-World, both prompted and default-vocab
  passes), color verification (real per-pixel sampling on the detection's
  segmentation mask), Call #2 (LLM or deterministic, toggleable), the motion
  controller's math (uses the real JetAcker's wheelbase/steering numbers from
  HiWonder's official docs), 3D depth (`vision.py`'s `get_real_depth()` --
  uses intrinsic camera matrix with depth camera data).
- **Mocked, clearly labeled in code:** N/A
