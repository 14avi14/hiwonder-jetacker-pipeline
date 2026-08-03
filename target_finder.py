"""
JetAcker VLA Pipeline - target_finder.py

Author: Arush Kotta
Starting Date: July 26, 2026
Date of Most Recent Update: August 2, 2026

Description: This module is the "brain" layer of the pipeline. It answers two questions:
                 1. Call #1 - Given a task instruction, what object name(s) should the
                              vision model (YOLOE / YOLO-World) go search for?
                 2. Call #2 - Given the vision model's detections plus the car's current
                              state and physical capabilities, where should the car go
                              next (a delta target), or should it stop/wait?

                 Each call has TWO implementations:
                     a) A real LLM-backed version (get_object_targets / get_delta),
                        which calls out to call_llm() -- currently wired to Anthropic.
                     b) A deterministic, no-LLM fallback (extract_simple_target /
                        extract_simple_delta), for simple single-object-type milestones
                        (e.g. "drive to the red ball") where an LLM call is overkill.

                 Two flags near the top (USE_LLM_FOR_CALL1 / USE_LLM_FOR_CALL2) decide
                 which implementation actually runs. Everything downstream calls
                 get_targets() / get_delta_auto() -- the dispatcher functions -- so the
                 rest of the pipeline never needs to know or care which mode is active.

                NEW IN V0.3:
                - Added extract_simple_delta() as a deterministic fallback for Call #2.
                - Added get_targets() / get_delta_auto() dispatcher functions + the two
                  USE_LLM_FOR_CALLx toggle flags.
                - Wired call_llm() to a real Anthropic API implementation.
"""


#============================== LIBRARY IMPORTS ======================================

#The standard library imports:
import json #This parses/dumps the JSON payloads sent to and received from the LLM
import math #This is used for the distance check in extract_simple_delta
import os #This reads the DEBUG env var
import re #This strips markdown fences out of raw LLM text before parsing


#Debug flag and helper:
DEBUG_MODE = os.getenv("PIPELINE_DEBUG", "false").lower() == "true"

def dprint(*args, **kwargs): #Debug Printer
    if DEBUG_MODE:
        print(*args, **kwargs)


#============================== MODE TOGGLES ==========================================

#These decide whether each call actually hits the LLM, or uses the deterministic
#fallback instead. Independent per call since they're independent decisions -- Call #1
#might not need the LLM yet (simple single-object-type milestones), while Call #2 might
#already be worth keeping real.
USE_LLM_FOR_CALL1 = False #False = extract_simple_target() (no LLM)
USE_LLM_FOR_CALL2 = True  #True  = get_delta() (real LLM call)


#============================== LLM CONNECTION ========================================

#This is the only function that needs to change if you swap providers. Everything else
#in this file (prompt construction, JSON parsing, validation) is provider-agnostic.
def call_llm(system_prompt, user_content):
    """
    Real implementation using the Anthropic API. Requires:
      pip install anthropic
      export ANTHROPIC_API_KEY=your_key_here

    Swap this out for a different provider (OpenAI, Cerebras, local model, etc.) by
    replacing the body below -- the function signature and return type (raw text
    string, expected to contain JSON) stay the same either way.
    """
    import anthropic #This import is local so the rest of the file works without it installed

    client = anthropic.Anthropic() #This reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model = "claude-sonnet-4-6",
        max_tokens = 500,
        system = system_prompt,
        messages = [{"role": "user", "content": json.dumps(user_content)}]
    )

    dprint("DEBUG call_llm RAW RESPONSE: ", response.content[0].text) #Debug

    return response.content[0].text


#This function pulls a JSON object out of raw LLM text, tolerating markdown fences or
#stray prose the model might add despite being told not to
def _extract_json(raw_text):
    text = raw_text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError as e:
        dprint("DEBUG _extract_json FIRST PARSE FAILED: ", e) #Debug

        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"LLM did not return valid JSON. Raw output:\n{raw_text}") from e


#============================ CALL #1 - DETERMINISTIC FALLBACK ========================

#Known color words we check the instruction for -- used by extract_simple_target below
KNOWN_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "white", "black"]


#This function pulls a color word (if any) out of the instruction and always searches
#for object_name, regardless of what the instruction literally says the object is
def extract_simple_target(instruction, object_name = "cube"):
    instruction_lower = instruction.lower()

    found_color = None
    for color in KNOWN_COLORS: #This checks each known color against the instruction text
        if color in instruction_lower:
            found_color = color
            break

    dprint("DEBUG extract_simple_target FOUND COLOR: ", found_color) #Debug

    return {"targets": [{"name": object_name, "attribute": found_color}]}


#This is a convenience wrapper for the original cube-only milestone
def extract_cube_target(instruction):
    return extract_simple_target(instruction, object_name = "cube")


#================================ CALL #1 - REAL LLM ===================================

CALL1_SYSTEM_PROMPT = """You are the object-identification stage of a robot vision pipeline. Your
job is to convert a task instruction into a short list of concrete object
names that an open-vocabulary object detector will search for in a camera
frame.

Rules:
- Output plain, concrete object names or short 2-3 word noun phrases only.
  Prefer bare nouns ("cube") over compound descriptive phrases when a bare
  noun would still let the target be distinguished later using the color
  field separately provided.
- Never use negation ("not X"), comparison ("the reddest"), or vague
  categories ("food") as a detector target -- resolve these into concrete
  candidate object names yourself. If the instruction implies a category,
  expand it into a short list of specific real-world object names that fall
  in that category.
- If the instruction specifies an attribute (e.g. a color), include it
  as a separate "attribute" field rather than folding it into the object
  name string, unless the object name is more reliably found as a single
  fused phrase. Default to separating color from noun.
- Output between 1 and 5 target object names. Do not pad the list with
  unrelated objects.
- Output ONLY valid JSON matching this schema, nothing else:
{"targets": [{"name": "<string>", "attribute": "<string or null>"}]}
"""


#This function calls the LLM to turn an instruction into a list of detector-ready
#object names -- returns {"targets": [{"name": ..., "attribute": ...}, ...]}
def get_object_targets(instruction, known_object_vocabulary_hint = None):
    user_content = {"instruction": instruction}
    if known_object_vocabulary_hint: #This is optional context -- see docstring in the LLM framework doc
        user_content["known_object_vocabulary_hint"] = known_object_vocabulary_hint

    raw = call_llm(CALL1_SYSTEM_PROMPT, user_content)
    parsed = _extract_json(raw)

    #====== Validating the response shape ======

    if "targets" not in parsed or not isinstance(parsed["targets"], list):
        raise ValueError(f"Call #1 response missing valid 'targets' list: {parsed}")

    if not (1 <= len(parsed["targets"]) <= 5):
        raise ValueError(f"Call #1 returned {len(parsed['targets'])} targets, expected 1-5: {parsed}")

    for t in parsed["targets"]:
        if "name" not in t or not isinstance(t["name"], str) or not t["name"].strip():
            raise ValueError(f"Call #1 target missing valid 'name': {t}")
        t.setdefault("attribute", None)

    return parsed


#============================ CALL #2 - DETERMINISTIC FALLBACK ========================

#This function is the no-LLM stand-in for get_delta(). For simple "go to the [color]
#[object]" instructions, deciding which detection to move toward doesn't need
#reasoning -- filter by requested name + verified color match, pick the
#highest-confidence survivor, done.
def extract_simple_delta(call1_result, detections, car_capabilities):
    targets_by_name = {t["name"]: t for t in call1_result.get("targets", [])}

    #====== Filtering down to valid candidates ======

    candidates = []
    for d in detections:
        if d.get("source") != "prompted": #Default-vocab detections are context, not valid targets
            continue

        spec = targets_by_name.get(d["name"])
        if spec is None:
            continue

        if spec.get("attribute") is not None and d.get("attribute_match") is not True:
            continue #A color was requested but never verified on this detection

        candidates.append(d)

    dprint("DEBUG extract_simple_delta CANDIDATES: ", candidates) #Debug

    if not candidates:
        return {
            "status": "target_not_found",
            "delta": None,
            "target_name": None,
            "target_attribute": None,
            "reasoning": "No prompted detection matched the requested name/attribute with a verified color check."
        }

    #====== Picking the best candidate ======

    best = max(candidates, key = lambda d: d["confidence"])
    distance = math.hypot(best["x"], best["y"])
    min_stop = car_capabilities.get("min_stopping_distance_m", 0.10)

    if distance <= min_stop:
        return {
            "status": "target_reached",
            "delta": None,
            "target_name": best["name"],
            "target_attribute": targets_by_name.get(best["name"], {}).get("attribute"),
            "reasoning": f"Target within stopping distance ({distance:.2f}m <= {min_stop}m)."
        }

    return {
        "status": "moving",
        "delta": {"dx": best["x"], "dy": best["y"]},
        "target_name": best["name"],
        "target_attribute": targets_by_name.get(best["name"], {}).get("attribute"),
        "reasoning": f"Deterministic pick: highest-confidence verified match (confidence={best['confidence']})."
    }


#================================ CALL #2 - REAL LLM ===================================

CALL2_SYSTEM_PROMPT = """You are the navigation-decision stage of a robot control pipeline. You will
be given: the original task instruction, a list of detected objects with
their 3D positions already expressed in the car's own navigation frame
(x = forward/back, y = left/right, z = up/down, in meters, relative to the
car's current position), the car's current state, and the car's physical
capabilities and dimensions.

Your job is to decide the next delta target for the car to move toward, or
declare an explicit failure/waiting state if the task cannot currently be
completed.

Rules:
- Only reason about objects actually present in the detections list. Do not
  assume an object exists if it is not listed.
- If multiple detections could match the instruction, pick the best match
  and explain your choice briefly in the "reasoning" field.
- If no detection matches the instruction, or the match is low-confidence,
  set "status" to "target_not_found" and leave "delta" null. Do not
  fabricate a position.
- Respect the car's physical capabilities: do not output a delta that
  requires exceeding max velocity, exceeds the car's min turning radius,
  or would place the target closer than the car's minimum safe stopping
  distance.
- If the target is already within the car's stopping distance, set
  "status" to "target_reached" and "delta" to null.
- Output ONLY valid JSON matching this schema, nothing else:
{"status": "moving|target_reached|target_not_found|ambiguous",
 "delta": {"dx": <number or null>, "dy": <number or null>},
 "target_name": "<string or null>",
 "target_attribute": "<string or null>",
 "reasoning": "<string>"}
"""

VALID_STATUSES = {"moving", "target_reached", "target_not_found", "ambiguous"}


#This function calls the LLM to decide the next delta target (or a stop/wait status)
#given the current detections and car state -- returns a dict matching the schema above
def get_delta(instruction, detections, car_state, car_capabilities):
    user_content = {
        "instruction": instruction,
        "detections": detections,
        "car_state": car_state,
        "car_capabilities": car_capabilities
    }

    raw = call_llm(CALL2_SYSTEM_PROMPT, user_content)
    parsed = _extract_json(raw)

    #====== Validating the response shape ======

    if parsed.get("status") not in VALID_STATUSES:
        raise ValueError(f"Call #2 returned invalid status: {parsed.get('status')}")

    if parsed["status"] == "moving":
        delta = parsed.get("delta")
        if not isinstance(delta, dict) or "dx" not in delta or "dy" not in delta:
            raise ValueError(f"Call #2 status='moving' but delta is missing/invalid: {parsed}")
        if delta["dx"] is None or delta["dy"] is None:
            raise ValueError(f"Call #2 status='moving' but delta contains null values: {parsed}")
    else:
        parsed["delta"] = None

    parsed.setdefault("reasoning", "")

    return parsed


#================================== DISPATCHERS ========================================

#These are what the rest of the pipeline should call -- NOT the individual LLM /
#deterministic functions directly. They check the USE_LLM_FOR_CALLx flags above and
#route accordingly, so run_pipeline.py etc. don't need to know which mode is active.

#This function dispatches Call #1 to either the LLM or deterministic version
def get_targets(instruction, object_name = "cube", known_object_vocabulary_hint = None):
    if USE_LLM_FOR_CALL1:
        return get_object_targets(instruction, known_object_vocabulary_hint)
    else:
        return extract_simple_target(instruction, object_name = object_name)


#This function dispatches Call #2 to either the LLM or deterministic version
def get_delta_auto(instruction, call1_result, detections, car_state, car_capabilities):
    if USE_LLM_FOR_CALL2:
        return get_delta(instruction, detections, car_state, car_capabilities)
    else:
        return extract_simple_delta(call1_result, detections, car_capabilities)
