"""
JetAcker VLA Pipeline - run_pipeline_manual.py

Author: Arush Kotta
Starting Date: August 1, 2026
Date of Most Recent Update: August 2, 2026

Description: Same as run_pipeline.py for Stages 1-2 (REAL, no changes there), but Call
             #2 doesn't need an API key here -- instead it prints a ready-to-paste block
             for you to run manually in any chat GUI (Claude.ai, ChatGPT, Gemini,
             whatever), then waits for you to paste the response back in and
             validates/parses it using the exact same logic get_delta() would use for a
             real API call.

             RUN:
                 python3 run_pipeline_manual.py
"""


#============================== LIBRARY IMPORTS ======================================

#The standard library imports:
import json #This pretty-prints the JSON at each stage and builds the paste block
import math #This is used for the CAR_CAPABILITIES calculation below

#The local module imports:
import target_finder
import vision


#============================== CAR STATE / CAPABILITIES ==============================

STARTING_CAR_STATE = {
    "velocity": 0.0,
    "heading_deg": 0.0,
    "position_xy": [0.0, 0.0]
}

CAR_CAPABILITIES = {
    #====== REAL, from HiWonder's official JetAcker motion control docs ======
    "wheelbase_m": 0.216,
    "track_width_m": 0.195,
    "wheel_diameter_m": 0.097,
    "max_steering_angle_deg": 37,
    "max_velocity_mps": 0.6,

    #====== CALCULATED from the real numbers above ======
    "min_turn_radius_m": round(0.216 / math.tan(math.radians(37)), 3),

    #====== ESTIMATED / design choices, NOT from spec ======
    "min_velocity_mps": 0.05,
    "min_stopping_distance_m": 0.10,

    #====== PLACEHOLDER, NOT MEASURED -- for future gap/corridor reasoning ======
    #Currently INERT: no perception logic measures gap width yet.
    "length_m": 0.30, #GUESS, not measured
    "width_m": 0.18,  #GUESS, not measured
    "height_m": 0.15  #GUESS, not measured
}


#================================= INPUT HELPER ========================================

#This function reads lines until the user types a blank line by itself, or EOF (Ctrl+D)
def read_multiline_input(prompt):
    print(prompt)
    lines = []

    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == "" and lines:
            break

        lines.append(line)

    return "\n".join(lines)


#=================================== MAIN ROUTINE ======================================

def main():
    print("=" * 70)
    print("MANUAL-MODE pipeline run (no API key needed -- Call #2 via GUI copy/paste)")
    print("=" * 70)

    instruction = input("\nEnter instruction (e.g. 'drive to the red ball'): ").strip()
    object_name = input("Enter object type to search for (e.g. 'ball'): ").strip() or "cube"
    image_path = input("Enter path to a real photo to test against: ").strip().strip('"').strip("'")

    #====== Stage 1: extraction ======

    print("\n--- Stage 1: extract_simple_target() [REAL, no LLM call] ---")
    call1_result = target_finder.extract_simple_target(instruction, object_name = object_name)
    print(json.dumps(call1_result, indent = 2))

    #====== Stage 2: vision (REAL detector, both passes) ======

    print("\n--- Stage 2: vision.run_vision() [REAL detector, x/y/z MOCKED] ---")
    detections = vision.run_vision(None, None, call1_result["targets"], device = "cpu", image_path=image_path)
    print(f"{len(detections)} total detections (prompted + default_vocab)")

    prompted = [d for d in detections if d["source"] == "prompted"]
    print(f"  {len(prompted)} from the PROMPTED pass:")
    for d in prompted:
        print(f"    {d}")

    if not prompted:
        print("\n  No prompted detections found -- expect 'target_not_found' from Call #2.")

    #====== Stage 3: Call #2, via GUI copy/paste ======

    user_content = {
        "instruction": instruction,
        "detections": detections,
        "car_state": STARTING_CAR_STATE,
        "car_capabilities": CAR_CAPABILITIES
    }

    paste_block = (
        target_finder.CALL2_SYSTEM_PROMPT.strip()
        + "\n\nNow process this input and respond with ONLY the JSON:\n"
        + json.dumps(user_content, indent = 2)
    )

    print("\n" + "=" * 70)
    print("COPY EVERYTHING BELOW THIS LINE, paste it into a new chat in Claude/ChatGPT/Gemini:")
    print("=" * 70)
    print(paste_block)
    print("=" * 70)
    print("COPY EVERYTHING ABOVE THIS LINE")
    print("=" * 70)

    raw_response = read_multiline_input(
        "\nPaste the LLM's JSON response below, then press Enter on a blank line to continue:"
    )

    #====== Parsing + validating the pasted response ======

    print("\n--- Parsing + validating the pasted response ---")
    try:
        parsed = target_finder._extract_json(raw_response)

        if parsed.get("status") not in target_finder.VALID_STATUSES:
            raise ValueError(f"Invalid status: {parsed.get('status')}")

        if parsed["status"] == "moving":
            delta = parsed.get("delta")
            if not isinstance(delta, dict) or delta.get("dx") is None or delta.get("dy") is None:
                raise ValueError(f"status='moving' but delta is missing/invalid: {parsed}")
        else:
            parsed["delta"] = None

        parsed.setdefault("reasoning", "")

        print("VALID. Call #2 result:")
        print(json.dumps(parsed, indent = 2))

    except ValueError as e:
        print(f"INVALID RESPONSE: {e}")
        print("The LLM's output didn't match the expected schema -- try again,")
        print("or check whether it added extra prose/markdown despite instructions not to.")

    print("\n" + "=" * 70)
    print("Done. Reminder: x/y/z above are MOCKED (bbox-derived placeholders),")
    print("not real depth camera data. Stages 1-2 were real; stage 3 was a real")
    print("LLM response, just obtained manually instead of via API.")
    print("=" * 70)


if __name__ == "__main__":
    main()
