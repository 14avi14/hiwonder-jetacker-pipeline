"""
JetAcker VLA Pipeline - test_pipeline.py

Author: Arush Kotta
Starting Date: July 26, 2026
Date of Most Recent Update: August 2, 2026

Description: Runs the full pipeline end-to-end using MOCKED LLM responses and MOCK
             detection data (standing in for the real YOLOE + depth camera output),
             plus a simulated multi-tick run of the motion controller. This lets you
             see and verify the whole chain -- Call #1 -> mock detections -> Call #2 ->
             path planner -- working with zero API keys and zero real hardware needed.

             When ready to go live: implement target_finder.call_llm() for your chosen
             provider (already done, see target_finder.py), and swap MOCK_DETECTIONS
             for real output from vision.py (see run_pipeline.py for that version).
"""


#============================== LIBRARY IMPORTS ======================================

#The standard library imports:
import json #This builds the canned mock LLM responses and pretty-prints output
import math #This drives the Stage 4 movement simulation

#The local module imports:
import target_finder
import path_planner


#================================== MOCK LLM ============================================

#This returns a canned, schema-valid response instead of calling a real API. Swap
#target_finder.call_llm back to the real implementation when testing against a live
#provider -- see run_pipeline.py for that version.
def mock_call_llm(system_prompt, user_content):
    #Only Call #2 goes through the LLM here -- Call #1 is bypassed by
    #target_finder.extract_cube_target() for the current cube milestone.
    return json.dumps({
        "status": "moving",
        "delta": {"dx": 1.2, "dy": -0.3},
        "target_name": "cube",
        "target_attribute": "red",
        "reasoning": "Selected the red cube over the blue cube since the instruction specified red."
    })


#================================== MOCK DATA ==========================================

#This stands in for vision.py's real output (YOLOE + depth camera + the
#attribute_match color-check) until run against a real image
MOCK_DETECTIONS = [
    {"name": "cube", "attribute_match": "red", "x": 1.2, "y": -0.3, "z": 0.0, "confidence": 0.81},
    {"name": "cube", "attribute_match": "blue", "x": 0.9, "y": 0.4, "z": 0.0, "confidence": 0.77}
]

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
    "length_m": 0.30, #GUESS, not measured
    "width_m": 0.18,  #GUESS, not measured
    "height_m": 0.15  #GUESS, not measured
}


#=================================== MAIN ROUTINE ======================================

def main():
    target_finder.call_llm = mock_call_llm #This patches in the mock LLM for this test run

    instruction = "drive to the red cube"
    print(f"Instruction: {instruction!r}\n")

    #====== Call #1 ======

    call1_result = target_finder.extract_cube_target(instruction)
    print("Call #1 output (object targets for detector):")
    print(json.dumps(call1_result, indent = 2))
    print()

    #In a real run, vision.py would take call1_result["targets"], run YOLOE/YOLO-World
    #with those names, do the color-check, and produce MOCK_DETECTIONS-shaped output
    #for real. Skipping straight to mock data here since that part is tested separately
    #in run_pipeline.py.
    print("Using MOCK detections (stand-in for real vision.py output):")
    print(json.dumps(MOCK_DETECTIONS, indent = 2))
    print()

    #====== Call #2 ======

    call2_result = target_finder.get_delta(
        instruction, MOCK_DETECTIONS, STARTING_CAR_STATE, CAR_CAPABILITIES
    )
    print("Call #2 output (delta/status):")
    print(json.dumps(call2_result, indent = 2))
    print()

    #====== Continuous motion controller ======

    controller = path_planner.MotionController(CAR_CAPABILITIES)
    got_target = controller.set_target(call2_result, STARTING_CAR_STATE)
    print(f"Target set in world frame: {controller.target_world_xy} (set_target returned {got_target})")
    print()

    #This simulates a few control ticks with a crude mock car model, just to show the
    #controller re-aiming as the car "moves" -- NOT a real physics sim, just enough to
    #demonstrate compute_control() responding to changing car_state each tick.
    sim_car_state = dict(STARTING_CAR_STATE)
    print("Simulated control ticks (mock odometry, NOT real physics):")

    for tick in range(6):
        speed, steering_angle, status = controller.compute_control(sim_car_state)
        print(f"  tick {tick}: car_state={sim_car_state} -> speed={speed} m/s, "
              f"steer={steering_angle} deg, status={status}")

        if status in ("arrived", "no_target"):
            break

        #This crudely fakes odometry ticking forward -- NOT real physics
        dt = 0.3
        heading_rad = math.radians(sim_car_state["heading_deg"])
        sim_car_state["position_xy"] = [
            sim_car_state["position_xy"][0] + speed * dt * math.cos(heading_rad),
            sim_car_state["position_xy"][1] + speed * dt * math.sin(heading_rad)
        ]
        sim_car_state["heading_deg"] += steering_angle * 0.4
        sim_car_state["velocity"] = speed


if __name__ == "__main__":
    main()
