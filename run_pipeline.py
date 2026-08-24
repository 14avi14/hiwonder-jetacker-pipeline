"""
JetAcker VLA Pipeline - run_pipeline.py

Author: Arush Kotta
Starting Date: July 26, 2026
Date of Most Recent Update: August 2, 2026

Description: This is the main REAL end-to-end runner. Asks for an instruction, an
             object type, and an image path, then runs the full chain:
                 1. target_finder.get_targets()      -- Call #1 (LLM or deterministic,
                                                          see USE_LLM_FOR_CALL1)
                 2. vision.run_vision()               -- REAL YOLOE / YOLO-World, both
                                                          prompted and default-vocab
                                                          passes, on the real photo
                 3. target_finder.get_delta_auto()    -- Call #2 (LLM or deterministic,
                                                          see USE_LLM_FOR_CALL2)
                 4. path_planner.MotionController     -- REAL controller logic, car
                                                          movement SIMULATED since no
                                                          real odometry is connected yet

             *** MOCKED / SIMULATED (clearly labeled at runtime) ***
                 - x, y, z depth values from vision.py's get_mock_depth().
                 - Car movement in Stage 4 -- the controller math is real, but there's
                   no real car to move, so a crude mock movement model fakes ticking
                   odometry forward just to show the controller responding.

             SETUP:
                 pip install anthropic ultralytics opencv-python torch --break-system-packages
                 export ANTHROPIC_API_KEY=your_key_here

                 (If using YOLOE, also: pip install git+https://github.com/ultralytics/CLIP.git --break-system-packages)

             RUN:
                 python3 run_pipeline.py

                NEW IN V0.2:
                - Added Stage 4 -- wires path_planner.MotionController into the run,
                  closing the gap between Call #2's delta and an actual drive command.
"""


#============================== LIBRARY IMPORTS ======================================

#The standard library imports:
import json #This pretty-prints the JSON at each stage
import math #This is used by the Stage 4 movement simulation
import socket #For transfering information between car and server
import cv2 #For visualizing incoming image

#The local module imports:
import target_finder
import vision
import socket_ops


#============================== CAR STATE / CAPABILITIES ==============================

#Starting state -- zeroed since we're not connected to real odometry yet. On the real
#car, replace this with a live read from /odom or /odom_raw each cycle
#(nav_msgs/msg/Odometry -- gives position + heading directly).
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
    "max_velocity_mps": 0.6, #Docs: keep linear.x within -0.6 to 0.6

    #====== CALCULATED from the real numbers above ======
    #Not directly published, but a straightforward derivation:
    #R = wheelbase / tan(max_steering)
    "min_turn_radius_m": round(0.216 / math.tan(math.radians(37)), 3),

    #====== ESTIMATED / our own design choices, NOT from any spec ======
    #Untested on real hardware -- treat these as starting guesses to tune.
    "min_velocity_mps": 0.05, #Arbitrary minimum crawl speed
    "min_stopping_distance_m": 0.10, #Arbitrary safety margin

    #====== PLACEHOLDER, NOT MEASURED -- forward-looking only ======
    #For "can the car fit through this gap / turn in this corridor" reasoning.
    #Currently INERT: nothing in vision.py measures gap or corridor width yet, so even
    #accurate numbers here have nothing to compare against. Two separate todos before
    #this is real:
    #    1. Measure the actual chassis with a tape measure (no official spec sheet
    #       found online) and replace the guesses below.
    #    2. Build the perception logic that outputs a gap/clear-width measurement for
    #       the LLM to compare width_m against.
    "length_m": 0.30, #GUESS, not measured
    "width_m": 0.18,  #GUESS, not measured
    "height_m": 0.15  #GUESS, not measured
}

USE_DEPTH = True #Need intrinsic matrix -- see vision.py


#=================================== MAIN ROUTINE ======================================

def main():
    print("=" * 70)
    print("REAL end-to-end pipeline run")
    print("=" * 70)

    print(f"\nMode: Call #1 = {'LLM' if target_finder.USE_LLM_FOR_CALL1 else 'deterministic'}, "
          f"Call #2 = {'LLM' if target_finder.USE_LLM_FOR_CALL2 else 'deterministic'}")

    instruction = input("\nEnter instruction (e.g. 'drive to the red ball'): ").strip()
    object_name = input("Enter object type to search for (e.g. 'ball', 'cube'): ").strip() or "cube"
    #image_path = input("Enter path to a real photo to test against: ").strip().strip('"').strip("'")

    #====== Stage 0: Preparing for message from car =====
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", socket_ops.PORT))

    server.listen(1) #Max of 1 connect request
    print("Server started listening")

    server.settimeout(15.0) #Time out if any socket operation takes too long

    try:
        #Accept connection
        (conn, conn_addr) = server.accept()
        print("Got connnection from", conn_addr)

        #Check incoming message type. 
        #  1. msg_type=socket_ops.CLOSE_CONNECTION_MSG -> ending operations
        #  2. msg_type=socket_ops.INITIAL_PREDICTION_MSG -> will send images(bgr+pos) for get initial delta position prediction
        #==== FUTURE IDEAS ====
        #  3. msg_type=socket_ops.OBSTACLE -> will send image along with delta from intitial position 
        #                                     and last target, and distance of object detected by LiDAR in front
        msg_type = socket_ops.recieve_string(conn)

        print("[INCOMING MESSAGE TYPE]", msg_type)

        if msg_type == socket_ops.CLOSE_CONNECTION_MSG:
            print("Closing connection...")
            conn.close()

        elif msg_type == socket_ops.INITIAL_PREDICTION_MSG:
            bgr_image_arr = socket_ops.recieve_image(conn) #incoming image (3D)
            depth_image_arr = socket_ops.recieve_image(conn)

            if not USE_DEPTH:
                depth_image_arr = None

            #Quick visualization of image
            cv2.imshow("img-recieved", bgr_image_arr)
            cv2.waitKey(0) #Close window to continue
            cv2.destroyAllWindows()

            #====== Stage 1: extraction ======

            print(f"\n--- Stage 1: get_targets() [{'LLM' if target_finder.USE_LLM_FOR_CALL1 else 'deterministic, no LLM'}] ---")
            call1_result = target_finder.get_targets(instruction, object_name = object_name)
            print(json.dumps(call1_result, indent = 2))

            #====== Stage 2: vision (REAL detector, both passes) ======

            print("\n--- Stage 2: vision.run_vision() [REAL detector, REAL x/y/z] ---")
            detections = vision.run_vision(bgr_image_arr, depth_image_arr, call1_result["targets"], device = "cpu")
            print(f"{len(detections)} total detections (prompted + default_vocab)")

            prompted = [d for d in detections if d["source"] == "prompted"]
            print(f"  {len(prompted)} from the PROMPTED pass:")
            for d in prompted:
                print(f"    {d}")

            if not prompted:
                print("\n  No prompted detections found -- Call #2 will likely return "
                    "'target_not_found'. This is a real result, not an error.")

            #====== Stage 3: Call #2 ======

            print(f"\n--- Stage 3: get_delta_auto() [{'LLM' if target_finder.USE_LLM_FOR_CALL2 else 'deterministic, no LLM'}] ---")
            call2_result = target_finder.get_delta_auto(
                instruction, call1_result, detections, STARTING_CAR_STATE, CAR_CAPABILITIES
            )
            print(json.dumps(call2_result, indent = 2))

            #===== Stage 4: sending instructions =====
            print("--- Stage 4: Sending instructions to car ---")

            instructions_json = json.dumps(call2_result)
            socket_ops.send_string(conn, instructions_json)

        conn.close() #Close the socket connection

        print("\n" + "=" * 70)
        if not USE_DEPTH:
            print("Done. Reminder: x/y/z from vision.py are MOCKED (bbox-derived placeholders),")
            print("not real depth camera data. Everything else in this run (extraction,")
            print("detection, color-check, and Call #2) was real.")
        else:
            print("Done. Everything in this run was real, including extraction, detection, ")
            print("color-check, and Call #2.")
        print("=" * 70)

    except socket.timeout:
        print("TimeoutError: Could not connect to other server in time")

    server.close() #Close the server

if __name__ == "__main__":
    main()
