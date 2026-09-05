"""
JetAcker VLA Pipeline - vision.py

Author: Arush Kotta
Starting Date: July 26, 2026
Date of Most Recent Update: September 5, 2026

Description: This module runs the actual object detector on a real image, in BOTH
             prompted mode (using Call #1's target names) and default-vocab / "unprompted"
             mode (for broader context, per the dual-pass detection idea), then formats
             results for target_finder.get_delta_auto() (Call #2).

             REAL and working in this file:
                 - Loading and running the detector, both passes.
                 - Bounding box + confidence extraction.
                 - Color attribute verification -- REAL pixel sampling on the detected
                   object's segmentation mask, classified pixel-by-pixel (majority vote,
                   NOT a naive hue average -- averaging breaks on red specifically, since
                   red straddles OpenCV's 0/179 hue wraparound point).
                 - Depth information -- uses camera's intrinsic matrix and depth camera data

                NEW IN V0.2:
                - Switched color-check from bbox-rectangle sampling to segmentation-mask
                  sampling (avoids background pixels inside a bbox, e.g. a ring's hollow
                  center, diluting the color reading).
                - Fixed a real bug: averaging hue values directly breaks for red, since
                  red straddles the hue wraparound point. Switched to per-pixel
                  classification + majority vote.
                - Prompted-pass detector prompt now uses the fused "attribute name"
                  phrase (e.g. "red ring") when a color was requested, since bare nouns
                  alone sometimes missed targets that the fused phrase caught.
"""


#============================== LIBRARY IMPORTS ======================================

#The third-party imports:
import cv2 #This handles image reading, color conversion, and mask resizing
import numpy as np #This handles pixel arrays for the color-check


#================================ DETECTOR SETUP ======================================

#Change this one line to switch detectors. "yoloworld" doesn't need YOLOE's extra CLIP
#package install, and had a slightly higher hit rate in earlier testing.
DETECTOR = "yoloe"

if DETECTOR == "yoloe":
    from ultralytics import YOLOE as _Model
    PROMPTED_CHECKPOINT = "yoloe-11l-seg.pt"
    DEFAULT_VOCAB_CHECKPOINT = "yoloe-11l-seg-pf.pt"

elif DETECTOR == "yoloworld":
    from ultralytics import YOLOWorld as _Model
    PROMPTED_CHECKPOINT = "yolov8l-worldv2.pt"
    DEFAULT_VOCAB_CHECKPOINT = "yolov8l-worldv2.pt" #Same checkpoint -- default vocab = skip set_classes()

else:
    raise ValueError(f"Unknown DETECTOR: {DETECTOR}")

_prompted_model = None #Lazily loaded on first use, cached after that
_default_model = None


#This function lazily loads and caches the prompted-mode model
def _get_prompted_model():
    global _prompted_model
    if _prompted_model is None:
        _prompted_model = _Model(PROMPTED_CHECKPOINT)
    return _prompted_model


#This function lazily loads and caches the default-vocab-mode model
def _get_default_model():
    global _default_model
    if _default_model is None:
        _default_model = _Model(DEFAULT_VOCAB_CHECKPOINT)
    return _default_model


#================================ COLOR VERIFICATION ===================================

#These hue-bucket thresholds are on OpenCV's 0-179 hue scale. Good enough as a first
#pass -- worth tuning against real captured lighting once available.
COLOR_HUE_RANGES = {
    "red":    [(0, 10), (170, 179)],
    "orange": [(11, 22)],
    "yellow": [(23, 34)],
    "green":  [(35, 85)],
    "blue":   [(86, 125)],
    "purple": [(126, 155)],
    "pink":   [(156, 169)]
}


#This function samples pixels either inside a segmentation mask (preferred, avoids
#background pixels diluting the reading) or a bbox rectangle (fallback), classifies
#EACH PIXEL individually into a color bucket, and takes the majority vote. Returns
#True/False, or None if no color was requested for this target.
def _color_check(image, bbox_xyxy, requested_color, mask = None):
    if requested_color is None:
        return None

    if mask is not None:
        ys, xs = mask.nonzero()
        if len(ys) == 0:
            return False
        pixels = image[ys, xs] #This is an Nx3 array of just the object's actual pixels

    else:
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        crop = image[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return False
        pixels = crop.reshape(-1, 3)

    hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hues = hsv_pixels[:, 0].astype(int)
    sats = hsv_pixels[:, 1]
    vals = hsv_pixels[:, 2]

    #====== Classifying each pixel, then taking the majority vote ======

    #NOTE: this is a majority vote, NOT an average hue. Averaging hue directly is wrong
    #for colors like red, which straddle the 0/179 hue wraparound -- pixels split
    #between "near 0" and "near 179" would average out to ~90 (looks like green/blue).
    #This was a real bug caught while testing this on an actual red object.
    bucket_counts = {name: 0 for name in COLOR_HUE_RANGES}
    bucket_counts["gray/black/white"] = 0

    for hue, sat, val in zip(hues, sats, vals):
        if sat < 40 or val < 40: #Low saturation/value = not a strong hue
            bucket_counts["gray/black/white"] += 1
            continue

        for name, ranges in COLOR_HUE_RANGES.items():
            if any(lo <= hue <= hi for lo, hi in ranges):
                bucket_counts[name] += 1
                break

    detected_color = max(bucket_counts, key = bucket_counts.get)

    return detected_color == requested_color.lower()


#================================= DEPTH (MOCKED) ======================================

#*** MOCK -- placeholder only, pending the real depth camera module ***
#Returns a fake (x, y, z) in meters, loosely derived from bbox position and size in the
#2D image (bigger box = assumed closer). NOT real depth data.
def get_mock_depth(bbox_xyxy, image_shape):
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = image_shape[:2]

    box_area_frac = ((x2 - x1) * (y2 - y1)) / (w * h)
    mock_depth_z = max(0.3, 2.0 - box_area_frac * 10) #Bigger box -> assumed closer

    mock_x = mock_depth_z #Crude "forward" stand-in
    mock_y = ((cx - w / 2) / w) * mock_depth_z #Crude left/right stand-in
    mock_z = 0.0

    return (round(mock_x, 3), round(mock_y, 3), round(mock_z, 3))

#*** REAL - Will work given that intrinsic matrix values are accurate ***
#TODO (whoever owns the depth camera work): replace this dictionary with real values
# from the car(should be found in /camera_info topic or similar).
INTRINSIC_MATRIX_VALUES = {
    "fx": 545.1777954101562,
    "fy": 545.1777954101562,
    "cx": 325.6365051269531,
    "cy": 237.0912322998047
}

def get_real_depth(bbox_xyxy, z):
    x1, y1, x2, y2 = bbox_xyxy

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    x = z * (cx - INTRINSIC_MATRIX_VALUES["cx"]) / INTRINSIC_MATRIX_VALUES["fx"]
    y = z * (cy - INTRINSIC_MATRIX_VALUES["cy"]) / INTRINSIC_MATRIX_VALUES["fy"]
    return (x/1000, y/1000, z/1000) #Convert mm to m

#================================== MAIN ENTRY POINT ===================================

#This function runs BOTH prompted and default-vocab passes on one real image, and
#returns a list of detection dicts matching what target_finder.get_delta_auto() expects
def run_vision(image_arr, depth_image_arr, targets, device = "cpu", image_path = None):
    use_depth_img = depth_image_arr is not None
    if image_arr is None:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
    else:
        image = image_arr

    detections = []

    #====== Prompted pass ======

    #This uses the FUSED "attribute name" phrase (e.g. "red ring") as the actual detector
    #prompt when an attribute was requested -- earlier testing showed this sometimes hits
    #when the bare noun alone doesn't. The color-check below still runs as an independent
    #verification step either way, so an unverified color claim is never blindly trusted.
    prompt_names = []
    name_by_prompt = {} #This maps the detector prompt string back to the plain object name
    attribute_by_prompt = {}

    for t in targets:
        if t.get("attribute"):
            fused = f"{t['attribute']} {t['name']}"
            prompt_names.append(fused)
            name_by_prompt[fused] = t["name"]
            attribute_by_prompt[fused] = t["attribute"]
        else:
            prompt_names.append(t["name"])
            name_by_prompt[t["name"]] = t["name"]
            attribute_by_prompt[t["name"]] = None

    model_p = _get_prompted_model()
    if DETECTOR == "yoloe":
        text_pe = model_p.get_text_pe(prompt_names)
        model_p.set_classes(prompt_names, text_pe)
    else:
        model_p.set_classes(prompt_names)

    results_p = model_p.predict(image, device = device, conf=0.1, verbose = False)[0]
    has_masks = results_p.masks is not None

    if results_p.boxes is not None:
        for i, box in enumerate(results_p.boxes):
            detector_label = results_p.names[int(box.cls[0])] #e.g. "red ring"
            plain_name = name_by_prompt.get(detector_label, detector_label)
            requested_attr = attribute_by_prompt.get(detector_label)
            bbox = box.xyxy[0].tolist()

            mask = None
            if has_masks:
                mask_arr = results_p.masks.data[i].cpu().numpy()
                mask = cv2.resize(mask_arr, (image.shape[1], image.shape[0])) > 0.5

            attr_match = _color_check(image, bbox, requested_attr, mask = mask) #Independent verification


            if use_depth_img:
                #TODO: Need to research, does the mask give the coordinates of the object in the image? If so, that may
                #be more accurate than simply using the center of the bounding box(for depth).
                x1, y1, x2, y2 = bbox

                if mask is not None:
                    valid_coords = mask * depth_image_arr
                    rows, cols = np.nonzero(valid_coords)
                    cx, cy = cols[-1], rows[-1] #Closest to ground
                else:
                    rows, cols = np.nonzero(depth_image_arr)
                    coords = np.vstack((rows, cols))
                    valid_coords = coords[:, (rows > y1 & rows < y2) & (cols > x1 & cols < x2)]
                    cy, cx = valid_coords[:, -1]

                cx = min(max(0, cx), len(depth_image_arr[0])-1)
                cy = min(max(0, cy), len(depth_image_arr)-1)
                depth_z = depth_image_arr[cy, cx] 
                x, y, z = get_real_depth(bbox, depth_z)
            else:
                x, y, z = get_mock_depth(bbox, image.shape) #MOCK -- see docstring above

            x, y, z = z, -x, y #Since x should be forward/depth, y should be left(pos)/right(neg), and z should be up down 

            detections.append({
                "name": plain_name,
                "attribute_match": attr_match,
                "x": x, "y": y, "z": z, 
                "confidence": round(float(box.conf[0]), 3),
                "source": "prompted"
            })

    #====== Default-vocab / "unprompted" pass ======

    model_d = _get_default_model()
    results_d = model_d.predict(image, device = device, verbose = False)[0]

    if results_d.boxes is not None:
        for box in results_d.boxes:
            name = results_d.names[int(box.cls[0])]
            bbox = box.xyxy[0].tolist()

            # Will need to be updated, see above
            if use_depth_img:
                x1, y1, x2, y2 = bbox
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cx = min(max(0, cx), len(depth_image_arr[0])-1)
                cy = min(max(0, cy), len(depth_image_arr)-1)
                depth_z = depth_image_arr[cy, cx]
                x, y, z = get_real_depth(bbox, depth_z)
            else:
                x, y, z = get_mock_depth(bbox, image.shape) #MOCK -- see docstring above

            x, y, z = z, -x, y #see above
            #Commented out for now due double detection difficulties
            '''detections.append({
                "name": name,
                "attribute_match": None, #Nothing requested to verify in this pass
                "x": x, "y": y, "z": z,
                "confidence": round(float(box.conf[0]), 3),
                "source": "default_vocab"
            })'''

    return detections
