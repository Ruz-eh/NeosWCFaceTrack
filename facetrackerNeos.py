import copy
import os
from socket import socket
import sys
import argparse
import traceback
import gc
import mediapipe

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
if os.name == 'nt':
    parser.add_argument("-l", "--list-cameras", type=int, help="Set this to 1 to list the available cameras and quit, set this to 2 or higher to output only the names", default=0)
    parser.add_argument("-a", "--list-dcaps", type=int, help="Set this to -1 to list all cameras and their available capabilities, set this to a camera id to list that camera's capabilities", default=None)
    parser.add_argument("-W", "--width", type=int, help="Set camera and raw RGB width", default=800)
    parser.add_argument("-H", "--height", type=int, help="Set camera and raw RGB height", default=600)
    parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=18)
    parser.add_argument("-D", "--dcap", type=int, help="Set which device capability line to use or -1 to use the default camera settings", default=None)
    parser.add_argument("-B", "--blackmagic", type=int, help="When set to 1, special support for Blackmagic devices is enabled", default=0)
else:
    parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=360)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-M", "--mirror-input", action="store_true", help="Process a mirror image of the input video")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking, to 2 to also show face ids, to 3 to add confidence values or to 4 to add numbers to the point display", default=0)
parser.add_argument("-P", "--pnp-points", type=int, help="Set this to 1 to add the 3D fitting points to the visualization", default=0)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=1)
parser.add_argument("--faces", type=int, help="Set the maximum number of faces (slow)", default=1)
parser.add_argument("--scan-retinaface", type=int, help="When set to 1, scanning for additional faces will be performed using RetinaFace in a background thread, otherwise a simpler, faster face detection mechanism is used. When the maximum number of faces is 1, this option does nothing.", default=0)
parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=1)
parser.add_argument("--discard-after", type=int, help="Set the how long the tracker should keep looking for lost faces", default=1)
parser.add_argument("--max-feature-updates", type=int, help="This is the number of seconds after which feature min/max/medium values will no longer be updated once a face has been detected.", default=900)
parser.add_argument("--no-3d-adapt", type=int, help="When set to 1, the 3D face model will not be adapted to increase the fit", default=1)
parser.add_argument("--try-hard", type=int, help="When set to 1, the tracker will try harder to find a face", default=0)
parser.add_argument("--raw-rgb", type=int, help="When this is set, raw RGB frames of the size given with \"-W\" and \"-H\" are read from standard input instead of reading a video", default=0)
parser.add_argument("--log-data", help="You can set a filename to which tracking data will be logged here", default="")
parser.add_argument("--log-output", help="You can set a filename to console output will be logged here", default="")
parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed, except for model 4, which is wink optimized. Models 1 and 0 tend to be too rigid for expression and blink detection. Model -2 is roughly equivalent to model 1, but faster. Model -3 is between models 0 and -1.", default=-2, choices=[-3, -2, -1, 0, 1, 2, 3, 4])
parser.add_argument("--model-dir", help="This can be used to specify the path to the directory containing the .onnx model files", default=None)
parser.add_argument("--gaze-tracking", type=int, help="When set to 1, experimental blink detection and gaze tracking are enabled, which makes things slightly slower", default=1)
parser.add_argument("--face-id-offset", type=int, help="When set, this offset is added to all face ids, which can be useful for mixing tracking data from multiple network sources", default=0)
parser.add_argument("--repeat-video", type=int, help="When set to 1 and a video file was specified with -c, the tracker will loop the video until interrupted", default=0)
parser.add_argument("--dump-points", type=str, help="When set to a filename, the current face 3D points are made symmetric and dumped to the given file when quitting the visualization with the \"q\" key", default="")
parser.add_argument("--smooth-rotation", type=int, help="When set to 1, the 3D face rotation data will be smoothed, reducing jitter considerably but also reducing rotation responsiveness slightly", default=0)
parser.add_argument("--smooth-translation", type=int, help="When set to 1, the 3D face translation data will be smoothed, reducing jitter considerably but also reducing translation responsiveness slightly", default=0)
if os.name == 'nt':
    parser.add_argument("--use-dshowcapture", type=int, help="When set to 1, libdshowcapture will be used for video input instead of OpenCV", default=1)
    parser.add_argument("--blackmagic-options", type=str, help="When set, this additional option string is passed to the blackmagic capture library", default=None)
    parser.add_argument("--priority", type=int, help="When set, the process priority will be changed", default=None, choices=[0, 1, 2, 3, 4, 5])
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.max_threads)

class OutputLog(object):
    def __init__(self, fh, output):
        self.fh = fh
        self.output = output
    def write(self, buf):
        if not self.fh is None:
            self.fh.write(buf)
        self.output.write(buf)
        self.flush()
    def flush(self):
        if not self.fh is None:
            self.fh.flush()
        self.output.flush()
output_logfile = None
if args.log_output != "":
    output_logfile = open(args.log_output, "w")
sys.stdout = OutputLog(output_logfile, sys.stdout)
sys.stderr = OutputLog(output_logfile, sys.stderr)

if os.name == 'nt':
    import dshowcapture
    if args.blackmagic == 1:
        dshowcapture.set_bm_enabled(True)
    if not args.blackmagic_options is None:
        dshowcapture.set_options(args.blackmagic_options)
    if not args.priority is None:
        import psutil
        classes = [psutil.IDLE_PRIORITY_CLASS, psutil.BELOW_NORMAL_PRIORITY_CLASS, psutil.NORMAL_PRIORITY_CLASS, psutil.ABOVE_NORMAL_PRIORITY_CLASS, psutil.HIGH_PRIORITY_CLASS, psutil.REALTIME_PRIORITY_CLASS]
        p = psutil.Process(os.getpid())
        p.nice(classes[args.priority])

if os.name == 'nt' and (args.list_cameras > 0 or not args.list_dcaps is None):
    cap = dshowcapture.DShowCapture()
    info = cap.get_info()
    unit = 10000000.;
    if not args.list_dcaps is None:
        formats = {0: "Any", 1: "Unknown", 100: "ARGB", 101: "XRGB", 200: "I420", 201: "NV12", 202: "YV12", 203: "Y800", 300: "YVYU", 301: "YUY2", 302: "UYVY", 303: "HDYC (Unsupported)", 400: "MJPEG", 401: "H264" }
        for cam in info:
            if args.list_dcaps == -1:
                type = ""
                if cam['type'] == "Blackmagic":
                    type = "Blackmagic: "
                print(f"{cam['index']}: {type}{cam['name']}")
            if args.list_dcaps != -1 and args.list_dcaps != cam['index']:
                continue
            for caps in cam['caps']:
                format = caps['format']
                if caps['format'] in formats:
                    format = formats[caps['format']]
                if caps['minCX'] == caps['maxCX'] and caps['minCY'] == caps['maxCY']:
                    print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
                else:
                    print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']}-{caps['maxCX']}x{caps['maxCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
    else:
        if args.list_cameras == 1:
            print("Available cameras:")
        for cam in info:
            type = ""
            if cam['type'] == "Blackmagic":
                type = "Blackmagic: "
            if args.list_cameras == 1:
                print(f"{cam['index']}: {type}{cam['name']}")
            else:
                print(f"{type}{cam['name']}")
    cap.destroy_capture()
    sys.exit(0)

import numpy as np
import time
import cv2
from mediapipe.python.solutions import hands
import requests
import asyncio
import websockets
import struct
import json
from OneEuroFilter import LowPassFilter, OneEuroFilter
from input_reader import InputReader, VideoReader, DShowCaptureReader, try_int
from tracker import Tracker, get_model_base_path

if args.faces >= 40:
    print("Transmission of tracking data over network is not supported with 40 or more faces.")
 
fps = 0
dcap = None
use_dshowcapture_flag = False
if os.name == 'nt':
    fps = args.fps
    dcap = args.dcap
    use_dshowcapture_flag = True if args.use_dshowcapture == 1 else False
    input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
    if args.dcap == -1 and type(input_reader) == DShowCaptureReader:
        fps = min(fps, input_reader.device.get_fps())
else:
    input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag)
if type(input_reader.reader) == VideoReader:
    fps = 0

log = None
out = None

features = ["eye_l", "eye_r", "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]

if args.log_data != "":
    log = open(args.log_data, "w")
    log.write("Frame,Time,Width,Height,FPS,Face,FaceID,RightOpen,LeftOpen,AverageConfidence,Success3D,PnPError,RotationQuat.X,RotationQuat.Y,RotationQuat.Z,RotationQuat.W,Euler.X,Euler.Y,Euler.Z,RVec.X,RVec.Y,RVec.Z,TVec.X,TVec.Y,TVec.Z")
    for i in range(66):
        log.write(f",Landmark[{i}].X,Landmark[{i}].Y,Landmark[{i}].Confidence")
    for i in range(66):
        log.write(f",Point3D[{i}].X,Point3D[{i}].Y,Point3D[{i}].Z")
    for feature in features:
        log.write(f",{feature}")
    log.write("\r\n")
    log.flush()

is_camera = args.capture == str(try_int(args.capture))

try:
    frame_time = time.perf_counter()
    target_duration = 0
    if fps > 0:
        target_duration = 1. / float(fps)

except KeyboardInterrupt:
    if args.silent == 0:
        print("Quitting")
    input_reader.close()
    if not out is None:
        out.release()
    cv2.destroyAllWindows()



async def facetrack(websocket,path):
    global input_reader
    global args
    global fps
    global use_dshowcapture_flag
    global dcap
    global is_camera
    
    oeMouthWideConfig = {
        'freq': 18,
        'mincutoff': 1.0,
        'beta': 0.7,
        'dcutoff': 1.0,
        }
    
    oeMouthOpenConfig = {
        'freq': 18,
        'mincutoff': 1.0,
        'beta': 0.7,
        'dcutoff': 1.0,
        }
        
    oeEyeBrowUDConfig = {
        'freq': 18,
        'mincutoff': 1.0,
        'beta': 0.5,
        'dcutoff': 1.0,
        }

    MouthOpenF = OneEuroFilter(**oeMouthOpenConfig)
    MouthWideF = OneEuroFilter(**oeMouthWideConfig)
    REyeBrowUDF = OneEuroFilter(**oeEyeBrowUDConfig)
    LEyeBrowUDF = OneEuroFilter(**oeEyeBrowUDConfig)
    LEyeBrowSteepF = OneEuroFilter(**oeEyeBrowUDConfig)
    REyeBrowSteepF = OneEuroFilter(**oeEyeBrowUDConfig)
    LEyeBrowQuirkF = OneEuroFilter(**oeEyeBrowUDConfig)
    
    # Head Rotation Filters
    if (args.smooth_rotation == 1):
        oeHeadRotationConfig = {
            'freq': 18,
            'mincutoff': 1.6,
            'beta': 1.4,
            'dcutoff': 1.0,
            }

        HRotXF = OneEuroFilter(**oeHeadRotationConfig)
        HRotYF = OneEuroFilter(**oeHeadRotationConfig)
        HRotZF = OneEuroFilter(**oeHeadRotationConfig)
        HRotWF = OneEuroFilter(**oeHeadRotationConfig)
    
    # Head Position Filters
    if (args.smooth_translation == 1):
        oeHeadPositionConfig = {
            'freq': 18,
            'mincutoff': 1.1,
            'beta': 0.8,
            'dcutoff': 1.0,
            }
        HPosXF = OneEuroFilter(**oeHeadPositionConfig)
        HPosYF = OneEuroFilter(**oeHeadPositionConfig)
        HPosZF = OneEuroFilter(**oeHeadPositionConfig)
    
    # Gaze Tracking Filters
    oeGazeConfig = {
            'freq': 18,
            'mincutoff': 0.9,
            'beta': 0.9,
            'dcutoff': 1.0,
            }
    GazeLR = OneEuroFilter(**oeGazeConfig)
    GazeUD = OneEuroFilter(**oeGazeConfig)

    LookLR = 0
    LookUD = 0

    lastSocketString = ""
    lastDetected = False
    
    height = 0
    width = 0
    tracker = None
    attempt = 0
    failures = 0
    repeat = args.repeat_video != 0 and type(input_reader.reader) == VideoReader
    source_name = input_reader.name
    need_reinit = 0
    now = time.time()

    handsModule = mediapipe.solutions.hands
    # Finger Constants
    WRIST = handsModule.HandLandmark.WRIST
    MMCP = handsModule.HandLandmark.MIDDLE_FINGER_MCP
    RMCP = handsModule.HandLandmark.RING_FINGER_MCP

    MPIP = handsModule.HandLandmark.MIDDLE_FINGER_PIP
    MTIP = handsModule.HandLandmark.MIDDLE_FINGER_TIP
    
    print("Connection Successful")

    try:
        while 1:
            if not input_reader.is_open() or need_reinit == 1:
                input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
                if input_reader.name != source_name:
                    print(f"Failed to reinitialize camera and got {input_reader.name} instead of {source_name}.")
                    sys.exit(1)
                need_reinit = 2
                continue
            if input_reader.is_ready():
                break

        while 1:
            ret, frame = input_reader.read()
            if ret:
                height, width, _ = frame.shape

                hands = handsModule.Hands(static_image_mode=False, min_detection_confidence=0.75, min_tracking_confidence=0.75, max_num_hands=2)
                tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, try_hard=args.try_hard == 1)
                break
            time.sleep(0.01)

        async for message in websocket:
            ret, frame = input_reader.read()
            if ret and args.mirror_input:
                frame = cv2.flip(frame, 1)
            if not ret:
                if repeat:
                    if need_reinit == 0:
                        need_reinit = 1
                    continue
                elif is_camera:
                    attempt += 1
                    if attempt > 20:
                        break
                    else:
                        time.sleep(0.01)
                        if attempt == 3:
                            need_reinit = 1
                        continue
                else:
                    break
            frame.flags.writeable = False
            attempt, need_reinit, now = 0, 0, time.time()

            try:
                faces = tracker.predict(frame)

                detected = False

                # Empty string to be sent
                socketString = ""

                for face_num, f in enumerate(faces):
                    detected = True
                    f = copy.copy(f)
                    f.id += args.face_id_offset

                    # If the AI loses tracking of a face, when it recovers it it'll return 0.0 on all data for the first frame
                    # This fixes that by sending the last good tracking data instead
                    if (f.euler[0] is None) or f.euler[0] == 0.0:
                        socketString = lastSocketString
                        continue

                    if f.eye_blink is None:
                        f.eye_blink = [1, 1]
                    # if args.silent == 0:
                        # right_state = "O" if f.eye_blink[0] > 0.30 else "-"
                        # left_state = "O" if f.eye_blink[1] > 0.30 else "-"
                        # print(f"Confidence[{f.id}]: {f.conf:.4f} / 3D fitting error: {f.pnp_error:.4f} / Eyes: {left_state}, {right_state}")

                    # Socketstring concatenation for output into neos
                    # Available properties: now f.id width height f.eye_blink[0] f.eye_blink[1]
                    # f.success f.pnp_error f.quaternion[0] f.quaternion[1] f.quaternion[2] f.quaternion[3]
                    # f.euler[0] f.euler[1] f.euler[2] f.translation[0] f.translation[1] f.translation[2]
                    
                    # Add the rotation floatQ
                    if args.smooth_rotation == 1:
                        try:
                            socketString += f"[{HRotXF(f.quaternion[0],now)};{HRotYF(f.quaternion[1],now)};{HRotZF(f.quaternion[2],now)};{HRotWF(f.quaternion[3],now)}],"
                        except:
                            socketString += f"[{f.quaternion[0]};{f.quaternion[1]};{f.quaternion[2]};{f.quaternion[3]}],"
                    else:
                        socketString += f"[{f.quaternion[0]};{f.quaternion[1]};{f.quaternion[2]};{f.quaternion[3]}],"
                    
                    # Add the translation float3
                    if args.smooth_translation == 1:
                        try:
                            socketString += f"[{HPosXF(f.translation[1], now):.6f};{HPosYF(f.translation[0], now):.6f};{HPosZF(f.translation[2], now):.6f}],"
                        except:
                            socketString += f"[{f.translation[1]:.6f};{f.translation[0]:.6f};{f.translation[2]:.6f}],"
                    else:
                        socketString += f"[{f.translation[1]:.6f};{f.translation[0]:.6f};{f.translation[2]:.6f}],"
                    
                    # Add the blink ratio, output the most opened eye's ratio for both if the difference between
                    #eye ratios is less than the threshold, otherwise pass the most opened eye's ratio and
                    #multiply the blinking eye's ratio to make winking possible
                    #if f.eye_blink[1]-f.eye_blink[0] > 0.17:
                    #    socketString += f"[{f.eye_blink[1]:.6f};{0.5*f.eye_blink[0]*f.eye_blink[0]:.6f}],"
                    #elif f.eye_blink[1]-f.eye_blink[0] < -0.17:
                    #    socketString += f"[{0.5*f.eye_blink[1]*f.eye_blink[1]:.6f};{f.eye_blink[0]:.6f}],"
                    #else:
                    #    avgBlink = (f.eye_blink[0] + f.eye_blink[1] + max(f.eye_blink[0],f.eye_blink[1])) / 3
                    #    socketString += f"[{avgBlink:.6f};{avgBlink:.6f}],"
                    
                    # Adds the raw blink ratio
                    socketString += f"[{f.eye_blink[1]:.6f};{f.eye_blink[0]:.6f}],"
                        
                    # Features: ["eye_l", "eye_r", "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]

                    # Add the mouth properties as float2, a positive number means more open
                    socketString += f"[{f.current_features['mouth_open']:.6f};{MouthWideF(f.current_features['mouth_wide'], now):.6f}],"

                    # Add both eyebrow's up/down ratio as float2, a positive number means up
                    socketString += f"[{LEyeBrowUDF(f.current_features['eyebrow_updown_l'], now):.6f};{REyeBrowUDF(f.current_features['eyebrow_updown_r'], now):.6f}],"

                    # Gaze Tracking
                    
                    RGazepoint = (f.pts_3d[66][0],f.pts_3d[66][1])
                    LGazepoint = (f.pts_3d[67][0],f.pts_3d[67][1])
                    RTopRight = (f.pts_3d[37][0],f.pts_3d[37][1])
                    RTopLeft = (f.pts_3d[38][0],f.pts_3d[38][1])
                    RBotRight = (f.pts_3d[41][0],f.pts_3d[41][1])
                    RBotLeft = (f.pts_3d[40][0],f.pts_3d[40][1])
                    LTopRight = (f.pts_3d[43][0],f.pts_3d[43][1])
                    LTopLeft = (f.pts_3d[44][0],f.pts_3d[44][1])
                    LBotRight = (f.pts_3d[47][0],f.pts_3d[47][1])
                    LBotLeft = (f.pts_3d[46][0],f.pts_3d[46][1])
                    
                    RBorderRight = (RTopRight[0] + RBotRight[0]) / 2
                    RBorderLeft = (RTopLeft[0] + RBotLeft[0]) / 2
                    RHorizontalCenter = (RBorderRight + RBorderLeft) / 2
                    RHorizontalRadius = (abs(RHorizontalCenter - RBorderRight) + abs(RBorderLeft - RHorizontalCenter)) / 2
                    RBorderTop = (RTopRight[1] + RTopLeft[1]) / 2
                    RBorderBot = (RBotRight[1] + RBotLeft[1]) / 2
                    RVerticalCenter = (RBorderTop + RBorderBot) / 2
                    RVerticalRadius = (abs(RVerticalCenter - RBorderTop) + abs(RBorderBot - RVerticalCenter)) / 2
                    RLookLR = (RGazepoint[0] - RHorizontalCenter) / RHorizontalRadius
                    RLookUD = (RGazepoint[1] - RVerticalCenter) / RVerticalRadius
                    LBorderRight = (LTopRight[0] + LBotRight[0]) / 2
                    LBorderLeft = (LTopLeft[0] + LBotLeft[0]) / 2
                    LHorizontalCenter = (LBorderRight + LBorderLeft) / 2
                    LHorizontalRadius = (abs(LHorizontalCenter - LBorderRight) + abs(LBorderLeft - LHorizontalCenter)) / 2
                    LBorderTop = (LTopRight[1] + LTopLeft[1]) / 2
                    LBorderBot = (LBotRight[1] + LBotLeft[1]) / 2
                    LVerticalCenter = (LBorderTop + LBorderBot) / 2
                    LVerticalRadius = (abs(LVerticalCenter - LBorderTop) + abs(LBorderBot - LVerticalCenter)) / 2
                    LLookLR = (LGazepoint[0] - LHorizontalCenter) / LHorizontalRadius
                    LLookUD = (LGazepoint[1] - LVerticalCenter) / LVerticalRadius
                    # If the eye is a little bit closed the gaze measurement will be prone to error
                    if (f.eye_blink[1] < 0.6 or f.eye_blink[0] < 0.6):
                        LookLR = GazeLR(LookLR, now)
                        LookUD = GazeUD(LookUD, now)
                    else:
                        LookLR = GazeLR((RLookLR + LLookLR) / 2, now)
                        LookUD = GazeUD((RLookUD + LLookUD) / 2, now)

                    socketString += f"[{LookLR-(f.euler[2]-90)*0.009:.6f};{-LookUD:.6f}],"

                    # Add the eyebrow steepness (sad/angry)
                    try:
                        steepness = (LEyeBrowSteepF(f.current_features['eyebrow_steepness_l'], now) + REyeBrowSteepF(f.current_features['eyebrow_steepness_r'], now)) / 2.0
                        socketString += f"{steepness:.6f},"
                    except:
                        socketString += "0.0,"

                handResults = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if handResults.multi_hand_landmarks != None:
                    amount = len(handResults.multi_handedness)
                    index = handResults.multi_handedness[0].classification[0].index  # 0 - Right, 1 - Left
                    if amount == 2:
                        if index == 0:
                            lHand = handResults.multi_hand_landmarks[1]
                            rHand = handResults.multi_hand_landmarks[0]
                        else:
                            lHand = handResults.multi_hand_landmarks[0]
                            rHand = handResults.multi_hand_landmarks[1]
                        socketString += f"[{-lHand.landmark[WRIST].x:.6f};{-lHand.landmark[WRIST].y:.6f};{-lHand.landmark[WRIST].z:.6f}],"
                        socketString += f"[{-rHand.landmark[WRIST].x:.6f};{-rHand.landmark[WRIST].y:.6f};{-rHand.landmark[WRIST].z:.6f}],"
                        socketString += f"[{-lHand.landmark[MMCP].x:.6f};{-lHand.landmark[MMCP].y:.6f};{-lHand.landmark[MMCP].z:.6f}],"
                        socketString += f"[{-lHand.landmark[RMCP].x:.6f};{-lHand.landmark[RMCP].y:.6f};{-lHand.landmark[RMCP].z:.6f}],"
                        socketString += f"[{-rHand.landmark[MMCP].x:.6f};{-rHand.landmark[MMCP].y:.6f};{-rHand.landmark[MMCP].z:.6f}],"
                        socketString += f"[{-rHand.landmark[RMCP].x:.6f};{-rHand.landmark[RMCP].y:.6f};{-rHand.landmark[RMCP].z:.6f}],"
                        socketString += f"[{-lHand.landmark[MPIP].x:.6f};{-lHand.landmark[MPIP].y:.6f};{-lHand.landmark[MPIP].z:.6f}],"
                        socketString += f"[{-lHand.landmark[MTIP].x:.6f};{-lHand.landmark[MTIP].y:.6f};{-lHand.landmark[MTIP].z:.6f}],"
                    else:
                        hand = handResults.multi_hand_landmarks[0]
                        if index == 0:
                            socketString += "[0.0;0.0;0.0],"
                            socketString += f"[{-hand.landmark[WRIST].x:.6f};{-hand.landmark[WRIST].y:.6f};{-hand.landmark[WRIST].z:.6f}],"
                            socketString += "[0.0;0.0;0.0],"
                            socketString += "[0.0;0.0;0.0],"
                            socketString += f"[{-hand.landmark[MMCP].x:.6f};{-hand.landmark[MMCP].y:.6f};{-hand.landmark[MMCP].z:.6f}],"
                            socketString += f"[{-hand.landmark[RMCP].x:.6f};{-hand.landmark[RMCP].y:.6f};{-hand.landmark[RMCP].z:.6f}],"
                            socketString += "[0.0;0.0;0.0],"
                            socketString += "[0.0;0.0;0.0],"
                        else:
                            socketString += f"[{-hand.landmark[WRIST].x:.6f};{-hand.landmark[WRIST].y:.6f};{-hand.landmark[WRIST].z:.6f}],"
                            socketString += "[0.0;0.0;0.0],"
                            socketString += f"[{-hand.landmark[MMCP].x:.6f};{-hand.landmark[MMCP].y:.6f};{-hand.landmark[MMCP].z:.6f}],"
                            socketString += f"[{-hand.landmark[RMCP].x:.6f};{-hand.landmark[RMCP].y:.6f};{-hand.landmark[RMCP].z:.6f}],"
                            socketString += "[0.0;0.0;0.0],"
                            socketString += "[0.0;0.0;0.0],"
                            socketString += f"[{-hand.landmark[MPIP].x:.6f};{-hand.landmark[MPIP].y:.6f};{-hand.landmark[MPIP].z:.6f}],"
                            socketString += f"[{-hand.landmark[MTIP].x:.6f};{-hand.landmark[MTIP].y:.6f};{-hand.landmark[MTIP].z:.6f}],"


                else:
                    socketString += "[0.0;0.0;0.0],[0.0;0.0;0.0],"
                    socketString += "[0.0;0.0;0.0],[0.0;0.0;0.0],"

                if detected:
                    if not lastDetected:
                        #Prints on console every time the AI loses/recovers tracking of a face
                        print("Got Tracking")
                        lastDetected = True
                    # This is valid data, so we store it as the last valid data
                    lastSocketString = socketString
                    # The boolean value is meant for Logix, usually returning the data to a neutral state
                    await websocket.send(f"True,{socketString}")
                else:
                    if lastDetected:
                        print("Lost Tracking")
                        lastDetected = False
                    # We send the last valid data as to not make the user's head jump around back to 0 rotation
                    await websocket.send(f"False,{lastSocketString}")
                
                failures = 0
            except Exception as e:
                if e.__class__ == KeyboardInterrupt:
                    if args.silent == 0:
                        print("Quitting")
                    break
                #traceback.print_exc()
                failures += 1
                if failures > 30:
                    break
            del frame
    except Exception as e:
        print("Lost connection to object")
        #traceback.print_exc()

print("Awaiting for Neos websocket client connection...\n")

# Opens server in port 7010
asyncio.get_event_loop().run_until_complete(
    websockets.serve(facetrack, 'localhost', 7010))
asyncio.get_event_loop().run_forever()