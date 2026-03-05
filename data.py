#### DATA.PY (WIP) ##########################
## involves data processing and treatment
#############################################
import json
import cv2

def read_video(path):
    frames = []
    video = cv2.VideoCapture(path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        h, w, _ = frame.shape
        frame = cv2.resize(frame, (int(w*2), int(h*2)), cv2.INTER_LINEAR)
        frames.append(frame)
    video.release()
    cv2.destroyAllWindows()
    return frames

def save_video(frames, path):
    h,w,_ = frames[0].shape
    codec = cv2.VideoWriter_fourcc(*"XVID")
    output = cv2.VideoWriter(path, codec, fps=30.0, frameSize=(w,h))
    for frame in frames:
        output.write(frame)
    output.release()

def filter_annotations(path):
    #filtering for not null segments annotations
    filtered_file = "dataset/filtered_annotations.json"
    events = {
        1: "dataset/vault.json",
        2: "dataset/floorexercise.json",
        3: "dataset/balancebeam.json",
        4: "dataset/unevenbars.json"
    }

    with open(path, "r") as input:
        data = json.load(input)

    filtered = {}
    for video_id, routine in data.items():
        valid_routines = {}
        for routine_id, info in routine.items():
            if info.get("segments") is not None: #segment: "null"
                valid_routines[routine_id] = info
        
        if valid_routines:
            filtered[video_id] = valid_routines

    with open(filtered_file, "w") as output:
        json.dump(filtered, output, indent=4)

    #file event separation (vault, floor exercise, balance beam, uneven bars)
    with open(filtered_file, "r") as input:
        filtered_data = json.load(input)

    filtered_events = {1: {}, 2: {}, 3: {}, 4: {}}
    for video_id, routine in filtered_data.items():
        for routine_id, info in routine.items():
            event_id = info.get("event")

            if event_id in filtered_events:
                if video_id not in filtered_events[event_id]:
                    filtered_events[event_id][video_id] = {}
                filtered_events[event_id][video_id][routine_id] = info
        
    for event_id, filename in events.items():
        with open(filename, "w") as output:
            json.dump(filtered_events[event_id], output, indent=4)