import os
import cv2
import sys
import time
import json
import redis
import ffmpeg
import numpy as np
import subprocess as sp

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from yolov8_utils import *
from rknnlite.api import RKNNLite
from motrackers import CentroidKF_Tracker 
from concurrent.futures import ThreadPoolExecutor


def read_frame(share_dict, flowunit_data, queue_dict, data):
    pull_video_url = flowunit_data["config"]["pull_video_url"]
    height = flowunit_data["config"]["height"]
    width = flowunit_data["config"]["width"]
    fps = flowunit_data["config"]["fps"]
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-c:v', 'h264_rkmpp',
        '-i', pull_video_url,  
        '-r', f'{fps}',
        '-loglevel', 'info',
        '-s', f'{width}x{height}',
        '-an', '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:'
    ]

    ffmpeg_process = sp.Popen(ffmpeg_cmd, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=10**7)

    index = 0
    while True:
        index += 1
        raw_frame = ffmpeg_process.stdout.read(width * height * 3)
        if not raw_frame:
            break
        else:
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, -1))
            data["frame"] = frame
            for queue_name in flowunit_data["multi_output"]:
                queue_dict[queue_name].put(data)

    # 读取结束，图片数据置为None
    data["frame"] = None
    for queue_name in flowunit_data["multi_output"]:
        queue_dict[queue_name].put(data)
    ffmpeg_process.stdout.close()
    ffmpeg_process.terminate()


def model_infer(share_dict, flowunit_data, queue_dict, data):
    model_file = flowunit_data["config"]["model_file"]
    model_info = flowunit_data["config"]["model_info"]
    batch_size = flowunit_data["config"]["batch_size"]

    rknn_lite_list = []
    for i in range(batch_size):
        rknn_lite = RKNNLite()
        rknn_lite.load_rknn(model_file)
        rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        rknn_lite_list.append(rknn_lite)

    with open(model_info, "r") as f:
        model_info = json.load(f)
    labels = []
    for label in list(model_info["model_classes"].values()):
        labels.append(label)
    IMG_SIZE = model_info["input_shape"][0][-2:]
    OBJ_THRESH = model_info["conf_threshold"]
    NMS_THRESH = model_info["nms_threshold"]

    exist = False
    index = 0
    while True:
        index += 1
        image_batch = []
        if flowunit_data["single_input"] is not None:
            for i in range(batch_size):
                data = queue_dict[flowunit_data["single_input"]].get()
                # 图片数据为None就退出循环
                if data["frame"] is None:
                    exist = True
                    break
                image_batch.append(data)
        else:
            break
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(executor.map(infer_single_image, 
                                        [(data["frame"], rknn_lite_list[i % batch_size], 
                                          IMG_SIZE, OBJ_THRESH, NMS_THRESH) for i, data in enumerate(image_batch)]))
        for i, (boxes, classes, scores) in enumerate(results):
            classes = [labels[class_id] for class_id in classes]
            data = image_batch[i] 
            if data.get("boxes") is None:
                data["boxes"] = boxes
                data["classes"] = classes
                data["scores"] = scores
            else:
                data["boxes"].extend(boxes)
                data["classes"].extend(classes)
                data["scores"].extend(scores)
            for queue_name in flowunit_data["multi_output"]:
                queue_dict[queue_name].put(data)
        if exist:
            break

    # 读取结束，图片数据置为None
    data["frame"] = None
    for queue_name in flowunit_data["multi_output"]:
        queue_dict[queue_name].put(data)
    for rknn_lite in rknn_lite_list:
        rknn_lite.release()

    
def kf_tracker(share_dict, flowunit_data, queue_dict, data):
    tracker = CentroidKF_Tracker(max_lost=30)
    index = 0
    while True:
        index += 1
        if flowunit_data["single_input"] is not None:
            data = queue_dict[flowunit_data["single_input"]].get()
        else:
            break
        # 图片数据为None就退出循环
        if data["frame"] is None:
            break
        boxes, classes, scores = data.get("boxes"), data.get("classes"), data.get("scores")
        boxes = np.array(boxes)
        classes = np.array(classes)
        scores = np.array(scores)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  
        results = tracker.update(boxes, scores, classes)
        boxes = []
        classes = []
        scores = []
        tracks = []
        for result in results:
            frame_num, id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z, class_id = result 
            boxes.append([bb_left, bb_top, bb_left + bb_width, bb_top + bb_height])
            classes.append(class_id)
            scores.append(confidence)
            tracks.append(id)
        data["boxes"] = boxes
        data["classes"] = classes
        data["scores"] = scores
        data["tracks"] = tracks
        for queue_name in flowunit_data["multi_output"]:
            queue_dict[queue_name].put(data)

    # 读取结束，图片数据置为None
    data["frame"] = None
    for queue_name in flowunit_data["multi_output"]:
        queue_dict[queue_name].put(data)


def draw_boxes(share_dict, flowunit_data, queue_dict, data):
    index = 0
    while True:
        index += 1
        if flowunit_data["single_input"] is not None:
            data = queue_dict[flowunit_data["single_input"]].get()
        else:
            break
        # 图片数据为None就退出循环
        if data["frame"] is None:
            break
        boxes, classes, scores = data.get("boxes"), data.get("classes"), data.get("scores")
        if boxes is not None:
            tracks = data.get("tracks")
            if tracks is not None:
                for boxe, clss, track in zip(boxes, classes, tracks):
                    cv2.rectangle(data["frame"], (boxe[0], boxe[1]), (boxe[2], boxe[3]), (0, 255, 0), 2)
                    cv2.putText(data["frame"], f"{clss} {track}", (boxe[0], boxe[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                for boxe, clss, conf in zip(boxes, classes, scores):
                    cv2.rectangle(data["frame"], (boxe[0], boxe[1]), (boxe[2], boxe[3]), (0, 255, 0), 2)
                    cv2.putText(data["frame"], f"{clss} {conf * 100:.2f}%", (boxe[0], boxe[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for queue_name in flowunit_data["multi_output"]:
            queue_dict[queue_name].put(data)

    # 读取结束，图片数据置为None
    data["frame"] = None
    for queue_name in flowunit_data["multi_output"]:
        queue_dict[queue_name].put(data)


def push_frame(share_dict, flowunit_data, queue_dict, data):
    push_video_url = flowunit_data["config"]["push_video_url"]
    format = flowunit_data["config"]["format"]
    height = flowunit_data["config"]["height"]
    width = flowunit_data["config"]["width"]
    fps = flowunit_data["config"]["fps"]

    process_stdin = (
        ffmpeg
        .input('pipe:',
                format='rawvideo',  
                pix_fmt='bgr24',  
                s="{}x{}".format(width, height),  
                framerate=fps)  
        .filter('fps',
                fps=fps,  
                round='up')  
        .output(
                push_video_url,
                vcodec='h264_rkmpp',  
                bitrate='2500k',
                f=format,
                g=fps,  
                an=None,  
                timeout='0'  
                )
        .overwrite_output()
        .run_async(cmd=["ffmpeg", "-re"], pipe_stdin=True)
    )

    index = 0
    while True:
        index += 1
        if flowunit_data["single_input"] is not None:
            data = queue_dict[flowunit_data["single_input"]].get()
        else:
            break
        # 图片数据为None就退出循环
        if data["frame"] is None:
            break
        frame = data["frame"]
        frame = cv2.resize(frame, (width, height))
        process_stdin.stdin.write(frame.tobytes())
    process_stdin.stdin.close()
    process_stdin.terminate()


def redis_push(share_dict, flowunit_data, queue_dict, data):
    task_id = flowunit_data["config"]["task_id"]
    host = flowunit_data["config"]["host"]
    port = flowunit_data["config"]["port"]
    username = flowunit_data["config"]["username"]
    password = flowunit_data["config"]["password"]
    db = flowunit_data["config"]["db"]

    r = redis.Redis(
        host = host,
        port = port,
        username = username,
        password = password,
        db = db,
        decode_responses = True
    )

    index = 0
    while True:
        index += 1
        if flowunit_data["single_input"] is not None:
            data = queue_dict[flowunit_data["single_input"]].get()
        else:
            break
        # 图片数据为None就退出循环
        if data["frame"] is None:
            break
        track_objs = []
        height, width = data["frame"].shape[:2]
        boxes, classes, scores, tracks = data.get("boxes"), data.get("classes"), data.get("scores"), data.get("tracks")
        if boxes is not None:
            for boxe, clss, conf, track in zip(boxes, classes, scores, tracks):
                x1 = float(boxe[0] / width)
                y1 = float(boxe[1] / height)
                x2 = float(boxe[2] / width)
                y2 = float(boxe[3] / height)
                track_obj = {
                    "bbox": [x1, y1, x2, y2],
                    "track_id": int(track),
                    "class_id": 0,
                    "class_name": str(clss)
                }
                track_objs.append(track_obj)
            key = 'vision:track:' + str(task_id) + ':frame:' + str(index)
            value = json.dumps({"track_result": track_objs})
            r.set(key, value)
            r.expire(key, 2)
            print(track_objs)
    r.close()
