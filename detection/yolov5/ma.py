from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import time
from djitellopy.tello import Tello
from time import sleep
import numpy as np
import cv2.cv2 as cv2
import torch
import time
from PIL import Image
import djitellopy as tello
import requests
import json
import base64

global alarm
global seen
global me
global stringtext
global finalImage
seen = 0
alarm = 0
stringtext =""
app = FastAPI()

from werkzeug.utils import secure_filename, send_from_directory
import os
import sys
from io import BytesIO
import cv2
sys.path.append('./utils')
# uploads_dir = os.path.join(app.instance_path, 'uploads')
# os.makedirs(uploads_dir, exist_ok=True)

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import math
import torch
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def distance(x,y):
    global stringtext
    # 0~9
    arr18 = [[59, 55, 52, 49, 47, 46, 45, 44, 43, 43, 43, 43, 44, 45, 46, 47, 49, 52, 55, 59],
            [49, 45, 43, 42, 39, 38, 38, 37, 37, 37, 37, 37, 37, 38, 38, 39, 42, 43, 45, 49],
            [42, 39, 37, 36, 34, 33, 33, 32, 32, 32, 32, 32, 32, 33, 33, 34, 36, 37, 39, 42],
            [36, 33, 31, 30, 29, 28, 27, 27, 27, 27, 27, 27, 27, 27, 28, 29, 30, 31, 33, 36],
            [31, 29, 27, 26, 26, 25, 25, 24, 24, 24, 24, 24, 24, 25, 25, 26, 26, 27, 29, 31],
            [28, 27, 26, 25, 24, 23, 22, 22, 22, 22, 22, 22, 22, 22, 23, 24, 25, 26, 27, 28],
            [27, 25, 24, 23, 22, 21, 20, 20, 20, 19, 19, 20, 20, 20, 21, 22, 23, 24, 25, 27],
            [25, 24, 22, 21, 20, 19, 19, 18, 18, 18, 18, 18, 18, 19, 19, 20, 21, 22, 24, 25],
            [24, 22, 21, 19, 18, 17, 17, 17, 16, 16, 16, 16, 17, 17, 17, 18, 19, 21, 22, 24],
            [22, 21, 19, 18, 17, 16, 15, 15, 15, 15, 15, 15, 15, 15, 16, 17, 18, 19, 21, 22]]
            
    # 10~16
    arr36 = [[23, 19, 17, 16, 16, 15, 14, 14, 14, 13, 13, 14, 14, 14, 15, 16, 16, 17, 19, 23],
            [18, 16, 15, 14, 14, 13, 12, 12, 12, 12, 12, 12, 12, 12, 13, 14, 14, 15, 16, 18],
            [16, 15, 14, 13, 12, 12, 11, 11, 11, 10, 10, 11, 11, 11, 12, 12, 13, 14, 15, 16],
            [16, 14, 13, 12, 11, 11, 10, 10, 10, 9, 9, 10, 10, 10, 11, 11, 12, 13, 14, 16],
            [14, 13, 12, 11, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 11, 12, 13, 14],
            [13, 12, 11, 10, 9, 9, 9, 8, 8, 8, 8, 8, 8, 9, 9, 9, 10, 11, 12, 13],
            [12, 11, 10, 9, 9, 8, 8, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9, 10, 11, 12]]
    n = int((x / 48) - 1)

    if y < 468:
        m = int(((y - 288) / 18))
        dis = arr18[m][n]
    else:
        m = int(((y - 468) / 36))
        dis = arr36[m][n]
    if m>=0 and n>=0:
        stringtext = stringtext  + "드론으로 부터 거리 : " + str(dis)
    else:
        stringtext = stringtext + ("거리추정불가, 반대편 드론의 좌표 받아와야함")

def direction(x,y):
    global stringtext
    #angle = np.arctan(720-y/480-x)
    #angle = np.arctan(720-y/(x-480))
    #print(math.degrees(math.pi))

    # 960 x 720
    # 중심 기준 왼쪽이면
    if(x < 480):
        azimuth =  '북서쪽'
        angle = math.atan((720-y)/(480-x))
        angle = math.degrees(angle)

    # 중심 기준 오른쪽이면
    elif(x > 480):
        azimuth = '북동쪽'
        angle = math.atan((720-y)/(x-480))
        angle = math.degrees(angle)

    # 중심이면 정북쪽
    else:
        azimuth = '정북쪽'
        angle = 0  
    stringtext = stringtext + "방향 : " + str(azimuth)
    strtmp = str(round(angle))
    stringtext = stringtext + "각도 : " + strtmp
    # print(f"각도 : {angle:.2f}")


def run(
        weights=ROOT / '640510best.pt',  # model path or triton URL
        source=ROOT / '9.mp4',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global alarm
    global seen
    global finalImage
    frame_array = [0] * 50 # 50프레임짜리 칸
    global stringtext
    stringtext = ""
    #yolov5 기존 detect과정
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    alarm = 0
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            # 50칸짜리 리스트 한 번 다 채운 뒤에
            if (seen > 50):
                # 첫번째 요소 없애고
                frame_array.pop(0)
                # 다시 끝에 0으로 채우기
                frame_array.append(0)

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                #동영상이라 여기로 들어감
                #im0가 이미지, frame이 현재 몇 프레임째인지 세어주는 역할임
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            height, width = im0.shape[:2]

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    # print("name은!! ",names[int(cls)])

                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(f'{txt_path}.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') 이거 하면 drowning 0.52 이렇게 숫자처럼 나오는거고 아래는 drowning(label)만 나오는거
                        if (names[int(cls)] == 'Person drowning'):
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}') #익수 상태인것만 출력
                            if (seen <= 50):
                                # 50프레임까지는 (현재 프레임 -1)번째의 0을 1로 바꾸기
                                frame_array[seen - 1] = 1
                            else:
                                # 50프레임 후로는 제일 마지막 칸을 1로 바꾸기
                                frame_array[49] = 1
                        # plot_one_box(xyxy, im0, label=label)
                            #박스 센터 잡아서 깊이추정해주는 함수
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            x_center = (c1[0]+c2[0])/2
                            y_center = (c1[1]+c2[1])/2
                            print("이름",names[int(cls)])
                            #print("x,y",x_center,y_center)
                            x_center = x_center / width * 960
                            y_center = y_center / height * 720
                            stringtext = ""
                            distance(x_center,y_center)
                            direction(x_center,y_center)
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            im0 = annotator.result()
            im0 = cv2.resize(im0, (300, 200)) #이거 안드로이드 스튜디오에서 받을때 화면이 잘려서 일단 넣어둔거임, 최적 비율 찾아서 조절해야할듯
            (flag, encodedImage) = cv2.imencode(".jpg", im0)
            print("리스트 출력 : ", frame_array)
            # 50프레임 전까지는 알림 안 보냄, 50프레임부터~
            if (seen >= 50):
                array_sum = sum(frame_array)
                array_average = array_sum / 50
                print("리스트 평균 : ", array_average)
                if (array_average >= 0.35):
                    print("**\n**\n알람 보냄\n**\n**")
                    finalImage = encodedImage
                    alarm = 1
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')#한 프레임씩 연속적으로 byte 형식으로 돌려주는 함수
        # 프레임 확인하려고 logger 다시 추가했어영
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")


def normal_run(
        weights=ROOT / '640510best.pt',  # model path or triton URL
        source=ROOT / 'normal1.mp4',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global alarm
    global seen
    global finalImage
    frame_array = [0] * 50 # 50프레임짜리 칸
    global stringtext
    stringtext = ""
    #yolov5 기존 detect과정
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    alarm = 0
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            # 50칸짜리 리스트 한 번 다 채운 뒤에
            if (seen > 50):
                # 첫번째 요소 없애고
                frame_array.pop(0)
                # 다시 끝에 0으로 채우기
                frame_array.append(0)

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                #동영상이라 여기로 들어감
                #im0가 이미지, frame이 현재 몇 프레임째인지 세어주는 역할임
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            height, width = im0.shape[:2]

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    # print("name은!! ",names[int(cls)])

                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(f'{txt_path}.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') 이거 하면 drowning 0.52 이렇게 숫자처럼 나오는거고 아래는 drowning(label)만 나오는거
                        if (names[int(cls)] == 'Person drowning'):
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}') #익수 상태인것만 출력
                            if (seen <= 50):
                                # 50프레임까지는 (현재 프레임 -1)번째의 0을 1로 바꾸기
                                frame_array[seen - 1] = 1
                            else:
                                # 50프레임 후로는 제일 마지막 칸을 1로 바꾸기
                                frame_array[49] = 1
                        # plot_one_box(xyxy, im0, label=label)
                            #박스 센터 잡아서 깊이추정해주는 함수
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            x_center = (c1[0]+c2[0])/2
                            y_center = (c1[1]+c2[1])/2
                            print("이름",names[int(cls)])
                            #print("x,y",x_center,y_center)
                            x_center = x_center / width * 960
                            y_center = y_center / height * 720
                            stringtext = ""
                            distance(x_center,y_center)
                            direction(x_center,y_center)
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            im0 = annotator.result()
            im0 = cv2.resize(im0, (300, 200)) #이거 안드로이드 스튜디오에서 받을때 화면이 잘려서 일단 넣어둔거임, 최적 비율 찾아서 조절해야할듯
            (flag, encodedImage) = cv2.imencode(".jpg", im0)
            print("리스트 출력 : ", frame_array)
            # 50프레임 전까지는 알림 안 보냄, 50프레임부터~
            if (seen >= 50):
                array_sum = sum(frame_array)
                array_average = array_sum / 50
                print("리스트 평균 : ", array_average)
                if (array_average >= 0.35):
                    print("**\n**\n알람 보냄\n**\n**")
                    finalImage = encodedImage
                    alarm = 1
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')#한 프레임씩 연속적으로 byte 형식으로 돌려주는 함수
        # 프레임 확인하려고 logger 다시 추가했어영
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")


def normal_run2(
        weights=ROOT / '640510best.pt',  # model path or triton URL
        source=ROOT / 'normal2.mp4',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global alarm
    global seen
    global finalImage
    frame_array = [0] * 50 # 50프레임짜리 칸
    global stringtext
    stringtext = ""
    #yolov5 기존 detect과정
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    alarm = 0
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            # 50칸짜리 리스트 한 번 다 채운 뒤에
            if (seen > 50):
                # 첫번째 요소 없애고
                frame_array.pop(0)
                # 다시 끝에 0으로 채우기
                frame_array.append(0)

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                #동영상이라 여기로 들어감
                #im0가 이미지, frame이 현재 몇 프레임째인지 세어주는 역할임
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            height, width = im0.shape[:2]

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    # print("name은!! ",names[int(cls)])

                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(f'{txt_path}.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') 이거 하면 drowning 0.52 이렇게 숫자처럼 나오는거고 아래는 drowning(label)만 나오는거
                        if (names[int(cls)] == 'Person drowning'):
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}') #익수 상태인것만 출력
                            if (seen <= 50):
                                # 50프레임까지는 (현재 프레임 -1)번째의 0을 1로 바꾸기
                                frame_array[seen - 1] = 1
                            else:
                                # 50프레임 후로는 제일 마지막 칸을 1로 바꾸기
                                frame_array[49] = 1
                        # plot_one_box(xyxy, im0, label=label)
                            #박스 센터 잡아서 깊이추정해주는 함수
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            x_center = (c1[0]+c2[0])/2
                            y_center = (c1[1]+c2[1])/2
                            print("이름",names[int(cls)])
                            #print("x,y",x_center,y_center)
                            x_center = x_center / width * 960
                            y_center = y_center / height * 720
                            stringtext = ""
                            distance(x_center,y_center)
                            direction(x_center,y_center)
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            im0 = annotator.result()
            im0 = cv2.resize(im0, (300, 200)) #이거 안드로이드 스튜디오에서 받을때 화면이 잘려서 일단 넣어둔거임, 최적 비율 찾아서 조절해야할듯
            (flag, encodedImage) = cv2.imencode(".jpg", im0)
            print("리스트 출력 : ", frame_array)
            # 50프레임 전까지는 알림 안 보냄, 50프레임부터~
            if (seen >= 50):
                array_sum = sum(frame_array)
                array_average = array_sum / 50
                print("리스트 평균 : ", array_average)
                if (array_average >= 0.35):
                    print("**\n**\n알람 보냄\n**\n**")
                    finalImage = encodedImage
                    alarm = 1
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')#한 프레임씩 연속적으로 byte 형식으로 돌려주는 함수
        # 프레임 확인하려고 logger 다시 추가했어영
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")


@app.get("/")
def hello():
    return "hello"

@app.get('/video')
def video1():
   return StreamingResponse(run(), media_type="multipart/x-mixed-replace; boundary=frame") #돌려주는 함수, 기존에 detect()였던 부분을 run으로 바꿈


@app.get('/normal1')
def normal1():
   return StreamingResponse(normal_run(), media_type="multipart/x-mixed-replace; boundary=frame") #돌려주는 함수, 기존에 detect()였던 부분을 run으로 바꿈

@app.get('/normal2')
def normal2():
   return StreamingResponse(normal_run2(), media_type="multipart/x-mixed-replace; boundary=frame") #돌려주는 함수, 기존에 detect()였던 부분을 run으로 바꿈



@app.get('/alarm')
def getalarm():
    return alarm

@app.get('/seen')
def getseen():
    return seen


@app.get("/land")
def connect():
    global me
    me = tello.Tello()
    me.connect()
    print(me.get_battery())

    # return {"status": "connected"}
    if me is None:
        return {"status": "drone not connected"}
    else:
        me.streamon()
        # me.takeoff()
        # me.move_forward(100)

        def generate_frames():
                while True:
                    frame = me.get_frame_read().frame
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
@app.get('/stop')
def stop():
    global me
    me.streamoff()  # Assuming this stops the strea
    # me.move_back(100)
    # me.land()
    # me = None

@app.get('/depth')
def getdepth():
    print("stringtext", stringtext)
    return stringtext
@app.get('/alarm_page')
def getcurrent():
    def getimage():
        global finalImage
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(finalImage) + b'\r\n')

    return StreamingResponse(getimage(), media_type='multipart/x-mixed-replace; boundary=frame')
