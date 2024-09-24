# 为了加速处理！ 移除车速信息，不再逐帧处理，每25帧处理一次

import cv2
import numpy as np
import pandas as pd
import os

# 加载YOLOv3模型
net = cv2.dnn.readNetFromDarknet('yolo/yolov3.cfg', 'yolo/yolov3.weights')
layer_names = net.getUnconnectedOutLayersNames()

# 定义检测的类（只检测车辆相关的类别）
classes_to_detect = ['bicycle', 'car', 'motorbike', 'bus', 'truck']
with open('yolo/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# 打开视频文件
path = 'XX/data/32.31.250.103/20240501_20240501140806_20240501152004_140807.mp4'
filename = os.path.basename(path)
video_name = os.path.splitext(filename)[0]
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率

# 定义用于计算流量的线的位置
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_position = frame_height // 2  # 根据需要调整位置

# 初始化变量
frame_count = 0
output_data = []

# 置信度阈值和NMS阈值
confidence_threshold = 0.3
nms_threshold = 0.3

# 定义处理间隔（每5帧处理一次）
process_frame_interval = 25  # 每25帧处理一次

# 定义输出间隔（以处理的帧数为单位）
output_frame_interval = 1  # 每处理25帧输出一次

processed_frame_count = 0  # 处理过的帧数计数器

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # 仅每隔指定的帧数处理一次
    if frame_count % process_frame_interval != 0:
        continue

    processed_frame_count += 1  # 增加处理过的帧数计数器

    # 创建一个blob
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    # 前向传播
    outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    # 解析输出
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                class_name = classes[class_id]
                if class_name in classes_to_detect:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # 非极大值抑制
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, nms_threshold)
    indices = np.array(indices).flatten().tolist()

    detections = []
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        detections.append({'box': [x, y, x + w, y + h],
                           'class_id': class_ids[i], 'confidence': confidences[i]})

    # 计算密度：当前帧中的车辆数量
    density = len(detections)

    # 计算流量：判断位于指定线位置的车辆数量
    flow = 0
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        # 检查车辆是否与线相交
        if y1 <= line_position <= y2:
            flow += 1

    # 每隔一定的处理帧数输出一次数据
    if processed_frame_count % output_frame_interval == 0:
        output = {
            'Frame': frame_count,
            'Flow': flow,
            'Density': density
        }
        print(output)
        output_data.append(output)
        df = pd.DataFrame(output_data)
        df.to_csv(
            f'XX/res/32.31.250.103/{video_name}.csv', index=False)
        # 如果不需要在此终止循环，可以移除下面的break
        # break

cap.release()

# 保存结果到文件
df = pd.DataFrame(output_data)
df.to_csv(f'XX/res/32.31.250.103/{video_name}.csv', index=False)
