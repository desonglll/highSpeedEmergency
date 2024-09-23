# 包含车速信息，代码运行很慢，因为有车速信息，所以必须要逐帧处理

import cv2
import numpy as np
import time
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
path = 'XX/data/32.31.250.103/20240501_20240501125647_20240501140806_125649.mp4'
filename = os.path.basename(path)
video_name = os.path.splitext(filename)[0]
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS) # 获取视频帧率

# 定义用于计算流量的线的位置
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_position = frame_height // 2  #根据需要调整位置

# 初始化变量
frame_count = 0
output_data = []
test_frame_saved = False

# 车辆跟踪相关变量
vehicle_id_counter = 0  # 用于分配新的车辆ID
vehicles = {}           # 存储车辆信息，格式：{vehicle_id: {'box': [x1, y1, x2, y2], 'frames': n}}

# 置信度阈值和NMS阈值
confidence_threshold = 0.3
nms_threshold = 0.3

# 定义输出间隔（以帧数为单位）
output_frame_interval = 25  # 每隔25帧(fps = 25)输出一次

# 定义一个函数来计算IOU（用于简单的目标跟踪）
def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # 创建一个blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    indices = np.array(indices).flatten().tolist()

    detections = []
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        detections.append({'box': [x, y, x + w, y + h], 'class_id': class_ids[i], 'confidence': confidences[i]})

    # # 在帧上绘制检测结果（用于测试）
    # for detection in detections:
    #     x1, y1, x2, y2 = detection['box']
    #     label = classes[detection['class_id']]
    #     confidence = detection['confidence']
    #     # 绘制边界框
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     # 显示标签和置信度
    #     cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # # 绘制流量计算的线
    # cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)

    # # 保存测试帧（只保存一次）
    # if not test_frame_saved:
    #     cv2.imwrite('test_frame.jpg', frame)
    #     print("测试帧已保存为 'test_frame.jpg'")
    #     test_frame_saved = True

    # 车辆跟踪和速度计算
    # 用于存储当前帧的车辆信息
    
    current_vehicles = {}
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        matched = False
        for vehicle_id, vehicle_data in vehicles.items():
            prev_box = vehicle_data['box']
            iou = compute_iou([x1, y1, x2, y2], prev_box)
            if iou > 0.5:
                # 更新车辆信息
                dx = x1 - prev_box[0]
                dy = y1 - prev_box[1]
                distance = np.sqrt(dx*dx + dy*dy)
                speed = distance * fps  # 像素/秒
                current_vehicles[vehicle_id] = {
                    'box': [x1, y1, x2, y2],
                    'speed': speed,
                    'frames': vehicle_data['frames'] + 1
                }
                matched = True
                break
        if not matched:
            # 分配新的车辆ID
            vehicle_id_counter += 1
            current_vehicles[vehicle_id_counter] = {
                'box': [x1, y1, x2, y2],
                'speed': 0,
                'frames': 1
            }
    vehicles = current_vehicles

    # 计算密度：当前帧中的车辆数量
    density = len(detections)

    # 计算流量：判断位于指定线位置的车辆数量
    flow = 0
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        # 检查车辆是否与线相交
        if y1 <= line_position <= y2:
            flow += 1

    # 计算平均速度
    speeds = [vehicle['speed'] for vehicle in vehicles.values() if vehicle['speed'] > 0]
    if speeds:
        average_speed = sum(speeds) / len(speeds)
    else:
        average_speed = 0

    # 每隔一定的帧数输出一次数据
    if frame_count % output_frame_interval == 0:
        output = {
            'Frame': frame_count,
            'Flow': flow,
            'Density': density,
            'Speed': average_speed
        }
        print(output)
        output_data.append(output)

cap.release()

# 保存结果到文件
df = pd.DataFrame(output_data)
df.to_csv(f'XX/res/32.31.250.103/{video_name}.csv', index=False)
