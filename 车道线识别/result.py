import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import cv2
import numpy as np

def video_solution(jpg):
    #o_jpg = cv2.cvtColor(jpg, cv2.COLOR_RGB2BGR)  # cv2 为BGR 先转换好
    o_jpg = jpg
    # print(o_jpg.shape)>>(540, 960, 3)
    # print(type(o_jpg))
    jpg_gray = cv2.cvtColor(o_jpg, cv2.COLOR_RGB2GRAY)
    # 高斯平滑处理
    jpg_gray_blur = cv2.GaussianBlur(jpg_gray, (5, 5), 0, 0)
    # Canny边缘检测 推荐为1：2 或 1：3
    jpg_edge = cv2.Canny(jpg_gray_blur, 90, 180)
    # POI提取
    mask = np.zeros_like(jpg_edge)
    a = np.array([[(0, 540), (460, 325), (520, 325), (960, 540)]])
    cv2.fillPoly(mask, a, 255)
    poi_edge = cv2.bitwise_and(jpg_edge, mask)

    # 霍夫变换 Hough直线检测 >> 点化为线
    # 返回的为一个容器 指定为np.array([])
    line_point = cv2.HoughLinesP(poi_edge, 1, np.pi / 180, 15,
                                 lines=np.array([]),
                                 minLineLength=40,
                                 maxLineGap=20)

    # 车道计算 >> 分段的线转化为连续的线
    # 通过k区分左车道与右车道
    # 存的都是点
    line_left = []
    line_right = []
    line_left_k = []
    line_right_k = []
    for line in line_point:  # line_point >> np.array([])
        for x1, y1, x2, y2 in line:  # line >> []
            k = (y2 - y1) / (x2 - x1)
            if k > 0:  # 左车道
                line_left.append(line)
                line_left_k.append(k)
            if k < 0:
                line_right.append(line)
                line_right_k.append(k)
            # 暂时不考虑是否为空表
    print(len(line_left),len(line_left_k),len(line_right),len(line_right_k))
    # 斜率过大的移除 >> 图片中没有需要被去除的
    for line in line_left:
        if(len(line_left) > 0):
            k_mean = np.mean(line_left_k)  # mean
            k_mean_difference = [abs(k - k_mean) for k in line_left_k]  # 计算斜率与均值的差
            print(1111,len(k_mean_difference),np.argmax(k_mean_difference))
            if k_mean_difference[np.argmax(k_mean_difference)] > 0.1:  # argmax返回最大数的索引
                print(line_left_k[np.argmax(k_mean_difference)-1])
                line_left.pop(np.argmax(k_mean_difference))
                line_left_k.pop(np.argmax(k_mean_difference))

    for line in line_right:
        if (len(line_right) > 0):
            k_mean = np.mean(line_right_k)  # mean
            k_mean_difference = [abs(k - k_mean) for k in line_right_k]  # 计算斜率与均值的差
            print(22221, len(k_mean_difference), np.argmax(k_mean_difference))
            if k_mean_difference[np.argmax(k_mean_difference)] > 0.1:  # argmax返回最大数的索引
                print(line_right_k[np.argmax(k_mean_difference)-1])
                line_right.pop(np.argmax(k_mean_difference))
                line_right_k.pop(np.argmax(k_mean_difference))
            print(22223, len(line_right), np.argmax(k_mean_difference))

    # 将点绘制成线 >> 线性回归
    x_left = []
    y_left = []
    if(len(line_left)>0):
        for point in line_left:
            x_left.append(point[0][0])
            x_left.append(point[0][2])
            y_left.append(point[0][1])
            y_left.append(point[0][3])
    if (len(x_left) > 0):
        fit = np.polyfit(y_left, x_left, 1)  # 注意图像显示的时候 x和y 是相反的
        fit_fn = np.poly1d(fit)  # 拟合曲线以获得曲线中的值
        x_left_min = int(fit_fn(325))
        x_left_max = int(fit_fn(540))
        cv2.line(o_jpg, (x_left_min, 325), (x_left_max, 540), [255, 0, 0], 2)

    x_right = []
    y_right = []
    if (len(line_right) > 0):
        for point in line_right:
            x_right.append(point[0][0])
            x_right.append(point[0][2])
            y_right.append(point[0][1])
            y_right.append(point[0][3])
    if(len(x_right) > 0):
        fit = np.polyfit(y_right, x_right, 1)  # 注意图像显示的时候 x和y 是相反的
        fit_fn = np.poly1d(fit)  # 拟合曲线以获得曲线中的值
        x_right_min = int(fit_fn(325))
        x_right_max = int(fit_fn(540))
        cv2.line(o_jpg, (x_right_min, 325), (x_right_max, 540), [255, 0, 0], 2)

    #cv2.imshow('', o_jpg)
    #cv2.waitKey()
    print("over")
    return o_jpg

video_clip = VideoFileClip("video_2.mp4")
video_output = video_clip.fl_image(video_solution)
video_output.write_videofile('video_1_sol.mp4', audio=False)