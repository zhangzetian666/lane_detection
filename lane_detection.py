import cv2 as cv
import numpy as np

'''
1、边缘检测
基本原理：检测亮度的急剧变化
canny 边缘检测
1'降噪  5*5 高斯滤波器
2'求亮度梯度
3'非极大值抑制：细化边缘
4'hysteresis thresholding :
    非极大值抑制后可以确认强像素在最终边缘映射中，但还要对弱像素进行进一步确定：
        梯度>maxval 是边缘   < minval 不是边缘并删除
        梯度在最小值和最大值之间：只有和边缘相连时才是边缘
'''
def do_canny(frame):
    # 转灰度图
    gray=cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    # 高斯滤波
    blur=cv.GaussianBlur(gray,(5,5),0)
    canny=cv.Canny(blur,50,150)
    return canny

'''
2、手动分割路边区域
'''
def do_segment(frame):
    height=frame.shape[0]
    weight=frame.shape[1]
    # polygons=np.array([[(0,height),(weight,height),(weight/2-20,height/2-10)]])
    polygons=np.array([
        [(0,height),(800,height),(380,290)]
    ])
    mask=np.zeros_like(frame)
    cv.fillPoly(mask,polygons,255)
    segment=cv.bitwise_and(frame,mask)
    return segment
'''
3、霍夫变换得到车道线
hough=cv.HoughLinesP(segment,2,np.pi/180,100,np.array([]),minLineLength=100,maxLineGap=50)
'''

'''
4、获取车道线并叠加到原始图像中
'''

# 取若干车道线的平均斜率和截距，得到最终两边的车道线
def calculate_lines(frame,lines):
    left=[]
    right=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)   # 根据直线上两点求斜率 截距
        slope=parameters[0]
        y_intercept=parameters[1]
        if slope<0:   # 如果斜率是小于 0 的，属于车道右边的线
            left.append((slope,y_intercept))
        else:
            right.append((slope,y_intercept))
    left_avg=np.average(left,axis=0)
    right_avg=np.average(right,axis=0)
    left_line=calculate_coordinates(frame,left_avg)
    right_line=calculate_coordinates(frame,right_avg)
    return np.array([left_line,right_line])


# 根据斜率，截距得到直线
def calculate_coordinates(frame,parameters):
    slope,intercept=parameters
    y1=frame.shape[0]
    y2=int(y1-150)    # 在图像底部往上 150 的地方
    x1=int((y1-intercept)/slope)   # 设置x
    x2=int((y2-intercept)/slope)   # 设置x
    return np.array([x1,y1,x2,y2])

# 叠加到原视频中
def visualize_lines(frame, lines):
    lines_visualize = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(lines_visualize, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=5)
    return lines_visualize


cap=cv.VideoCapture('input.mp4')
while cap.isOpened():
    ret,frame=cap.read()
    canny=do_canny(frame)
    cv.imshow('canny',canny)
    segment=do_segment(canny)
    hough=cv.HoughLinesP(segment,2,np.pi/180,100,np.array([]),minLineLength=100,maxLineGap=50)
    lines=calculate_lines(frame,hough)
    lines_visualize=visualize_lines(frame,lines)
    cv.imshow('hough',lines_visualize)
    output=cv.addWeighted(frame,0.9,lines_visualize,1,1)
    cv.imshow('output',output)
    if cv.waitKey(10) & 0xff==ord('q'):
        break
cap.release()
cv.destroyAllWindows()


