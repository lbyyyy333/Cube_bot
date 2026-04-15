#camera_node
控制机械臂拍三张照片覆盖魔方的六面
#image_process_node
就是对camera_node来的图片的处理
图片-》roi-》kmean->颜色识别-》放进各种容器里面来回转化成kociemba可以用的string
#kociemba_node
直接用了python自带的kc库