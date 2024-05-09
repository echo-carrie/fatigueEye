

# 清醒派

  <p align="center">
  <a href="https://github.com/shaojintian/Best_README_template/">
    <img src="https://i.postimg.cc/3Npy9sgc/image.png" alt="Logo" width="80" height="80">
  </a>
<h3 align="center">清醒派</h3>
  <p align="center">
    基于面部特征的疲劳驾驶智能识别系统
    <br />
  </p>

## 项目简介

​	“清醒派”系统是一款基于深度学习技术和flask框架的可提供准确、实时的疲劳驾驶检测的疲劳驾驶智能识别系统。 结合计算机视觉和神经科学领域的最新研究成果，并收集和分析驾驶员面部数据，训练出准确的人脸识别检测疲劳驾驶状态模型，能够有效地监测和识别疲劳驾驶的迹象。

​	在功能上，主要有实时检测、虚拟智行领航员、导航至休息点、生成检测报表、播放脑波音乐、上传文件检测六大功能。能够及时识别出驾驶员疲劳状态，并采取有效的缓解疲劳与避免疲劳驾驶带来危险的措施。

​	在设计上，网页端设计采用直观、简洁的用户界面，使驾驶员能够轻松理解和操作系统，不会增加额外的驾驶压力，并且增加了一个虚拟智行领航员，能够方便驾驶员在驾驶中能通过语音进行功能的使用。

​	在性能上，系统能够高效地处理大量的数据，并通过人脸识别疲劳驾驶检测技术进行实时分析。经过测评，系统在长时间运行中不会出现崩溃或错误，能够应用于长途驾驶。

​	“清醒派”系统相较于传统的疲劳驾驶检测系统来说，具有便利化、智能化、高效化的特点。在驾驶过程中使用该系统，能够及时监测到疲劳状态，并做出警报与缓解疲劳的相应措施，能够有效减少因疲劳驾驶而产生的事故，为公共交通安全增添一道坚实的防线，为乘客和驾驶员的生命安全保驾护航。

![image-20240509170130077](https://i.postimg.cc/nc3xgd0R/image.png)

## 部分界面

### 首页界面

![img](https://i.postimg.cc/rmGXJgD9/image.png)

###  功能界面

展示了系统提供的四大服务，分别是 “实时检测”、“导航”、“检测报告”、“文件上传”，点击方框可以进行跳转。

![img](https://i.postimg.cc/3NnMMRC4/image.png) 

### 导航界面

​	在导航页面会对当前用户所在的位置进行定位，若需要导航到附近的停车点，则可以点击“搜索周边位置”按钮查找附近的停车休息点，点击后页面会显示附近的停车点名称以及位置信息。

​	用户点击要去的停车点，再点击“规划驾车路线”，可以进行路线规划与导航，若用户想要退出导航，则可以点击“退出导航”按钮

![image-20240509170249612](https://i.postimg.cc/sDgdYYsp/image.png)

### 检测界面

​	检测界面的中部为人脸识别界面，通过打开摄像头可以显示驾驶员的实时面部图像。系统对驾驶员的面部特征进行识别和分析。高效地检测驾驶员的口腔宽窄比、眼宽比、头部倾斜、闭眼率指标，在发现驾驶员闭眼后会发出警告。

​	在界面右边展示了口腔宽窄比、眼宽比、头部倾斜、闭眼率四个实时检测到的面部数据

​	在界面的右边能够显示行驶距离记录与位置信息，放置了播放能够缓解疲劳的脑波音乐按钮，最下方还有“导航到停车场”按钮，用户点击后可以一键导航至附近的停车场。

![img](https://i.postimg.cc/VkJxTzSv/image.png) 

图 24：检测页面

### 报告界面

若今日没有打开疲劳检测器，则会提示用户打开使用。

![img](file:///C:\Users\12253\AppData\Local\Temp\ksohtml25048\wps11.jpg) 

若打开了检测器则会展示今日检测到的数据

![img](https://i.postimg.cc/FK83ZgM1/image.png) 



## 项目下载

```sh
git clone https://github.com/echo-carrie/
git clone https://github.com/echo-carrie/Wisegaze.git
cd Wisegaze.
```

## 文件目录说明

```
D:.
│  .all-contributorsrc
│  .gitattributes
│  .gitignore
│  app.py
│  Procfile
│  README.md
│  requirements.txt
│
├─fatigue
│  │  fatigue.py
│  │  shape_predictor_68_face_landmarks.dat
│  │  yawn.py
│  │
│  └─sounds
│          alarm.wav
│          alarm2.mp3
│
├─files
│      BLINK.txt
│      EAR.txt
│      YAWN.txt
│
├─screenshots
│      graph.PNG
│      main.PNG
│      report.PNG
│      reportpage.PNG
│      tip.PNG
│      tip2.PNG
│
├─static
│  │  main.css
│  │  script.js
│  │
│  └─media
│          about.svg
│          background.webm
│          home.jpg
│          loader.svg
│          refresh.svg
│          sleep.svg
│          tired.svg
│
├─templates
│      about.html
│      base.html
│      index.html
│      report.html
│      video.html
│
└─utils
    │  quotes.py
    │
    └─__pycache__
            quotes.cpython-39.pyc
```

## 使用到的框架及组件

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [PyTorch](https://pytorch.org)
- [jQuery](https://jquery.com)
- [Flask](https://flask.github.net.cn)
- [高德地图](https://github.com/ultralytics/yolov5)

## 作者

华南师范大学 陈秋羽

华南师范大学 林榘驰

华南师范大学 郭泳童

华南师范大学 杨嘉仪

华南师范大学 谭沛轩

华南师范大学 李文洁

## 鸣谢


- [SEED-VIG-CNN: 针对上海交大SSED-VIG数据集做的一个融合脑电和眼电的特征信息推断是否疲劳驾驶的网络 ](https://github.com/dttutty/SEED-VIG-CNN)
- [stevenjoezhang/live2d-widget: 把萌萌哒的看板娘抱回家 (ノ≧∇≦)ノ | Live2D widget for web platform](https://github.com/stevenjoezhang/live2d-widget)

[项目地址]:https://github.com/echo-carrie/fatigueEye	"清醒派"



