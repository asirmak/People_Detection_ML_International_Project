import sys
import threading
import cv2
import os

import emoji
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import requests
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from enum import Enum
from PIL import Image


PERSON_LABEL = "person"
CONFIDENCE_THRESHOLD = 0.7
FLAG = True
cropped_image_list = []
server_url = "http://130.61.137.186/getinfo"


class PeopleDetection:
    def __init__(self, model_file="yolov8n.pt"):
        self.model = YOLO(model_file)

    # Crop people from the frame and put each person in a list
    # Arrange the variable Flag such that personal card changes but main stream does not affected
    # Return the coordinates of each person in a list
    def cropping(self, frame):
        global FLAG, cropped_image_list
        coordinates = []
        results = self.model(frame)[0]
        try:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    c = box.cls
                    if self.model.names[int(c)] == PERSON_LABEL and box.conf > CONFIDENCE_THRESHOLD:
                        coordinates.append(box.xyxy[0])
                        if FLAG:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cropped_img_bgr = frame[y1:y2, x1:x2]
                            cropped_img = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)
                            cropped_image_list.append(cropped_img)

        except Exception as e:
            print(f"PeopleDetection.cropping(), Error processing frame: {e}")
        FLAG = False
        return coordinates

    # Annotate each person by using coordinates
    def detect_and_annotate(self, frame):
        coordinates = self.cropping(frame)

        annotator = Annotator(frame)

        for coord in coordinates:
            annotator.box_label(coord, PERSON_LABEL)  # Annotate the person box

        img = annotator.result()  # Get the annotated image
        return img


class JsonRead:

    @staticmethod
    def send_image_get_response(crp_image):
        global cropped_image_list

        file_path = "output_image.png"
        crp_image = cv2.cvtColor(crp_image, cv2.COLOR_BGR2RGB)
        plt.imsave(file_path, crp_image)

        with open(file_path, 'rb') as file:
            file_contents = file.read()

        files = {'file': (file_path, file_contents, 'image/png')}
        response = requests.post(server_url, files=files)

        if response.status_code == 200:
            data_x = json.loads(response.content)
            data_x[0] = json.loads(data_x[0])
            gesture_str = "Gestures: "
            fingers_str = "Fingers: "
            predicted_emotion_str = "Emotion: "
            gender_str = "Gender: "
            age_str = "Age: "

            if len(data_x[0]['gestures']) != 0:
                gesture_str += data_x[0]['gestures'][0]['name']
                if data_x[0]['gestures'][0]['name'] == "Closed_Fist":
                    gesture_str += "👊"
                elif data_x[0]['gestures'][0]['name'] == "Open_Palm":
                    gesture_str += "✋"
                elif data_x[0]['gestures'][0]['name'] == "Pointing_Up":
                    gesture_str += "☝️"
                elif data_x[0]['gestures'][0]['name'] == "Thumb_Up":
                    gesture_str += "👍"
                elif data_x[0]['gestures'][0]['name'] == "Thumb_Down":
                    gesture_str += "👎"
                elif data_x[0]['gestures'][0]['name'] == "Victory":
                    gesture_str += "✌️"
                elif data_x[0]['gestures'][0]['name'] == "ILoveYou":
                    gesture_str += "❤️"
            else:
                gesture_str += "No Detection"

            if str(data_x[0]['totalFingersAmount']) is not None:
                fingers_str += str(data_x[0]['totalFingersAmount'])
            else:
                fingers_str += "No Detection"

            if data_x[1]['predicted_emotion'] is not None:
                predicted_emotion_str += data_x[1]['predicted_emotion']
                if data_x[1]['predicted_emotion'] == "happy":
                    predicted_emotion_str += emoji.emojize(":grinning_face_with_big_eyes:")
                elif data_x[1]['predicted_emotion'] == "sad":
                    predicted_emotion_str += emoji.emojize(":disappointed_face:")
                elif data_x[1]['predicted_emotion'] == "angry":
                    predicted_emotion_str += emoji.emojize(":angry_face:")
                elif data_x[1]['predicted_emotion'] == "surprise":
                    predicted_emotion_str += "😮"
                elif data_x[1]['predicted_emotion'] == "fear":
                    predicted_emotion_str += emoji.emojize(":fearful_face:")
                elif data_x[1]['predicted_emotion'] == "disgust":
                    predicted_emotion_str += emoji.emojize(":nauseated_face:")
                elif data_x[1]['predicted_emotion'] == "neutral":
                    predicted_emotion_str += emoji.emojize(":neutral_face:")

            if (data_x[2][0]['gender']) is not None:
                gender_str += (data_x[2][0]['gender'])
                if (data_x[2][0]['gender']) == "Male":
                    gender_str += emoji.emojize(":man:")
                elif (data_x[2][0]['gender']) == "Female":
                    gender_str += emoji.emojize(":woman:")
            else:
                gender_str += "No Detection"

            if str((data_x[2][0]['age'])) is not None:
                age_str += str((data_x[2][0]['age']))
            else:
                age_str += "No Detection"

            result_str = (gesture_str + "\n" + fingers_str + "\n"
                          + predicted_emotion_str + "\n" + gender_str + "\n" + age_str)

            mainWindow.streamCroppedImage(crp_image, result_str)
        else:
            print(f"Failed to upload. Status code: {response.status_code}", response.text)


class ButtonState(Enum):
    ENABLED = 1
    DISABLED = 0


class MainWindow(QMainWindow):
    __CAMERA_FPS = 30
    __VIDEO_FPS = 33
    __BUTTON_WIDTH = 180

    def __init__(self):
        super().__init__()

        self.__mainVBoxLayout = None
        self.__stopButton = None
        self.__fileButton = None
        self.__cameraButton = None
        self.__previewLabel = None
        self.__verticalLayoutInfo_1 = None
        self.__horizontalLayout2 = None
        self.__horizontalLayout1 = None
        self.__cameraTimer = None
        self.__videoTimer = None
        self.__cameraCap = None
        self.__videoCap = None

        # Additional thread for the personal cards.
        # I add this not to slow down the main stream
        self.control_timer_thread = threading.Thread(target=self.control_flag_thread)
        self.control_timer_thread.daemon = True
        self.control_timer_thread.start()

        self.setWindowTitle("Machine Learning Project")
        self.setFixedSize(1800, 900)

        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self._initializeUI()
        self.__detection = PeopleDetection()

    # Checks in each 0.2ms that does model return a list consisting of cropped images
    @staticmethod
    def control_flag_thread():
        global FLAG, cropped_image_list
        while True:
            time.sleep(0.02)
            if not FLAG:
                for img_cropped in cropped_image_list:
                    JsonRead.send_image_get_response(img_cropped)
                FLAG = True
                cropped_image_list.clear()

    # Arrange personal cards
    def streamCroppedImage(self, crp_img, information):
        height, width, channel = crp_img.shape
        bytes_per_line = 3 * width
        q_image = QImage(crp_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        if not pixmap.isNull():
            self.__cardLabelImage_10.setPixmap(self.__cardLabelImage_9.pixmap())
            self.__cardLabelImage_10.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.__cardLabelImage_10.setScaledContents(True)
            self.__cardLabel_10.setText(self.__cardLabel_9.text())

            self.__cardLabelImage_9.setPixmap(self.__cardLabelImage_8.pixmap())
            self.__cardLabelImage_9.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.__cardLabelImage_9.setScaledContents(True)
            self.__cardLabel_9.setText(self.__cardLabel_8.text())

            self.__cardLabelImage_8.setPixmap(self.__cardLabelImage_7.pixmap())
            self.__cardLabelImage_8.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.__cardLabelImage_8.setScaledContents(True)
            self.__cardLabel_8.setText(self.__cardLabel_7.text())

            self.__cardLabelImage_7.setPixmap(self.__cardLabelImage_6.pixmap())
            self.__cardLabelImage_7.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.__cardLabelImage_7.setScaledContents(True)
            self.__cardLabel_7.setText(self.__cardLabel_6.text())

            self.__cardLabelImage_6.setPixmap(self.__cardLabelImage_5.pixmap())
            self.__cardLabelImage_6.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.__cardLabelImage_6.setScaledContents(True)
            self.__cardLabel_6.setText(self.__cardLabel_5.text())

            self.__cardLabelImage_5.setPixmap(self.__cardLabelImage_4.pixmap())
            self.__cardLabelImage_5.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.__cardLabelImage_5.setScaledContents(True)
            self.__cardLabel_5.setText(self.__cardLabel_4.text())

            self.__cardLabelImage_4.setPixmap(self.__cardLabelImage_3.pixmap())
            self.__cardLabelImage_4.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.__cardLabelImage_4.setScaledContents(True)
            self.__cardLabel_4.setText(self.__cardLabel_3.text())

            self.__cardLabelImage_3.setPixmap(self.__cardLabelImage_2.pixmap())
            self.__cardLabelImage_3.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.__cardLabelImage_3.setScaledContents(True)
            self.__cardLabel_3.setText(self.__cardLabel_2.text())

            self.__cardLabelImage_2.setPixmap(self.__cardLabelImage_1.pixmap())
            self.__cardLabelImage_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.__cardLabelImage_2.setScaledContents(True)
            self.__cardLabel_2.setText(self.__cardLabel_1.text())

            self.__cardLabelImage_1.setPixmap(pixmap)
            self.__cardLabelImage_1.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.__cardLabelImage_1.setScaledContents(True)
            self.__cardLabel_1.setText(information)

        else:
            print("Invalid image file")

    def _initializeUI(self):
        self.__horizontalLayout1 = QHBoxLayout()
        self.__horizontalLayout2 = QHBoxLayout()
        self.__verticalLayoutInfo_1 = QVBoxLayout()
        self.__verticalLayoutInfo_2 = QVBoxLayout()

        self.__previewLabel = QLabel(self.centralWidget)
        self.__previewLabel.setFixedWidth(1024)
        self.__previewLabel.setFixedHeight(768)
        self.__horizontalLayout1.addWidget(self.__previewLabel, 4)

        self.__personCardHL1 = QHBoxLayout()
        self.__personCardHL2 = QHBoxLayout()
        self.__personCardHL3 = QHBoxLayout()
        self.__personCardHL4 = QHBoxLayout()
        self.__personCardHL5 = QHBoxLayout()

        self.__personCardHL6 = QHBoxLayout()
        self.__personCardHL7 = QHBoxLayout()
        self.__personCardHL8 = QHBoxLayout()
        self.__personCardHL9 = QHBoxLayout()
        self.__personCardHL10 = QHBoxLayout()

        self.__cardLabelImage_1 = QLabel(self.centralWidget)
        self.__cardLabelImage_1.setFixedWidth(200)
        self.__cardLabelImage_1.setFixedHeight(150)

        self.__cardLabelImage_2 = QLabel(self.centralWidget)
        self.__cardLabelImage_2.setFixedWidth(200)
        self.__cardLabelImage_2.setFixedHeight(150)

        self.__cardLabelImage_3 = QLabel(self.centralWidget)
        self.__cardLabelImage_3.setFixedWidth(200)
        self.__cardLabelImage_3.setFixedHeight(150)

        self.__cardLabelImage_4 = QLabel(self.centralWidget)
        self.__cardLabelImage_4.setFixedWidth(200)
        self.__cardLabelImage_4.setFixedHeight(150)

        self.__cardLabelImage_5 = QLabel(self.centralWidget)
        self.__cardLabelImage_5.setFixedWidth(200)
        self.__cardLabelImage_5.setFixedHeight(150)

        self.__cardLabelImage_6 = QLabel(self.centralWidget)
        self.__cardLabelImage_6.setFixedWidth(200)
        self.__cardLabelImage_6.setFixedHeight(150)

        self.__cardLabelImage_7 = QLabel(self.centralWidget)
        self.__cardLabelImage_7.setFixedWidth(200)
        self.__cardLabelImage_7.setFixedHeight(150)

        self.__cardLabelImage_8 = QLabel(self.centralWidget)
        self.__cardLabelImage_8.setFixedWidth(200)
        self.__cardLabelImage_8.setFixedHeight(150)

        self.__cardLabelImage_9 = QLabel(self.centralWidget)
        self.__cardLabelImage_9.setFixedWidth(200)
        self.__cardLabelImage_9.setFixedHeight(150)

        self.__cardLabelImage_10 = QLabel(self.centralWidget)
        self.__cardLabelImage_10.setFixedWidth(200)
        self.__cardLabelImage_10.setFixedHeight(150)

        self.__cardLabel_1 = QLabel("Information Cards\n", self.centralWidget)
        self.__cardLabel_2 = QLabel("Information Cards\n", self.centralWidget)
        self.__cardLabel_3 = QLabel("Information Cards\n", self.centralWidget)
        self.__cardLabel_4 = QLabel("Information Cards\n", self.centralWidget)
        self.__cardLabel_5 = QLabel("Information Cards\n", self.centralWidget)

        self.__cardLabel_6 = QLabel("Information Cards\n", self.centralWidget)
        self.__cardLabel_7 = QLabel("Information Cards\n", self.centralWidget)
        self.__cardLabel_8 = QLabel("Information Cards\n", self.centralWidget)
        self.__cardLabel_9 = QLabel("Information Cards\n", self.centralWidget)
        self.__cardLabel_10 = QLabel("Information Cards\n", self.centralWidget)

        self.__personCardHL1.addWidget(self.__cardLabelImage_1)
        self.__personCardHL1.addWidget(self.__cardLabel_1)

        self.__personCardHL2.addWidget(self.__cardLabelImage_2)
        self.__personCardHL2.addWidget(self.__cardLabel_2)

        self.__personCardHL3.addWidget(self.__cardLabelImage_3)
        self.__personCardHL3.addWidget(self.__cardLabel_3)

        self.__personCardHL4.addWidget(self.__cardLabelImage_4)
        self.__personCardHL4.addWidget(self.__cardLabel_4)

        self.__personCardHL5.addWidget(self.__cardLabelImage_5)
        self.__personCardHL5.addWidget(self.__cardLabel_5)

        self.__personCardHL6.addWidget(self.__cardLabelImage_6)
        self.__personCardHL6.addWidget(self.__cardLabel_6)

        self.__personCardHL7.addWidget(self.__cardLabelImage_7)
        self.__personCardHL7.addWidget(self.__cardLabel_7)

        self.__personCardHL8.addWidget(self.__cardLabelImage_8)
        self.__personCardHL8.addWidget(self.__cardLabel_8)

        self.__personCardHL9.addWidget(self.__cardLabelImage_9)
        self.__personCardHL9.addWidget(self.__cardLabel_9)

        self.__personCardHL10.addWidget(self.__cardLabelImage_10)
        self.__personCardHL10.addWidget(self.__cardLabel_10)

        self.__verticalLayoutInfo_1.addLayout(self.__personCardHL1)
        self.__verticalLayoutInfo_1.addLayout(self.__personCardHL2)
        self.__verticalLayoutInfo_1.addLayout(self.__personCardHL3)
        self.__verticalLayoutInfo_1.addLayout(self.__personCardHL4)
        self.__verticalLayoutInfo_1.addLayout(self.__personCardHL5)

        self.__verticalLayoutInfo_2.addLayout(self.__personCardHL6)
        self.__verticalLayoutInfo_2.addLayout(self.__personCardHL7)
        self.__verticalLayoutInfo_2.addLayout(self.__personCardHL8)
        self.__verticalLayoutInfo_2.addLayout(self.__personCardHL9)
        self.__verticalLayoutInfo_2.addLayout(self.__personCardHL10)

        # End of Right part
        self.__horizontalLayout1.addLayout(self.__verticalLayoutInfo_1, 2)
        self.__horizontalLayout1.addLayout(self.__verticalLayoutInfo_2, 2)

        self.__cameraButton = self._createButton("Camera", "icons/glow-icon-video.png", "#4a90e2", "#357ABD",
                                                 self._cameraButtonClicked)
        self.__horizontalLayout2.addWidget(self.__cameraButton)

        self.__fileButton = self._createButton("File (Image/Video)", "icons/documents.png", "#81B622", "#59981A",
                                               self._fileButtonClicked)
        self.__horizontalLayout2.addWidget(self.__fileButton)

        self.__stopButton = self._createButton("Stop", None, "#A01D1E", "#800E13", self._stopButtonClicked,
                                               ButtonState.DISABLED)
        self.__horizontalLayout2.addWidget(self.__stopButton)

        self.__mainVBoxLayout = QVBoxLayout(self.centralWidget)
        self.__mainVBoxLayout.addLayout(self.__horizontalLayout1)
        self.__mainVBoxLayout.addLayout(self.__horizontalLayout2)
        self.centralWidget.setLayout(self.__mainVBoxLayout)

    def _createButton(self, text, icon_path, color, hover_color, slot, state=ButtonState.ENABLED):
        button = QPushButton(text, self.centralWidget)
        button.setFont(QFont("Arial", 14))
        button.clicked.connect(slot)
        button.setEnabled(state == ButtonState.ENABLED)

        if icon_path:
            button.setIcon(QIcon(icon_path))

        self._setButtonStyle(button, color, hover_color)

        return button

    def _setButtonStyle(self, button, color, hover_color):
        button.setStyleSheet(f"QPushButton {{"
                             f"border: 2px solid {color};"
                             f"border-radius: 20px;"
                             f"background-color: {color};"
                             f"color: #ffffff;"
                             f"font-size: 25px;"
                             f"padding: 10px 20px;"
                             f"text-align: center;"
                             f"min-width: {self.__BUTTON_WIDTH}px;"
                             f"font: bold 14px;"
                             f"}}"
                             f"QPushButton:hover {{"
                             f"background-color: {hover_color};"
                             f"}}"
                             f"QPushButton:pressed {{"
                             f"border: 2px solid #640D14;"
                             f"background-color: #640D14;"
                             f"}}"
                             f"QPushButton:disabled {{"
                             f"background-color: #CCCCCC;"
                             f"color: #666666;"
                             f"border: 2px solid #AAAAAA;"
                             f"}}")

    def _cameraButtonClicked(self):
        print("Camera button clicked")
        self.__stopButton.click()
        self.__cameraCap = cv2.VideoCapture(0)

        if not self.__cameraCap.isOpened():
            print("Error: Could not open camera.")
            return
        desired_width = 1024
        desired_height = 768
        self.__cameraCap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        self.__cameraCap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

        self.__cameraTimer = QTimer(self)
        self.__cameraTimer.timeout.connect(self._updateCamera)
        self.__cameraTimer.start(30)
        self.__stopButton.setEnabled(True)

    def _updateCamera(self):
        ret, frame = self.__cameraCap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            return
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.__detection.detect_and_annotate(frame)

        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.__previewLabel.setPixmap(pixmap)

    def _fileButtonClicked(self):
        print("File button clicked")
        self.__stopButton.click()
        file_dialog = QFileDialog()
        file_dialog.setWindowTitle("Open File")
        file_dialog.setNameFilter("Media Files (*.png *jpeg *.jpg *.bmp *.gif *.mp4 *.avi);;All Files (*)")
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected_file = file_dialog.selectedUrls()[0].toLocalFile()
            if selected_file.lower().endswith((".png", ".jpeg", ".jpg", ".bmp", ".gif")):
                self._displayImage(selected_file)
            elif selected_file.lower().endswith((".mp4", ".avi")):
                self._playVideo(selected_file)
            else:
                print("Unsupported file format")

    def _displayImage(self, selected_file):
        image = cv2.imread(selected_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.__detection.detect_and_annotate(image)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.__stopButton.setEnabled(True)

        if not pixmap.isNull():

            scaled_pixmap = pixmap.scaled(1024, 768, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)

            self.__previewLabel.setPixmap(scaled_pixmap)
        else:
            self.__previewLabel.setText("Invalid image file")

    def _playVideo(self, video_file):
        self.__stopButton.setEnabled(True)
        self.__videoCap = cv2.VideoCapture(video_file)
        if not self.__videoCap.isOpened():
            print("Error opening video file")
        self.__videoTimer = QTimer(self)
        self.__videoTimer.timeout.connect(self._updateVideo)
        self.__videoTimer.start(33)

    def _updateVideo(self):
        ret, frame = self.__videoCap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = self.__detection.detect_and_annotate(rgb_frame)

            height, width, channel = rgb_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.__previewLabel.setPixmap(pixmap)
            self.__previewLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.__previewLabel.setScaledContents(True)
        else:
            self.__videoTimer.stop()
            self.__videoCap.release()

    def _stopButtonClicked(self):
        if self.__cameraTimer is not None:
            self.__cameraTimer.stop()
        if self.__cameraCap is not None:
            self.__cameraCap.release()

        if self.__videoTimer is not None:
            self.__videoTimer.stop()
        if self.__videoCap is not None:
            self.__videoCap.release()
        self.__previewLabel.clear()
        self.__stopButton.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
