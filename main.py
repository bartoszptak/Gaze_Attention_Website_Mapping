import sys
import os
import numpy as np
import cv2

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *

from Heat_Mapper import Heat_Mapper
from Gaze_Tracker import Gaze_Tracker

project_name = 'Gaze Tracker - web-page layout attention map'
homepage = 'https://put.poznan.pl/'


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.isMonitoring = False
        self.setWindowTitle(project_name)
        self.setWindowFlags(Qt.WindowTitleHint)
        self.showMaximized()
        self.setWindowIcon(QIcon('data/UX/eye_log.png'))

        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl(homepage))

        self.setCentralWidget(self.browser)

        self.create_navbar()

        self.browser.urlChanged[QUrl].connect(self.update_urlbar)
        self.urlbar.returnPressed.connect(self.navigate_to_url)

        self.shortcut = QShortcut(QKeySequence("Space"), self)
        self.shortcut.activated.connect(self.spaceClick)

        self.pix = None
        self.map = None
        self.gt = Gaze_Tracker()
        self.cap = None

    def update_urlbar(self, q):
        self.urlbar.setText(q.toString())
        self.urlbar.setCursorPosition(0)

    def navigate_to_url(self):
        q = QUrl(self.urlbar.text())
        if q.scheme() == '':
            q.setScheme('https')
        self.browser.setUrl(q)

    def create_navbar(self):
        self.navbar = QToolBar('Navigation')
        self.navbar.setIconSize(QSize(32, 32))
        self.navbar.setMovable(False)
        self.addToolBar(self.navbar)

        self.create_urlbar()
        self.create_calibrate_button()
        self.create_start_button()
        self.create_stop_button()
        self.create_about_button()
        self.create_exit_button()

    def create_urlbar(self):
        self.urlbar = QLineEdit()
        self.urlbar.setText(homepage)
        f = self.urlbar.font()
        f.setPointSize(24)
        self.urlbar.setFont(f)
        self.navbar.addSeparator()
        self.navbar.addWidget(self.urlbar)

    def create_calibrate_button(self):
        self.calibration_button = QAction(
            QIcon(os.path.join('data', 'UX', 'headhunting.png')), 'Calibration', self)
        self.calibration_button.setStatusTip('Calibration')
        self.calibration_button.triggered.connect(self.button_calibrate_click)
        self.navbar.addAction(self.calibration_button)
        self.navbar.addSeparator()

    def create_start_button(self):
        self.start_button = QAction(QIcon(os.path.join(
            'data', 'UX', 'eye.png')), 'Start monitoring (press spacebar)', self)
        self.start_button.setStatusTip('Start monitoring')
        self.start_button.setEnabled(False)
        self.start_button.triggered.connect(self.button_start_click)
        self.navbar.addAction(self.start_button)
        self.navbar.addSeparator()

    def create_stop_button(self):
        self.stop_button = QAction(QIcon(os.path.join(
            'data', 'UX', 'hide.png')), 'Stop monitoring (press spacebar)', self)
        self.stop_button.setStatusTip('Stop monitoring')
        self.stop_button.setEnabled(False)
        self.stop_button.triggered.connect(self.button_stop_click)
        self.navbar.addAction(self.stop_button)
        self.navbar.addSeparator()

    def create_about_button(self):
        self.about_button = QAction(
            QIcon(os.path.join('data', 'UX', 'info.png')), 'Help', self)
        self.about_button.setStatusTip('Help')
        self.about_button.triggered.connect(self.button_about_click)
        self.navbar.addAction(self.about_button)

    def create_exit_button(self):
        self.exit_button = QAction(QIcon(os.path.join(
            'data', 'UX', 'error.png')), 'Exit', self, triggered=self.close)
        self.exit_button.setStatusTip('Exit program')
        self.navbar.addAction(self.exit_button)

    def spaceClick(self):
        if self.gt.cal:
            if self.isMonitoring:
                self.button_stop_click()
            else:
                self.button_start_click()

    def button_start_click(self):
        self.isMonitoring = True
        self.browser_control()
        self.pix = QPixmap(self.browser.size())
        self.map = Heat_Mapper(self.pix.size().width(),
                               self.pix.size().height())
        self.browser.render(self.pix)

        w, h = self.size().width(), self.size().height()

        while self.isMonitoring:
            _, image = self.cap.read()
            cor = self.gt.predict_position(image, w, h)
            if cor:
                self.map.increment_value(cor[0], cor[1])
            QCoreApplication.processEvents()

    def QPixmap_to_CvImage(self, pixmap):
        incomingImage = pixmap.toImage().convertToFormat(
            QImage.Format_RGB888).rgbSwapped()
        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        return np.array(ptr, np.uint8).reshape(incomingImage.height(), incomingImage.width(), 3)

    def CvImage_to_QPixmap(self, cvImg):
        height, width, _ = cvImg.shape
        bytesPerLine = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerLine,
                      QImage.Format_RGB888).rgbSwapped()
        return QPixmap(qImg)

    def button_stop_click(self):
        self.isMonitoring = False
        self.browser_control()

        cv_image = self.QPixmap_to_CvImage(self.pix)

        connecting = self.map.appyling_heatmap(cv_image)

        qmap = self.CvImage_to_QPixmap(connecting)
        dialog = Result(self, pix=qmap)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.show()

    def button_calibrate_click(self):
        dialog = Calibrate(self)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.show()
        self.browser_control()

    def button_about_click(self):
        dialog = About(self)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.show()

    def browser_control(self):
        if not self.gt.cal:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
        else:
            if self.isMonitoring:
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.browser.setEnabled(False)
                self.urlbar.setEnabled(False)
                self.calibration_button.setEnabled(False)
            else:
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.browser.setEnabled(True)
                self.urlbar.setEnabled(True)
                self.calibration_button.setEnabled(True)


class Result(QMainWindow):
    def __init__(self, parent=None, pix=None):
        super(Result, self).__init__(parent)

        self.pix = pix
        self.setWindowTitle('Result - ' + project_name)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        lay = QVBoxLayout(central_widget)

        save_button = QPushButton("Save result", self)
        save_button.clicked.connect(self.save)

        lay.addWidget(save_button)

        label = QLabel(self)
        self.pix = QPixmap.scaled(self.pix, int(
            self.pix.width() * 2/3), int(self.pix.height() * 2/3))
        label.setPixmap(self.pix)
        self.resize(int(self.pix.width() * 2/3), int(self.pix.height() * 2/3))
        lay.addWidget(label)

    def save(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                   "PNG Files (*.png)", options=options)
        if file_name:
            self.pix.save(file_name, 'png')


class About(QMainWindow):
    def __init__(self, parent=None):
        super(About, self).__init__(parent)

        self.setWindowTitle('About - ' + project_name)
        self.resize(400, 300)
        self.setFixedSize(400, 300)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        lay = QVBoxLayout(central_widget)

        label = QLabel(self)
        pixmap = QPixmap('data/UX/about.png')
        label.setPixmap(pixmap)
        lay.addWidget(label)


class Calibrate(QMainWindow):
    def __init__(self, parent=MainWindow):
        super(Calibrate, self).__init__(parent)
        self.setWindowTitle('Calibration - ' + project_name)
        self.count = 0
        self.x = parent.size().width()
        self.y = parent.size().height()

        capture = self.capture_combo()

        self.shortcut = QShortcut(QKeySequence("Space"), self)
        self.shortcut.activated.connect(self.clickMethod)

        self.pybutton = QPushButton('Press to start\n(press spacebar)', self)
        self.pybutton.clicked.connect(self.clickMethod)
        self.pybutton.resize(120, 60)

        self.pybutton.move(self.x/2 - 60, self.y/2 - 30)

        self.showMaximized()

        self.cap = cv2.VideoCapture(capture)

        self.flags = [False, False, False, False, False, False]

    def capture_combo(self):
        captures = self.check_available_captures()

        label = QLabel(self)
        label.setText("Select camera input:")
        label.move(50, 30)

        combo = QComboBox(self)

        for el in captures:
            combo.addItem(str(int(el)))

        combo.move(50, 60)
        return int(str(combo.currentText()))

    def clickMethod(self):
        _, image = self.cap.read()

        if self.count == 0:
            window.gt.reset()
            window.gt.cal = True
            self.pybutton.move(60, self.y / 2 - 30)
            self.pybutton.setText('Look && Press\n(press spacebar)')
            if not self.flags[5]:
                window.gt.test_cal_points(image)
                self.flags[5] = True
            self.count += 1
        elif self.count == 1:
            if not self.flags[0]:
                window.gt.add_to_cal_points(image)
                self.flags[0] = True
            self.pybutton.move(self.x - 60 - 120, self.y / 2 - 30)
            self.count += 1
        elif self.count == 2:
            if not self.flags[1]:
                window.gt.add_to_cal_points(image)
                self.flags[1] = True
            self.pybutton.move(self.x/2 - 60, 30)
            self.count += 1
        elif self.count == 3:
            if not self.flags[2]:
                window.gt.add_to_cal_points(image)
                self.flags[2] = True
            self.pybutton.move(self.x/2 - 60, self.y - 30 - 60)
            self.count += 1
        elif self.count == 4:
            if not self.flags[3]:
                window.gt.add_to_cal_points(image)
                self.flags[3] = True
            self.pybutton.move(self.x / 2 - 60, self.y / 2 - 30)
            self.pybutton.setText('Press to end\n(press spacebar)')
            self.count += 1
        else:
            if not self.flags[4]:
                window.gt.add_to_cal_points(image)
                self.flags[4] = True
            window.cap = self.cap
            window.browser_control()
            self.close()

    def check_available_captures(self):
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        return arr


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
