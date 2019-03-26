import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QSize
 
class App(QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.title = 'Home Screen'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.nameLabel = QLabel(self)
        self.nameLabel.setText('Twitter Handle:')
        self.line = QLineEdit(self)
        self.line.move(630, 300)
        self.line.resize(200, 32)
        self.nameLabel.move(555, 300)
        
        button = QPushButton('Check Personality', self)
        button.setToolTip('Retrieving Data')
        button.move(640,400)
        button.clicked.connect(self.on_click)
        self.show()
    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox.text()
        QMessageBox.question(self, 'Personality predictor alert!', "You typed: " + textboxValue, QMessageBox.Ok, QMessageBox.Ok)
        self.textbox.setText("")
 
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
