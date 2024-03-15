# stimulus_grid.py
import random
import time
from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout, QApplication
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor


class StimulusGrid(QWidget):
    closing_signal = pyqtSignal()
    highlighting_started = pyqtSignal()
    highlighting_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.labels = None
        self.stimuli = None
        self.grid = None
        self.initUI()

    def initUI(self):
        self.grid = QGridLayout()
        self.setLayout(self.grid)

        # Create labels for each stimulus
        self.stimuli = ['A', 'B', 'C', 'D', 'E', 'F',
                        'G', 'H', 'I', 'J', 'K', 'L',
                        'M', 'N', 'O', 'P', 'Q', 'R',
                        'S', 'T', 'U', 'V', 'W', 'X',
                        'Y', 'Z', '1', '2', '3', '4',
                        '5', '6', '7', '8', '9', '-']

        self.labels = []
        for row in range(6):
            for col in range(6):
                if row * 6 + col < len(self.stimuli):
                    label = QLabel(self.stimuli[row * 6 + col])  # Assign the letter to the label
                    label.setAlignment(Qt.AlignCenter)
                    self.grid.addWidget(label, row, col)
                    self.labels.append(label)

        self.setWindowTitle('Farewell and Donchin\'s P300 Speller Paradigm')
        self.setStyleSheet("background-color: black; color: white;")
        desktop = QApplication.desktop()  # Get desktop widget
        taskbar_height = desktop.height() - desktop.availableGeometry().height()  # Calculate height of the taskbar
        self.setGeometry(0, 0, desktop.screenGeometry().width(),
                         desktop.screenGeometry().height() - taskbar_height)  # Set window geometry to screen size minus taskbar height
        self.show()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.highlight_random_location)
        self.timer.start(1000)  # Update every 1000 ms (1 second)

    def closeEvent(self, event):
        self.closing_signal.emit()

    def highlight_random_location(self):
        self.highlighting_started.emit()  # Emit signal when highlighting starts

        # Repeat the highlighting process for three rounds
        for round_num in range(3):
            # Reset all labels to default style
            for label in self.labels:
                label.setStyleSheet('color: white')

            # Shuffle the list of labels
            random.shuffle(self.labels)

            # Highlight all labels with a delay of 1 second between each highlight
            for label in self.labels:
                label.setStyleSheet('color: red')
                time.sleep(1)  # Wait for 1 second
                QApplication.processEvents()  # Process pending events to update UI
                label.setStyleSheet('color: white')  # Reset color to original

        self.highlighting_finished.emit()  # Emit signal when highlighting finishes

    def resizeEvent(self, event):
        for label in self.labels:
            label.setFont(QFont("Arial", 80) if event.oldSize().height() < event.size().height() else QFont("Arial", 80))
