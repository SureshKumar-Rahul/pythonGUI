from PyQt5.QtWidgets import QWidget, QPushButton, QComboBox, QLabel
from PyQt5.QtCore import pyqtSlot, QTimer
from brainflow import BoardIds

from stimulus_grid import StimulusGrid
from data_acquisition_thread import DataAcquisitionThread


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.label_letters = None
        self.label_highlights = None
        self.label_rounds = None
        self.btn_start = None
        self.combo_letters = None
        self.combo_highlights = None
        self.combo_rounds = None
        self.stimulus_grid = None
        self.data_acquisition_thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('P300 Speller Controller')
        self.setGeometry(100, 100, 400, 400)

        # Select Letter dropdown
        self.label_letters = QLabel('Select Letter:', self)
        self.label_letters.setGeometry(50, 50, 100, 30)
        self.combo_letters = QComboBox(self)
        self.combo_letters.setGeometry(200, 50, 70, 30)
        self.combo_letters.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

        # Select No. of Highlights dropdown
        self.label_highlights = QLabel('Select No. of Highlights:', self)
        self.label_highlights.setGeometry(50, 100, 150, 30)
        self.combo_highlights = QComboBox(self)
        self.combo_highlights.setGeometry(200, 100, 70, 30)
        self.combo_highlights.addItems(['1', '2', '3', '4', '5'])

        # Select No. of Rounds dropdown
        self.label_rounds = QLabel('Select No. of Rounds:', self)
        self.label_rounds.setGeometry(50, 150, 150, 30)
        self.combo_rounds = QComboBox(self)
        self.combo_rounds.setGeometry(200, 150, 70, 30)
        self.combo_rounds.addItems(['1', '2', '3', '4', '5'])

        # Start button
        self.btn_start = QPushButton('Start', self)
        self.btn_start.setGeometry(150, 250, 100, 30)
        self.btn_start.clicked.connect(self.start_p300_speller)

        self.show()

    @pyqtSlot()
    def start_p300_speller(self):
        selected_letter = self.combo_letters.currentText()  # Get the selected letter from the combobox
        selected_highlight = int(self.combo_highlights.currentText())
        selected_round = int(self.combo_rounds.currentText())
        self.stimulus_grid = StimulusGrid(selected_letter, selected_highlight, selected_round)
        self.stimulus_grid.showMaximized()
        # Pass the selected letter to the constructor of DataAcquisitionThread
        self.data_acquisition_thread = DataAcquisitionThread(serial_port="COM5",
                                                             board_id=2,
                                                             letter=selected_letter)
        self.data_acquisition_thread.start()
        self.stimulus_grid.closing_signal.connect(self.close_all_windows)
        self.stimulus_grid.highlighting_finished.connect(self.close_all_windows)


    def close_all_windows(self):
        if self.stimulus_grid:
            self.stimulus_grid.close()
        if self.data_acquisition_thread:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self.data_acquisition_thread.stop)
            timer.start(500)
        self.close()
