from PyQt5.QtWidgets import QWidget, QPushButton, QComboBox
from PyQt5.QtCore import pyqtSlot, QTimer
from brainflow import BoardIds

from stimulus_grid import StimulusGrid
from data_acquisition_thread import DataAcquisitionThread


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.stimulus_grid = None
        self.data_acquisition_thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('P300 Speller Controller')
        self.setGeometry(100, 100, 300, 150)

        self.btn_start = QPushButton('Start', self)
        self.btn_start.clicked.connect(self.start_p300_speller)
        self.btn_start.setGeometry(50, 50, 100, 30)

        # Add a dropdown list (combobox) for selecting letters
        self.combo_letters = QComboBox(self)
        self.combo_letters.setGeometry(170, 50, 100, 30)
        self.combo_letters.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
        self.show()

    @pyqtSlot()
    def start_p300_speller(self):
        selected_letter = self.combo_letters.currentText()  # Get the selected letter from the combobox
        self.stimulus_grid = StimulusGrid()
        self.stimulus_grid.showMaximized()
        # Pass the selected letter to the constructor of DataAcquisitionThread
        self.data_acquisition_thread = DataAcquisitionThread(serial_port="COM5",
                                                             board_id=BoardIds.SYNTHETIC_BOARD.value,
                                                             letter=selected_letter)
        self.data_acquisition_thread.start()
        self.stimulus_grid.closing_signal.connect(self.stop_data_acquisition)
        self.stimulus_grid.closing_signal.connect(self.close_all_windows)
        self.stimulus_grid.highlighting_finished.connect(self.stop_data_acquisition)
        self.stimulus_grid.highlighting_finished.connect(self.close_all_windows)

    def stop_data_acquisition(self):
        if self.data_acquisition_thread:
            # Create a QTimer with a single-shot mode
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self.data_acquisition_thread.stop)

            # Start the timer with a 500 ms (half-second) delay
            timer.start(500)
    def close_all_windows(self):
        if self.stimulus_grid:
            self.stimulus_grid.close()
        if self.data_acquisition_thread:
            self.data_acquisition_thread.stop()
        self.close()
