import csv
import sys
import time
from datetime import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from PyQt5.QtCore import QThread


class DataAcquisitionThread(QThread):
    def __init__(self, serial_port, board_id, letter):
        super().__init__()
        self.board = None
        self.serial_port = serial_port
        self.board_id = board_id
        self.letter = letter
        self._running = True

    def run(self):
        params = BrainFlowInputParams()
        params.board_id = self.board_id
        params.serial_port = self.serial_port
        self.board = BoardShim(params.board_id, params)
        self.board.prepare_session()
        self.board.start_stream()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"data_{timestamp}.csv"

        with open(filename, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Add a column for letters
            eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            headers = [channel for channel in eeg_channels]
            headers.append("Letter")  # Add a new header for the letter column
            writer.writerow(headers)

            try:
                while self._running:
                    data = self.board.get_current_board_data(256)
                    for i in range(len(data[0])):
                        row = [data[channel][i] for channel in range(32)]
                        row.append(self.letter)  # Append the same letter for each row
                        writer.writerow(row)

                    time.sleep(1)
            finally:
                self.board.stop_stream()
                self.board.release_session()

    def stop(self):
        self._running = False
