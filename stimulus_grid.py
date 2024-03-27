# stimulus_grid.py
import random
import sys
import time
from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout, QApplication
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor


class StimulusGrid(QWidget):
    closing_signal = pyqtSignal()
    highlighting_started = pyqtSignal()
    highlighting_finished = pyqtSignal()

    def __init__(self, selected_letter, selected_highlight, selected_round, selected_pattern):
        super().__init__()
        self.labels = None
        self.stimuli = None
        self.grid = None
        self.selected_letter = selected_letter
        self.selected_highlight = selected_highlight
        self.selected_round = selected_round
        self.selected_pattern = selected_pattern
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
                         desktop.screenGeometry().height() - taskbar_height)  # Set window geometry to screen size
        # minus taskbar height
        self.show()

        self.timer = QTimer(self)
        if self.selected_pattern == 'SC':
            self.timer.timeout.connect(self.highlight_random_location_sc)
        else:
            self.timer.timeout.connect(self.highlight_random_location_rc)
        self.timer.start(1000)  # Update every 1000 ms (1 second)

    def closeEvent(self, event):
        self.closing_signal.emit()

    def highlight_random_location_rc(self):
        # Define the label you want to highlight more often
        selected_label = [label for label in self.labels if label.text() == self.selected_letter][0]
        selected_index = self.labels.index(selected_label)
        times_to_highlight = self.selected_highlight - 1

        # Calculate the row and column indices
        row_index = selected_index // 6  # Integer division to get the row index
        col_index = selected_index % 6  # Modulo operator to get the column index

        # Create a list of rows and columns
        rows = list(range(6))
        cols = list(range(6))

        # Increase row_index and col_index by times_to_highlight and add them to the rows and cols lists
        rows += [row_index] * times_to_highlight
        cols += [col_index] * times_to_highlight

        prev_row_index = None
        prev_col_index = None

        for _ in range(self.selected_round):
            # Shuffle the order of rows and columns separately
            random.shuffle(rows)
            random.shuffle(cols)

            for _ in range(len(rows) + len(cols)):
                for label in self.labels:
                    label.setStyleSheet('color: white')

                # Highlight the row or column
                if _ % 2 == 0:
                    # Highlight the row
                    row_index = rows.pop() if rows else random.randint(0, 5)
                    # Ensure that the row index is different from the previous one
                    while row_index == prev_row_index or row_index == prev_col_index:
                        row_index = rows.pop() if rows else random.randint(0, 5)
                    prev_row_index = row_index

                    for i in range(6):
                        self.labels[row_index * 6 + i].setStyleSheet(
                            'color: yellow' if self.labels[row_index * 6 + i] == selected_label else 'color: red')
                else:
                    # Highlight the column
                    col_index = cols.pop() if cols else random.randint(0, 5)
                    # Ensure that the column index is different from the previous one
                    while col_index == prev_col_index or col_index == prev_row_index:
                        col_index = cols.pop() if cols else random.randint(0, 5)
                    prev_col_index = col_index

                    for i in range(6):
                        self.labels[i * 6 + col_index].setStyleSheet(
                            'color: yellow' if self.labels[i * 6 + col_index] == selected_label else 'color: red')

                time.sleep(1)  # Wait for 1 second
                QApplication.processEvents()  # Process pending events to update UI

        time.sleep(2)  # Wait for 1 second before final highlighting
        self.highlighting_finished.emit()  # Emit signal when highlighting finishes

    def highlight_random_location_sc(self):
        self.highlighting_started.emit()  # Emit signal when highlighting starts
        # Define how many times the target label should be highlighted more than the others
        times_to_highlight = self.selected_highlight  # Adjust this number as needed

        # Define the label you want to highlight more often
        selected_label = [label for label in self.labels if label.text() == self.selected_letter][0]
        shuffled_labels = [selected_label] * times_to_highlight + [label for label in self.labels if
                                                                   label != selected_label]

        # Shuffle the list of labels
        random.shuffle(shuffled_labels)
        total_round = self.selected_round
        # Repeat the highlighting process for three rounds
        for round_num in range(total_round):
            # Shuffle the list of labels
            random.shuffle(shuffled_labels)

            # Highlight all labels with a delay of 1 second between each highlight
            for label in shuffled_labels:
                if self.selected_letter == label.text():
                    label.setStyleSheet('color: yellow')
                else:
                    label.setStyleSheet('color: red')
                time.sleep(1)  # Wait for 1 second
                QApplication.processEvents()  # Process pending events to update UI
                label.setStyleSheet('color: white')  # Reset color to original

        self.highlighting_finished.emit()  # Emit signal when highlighting finishes

    def resizeEvent(self, event):
        for label in self.labels:
            label.setFont(
                QFont("Arial", 80) if event.oldSize().height() < event.size().height() else QFont("Arial", 80))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    stimulus_grid = StimulusGrid('A', 1, 1, 'RC')
    stimulus_grid.showMinimized()
    sys.exit(app.exec_())
