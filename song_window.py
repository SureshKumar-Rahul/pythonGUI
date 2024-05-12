import os
import sys
import pygame
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QComboBox
from PyQt5.QtCore import QTimer
from brainflow import BoardIds
from PyQt5 import QtGui

from data_acquisition_thread_music import DataAcquisitionThread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Data Collection with Music Player")
        self.setGeometry(300, 300, 800, 800)

        self.tracks_folder = "audio"  # Folder containing audio tracks
        self.image_folder = "audio/images"  # Folder containing track images
        self.tracks = self.load_tracks()
        self.current_track_index = 0
        self.break_duration = 15  # Break duration in seconds
        self.data_acquisition_thread = None
        self.break_timer = QTimer(self)
        self.track_timer = QTimer(self)
        self.data_acquisition_thread = None  # Initialize EEG data acquisition thread
        self.song_duration = 30
        self.setup_ui()

    def setup_ui(self):
        # Play button
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.play_music)
        self.play_button.setGeometry(50, 50, 100, 50)

        # Pause button
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause_music)
        self.pause_button.setGeometry(200, 50, 100, 50)

        # Stop button
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_music)
        self.stop_button.setGeometry(350, 50, 100, 50)

        # Subject label and combo box
        self.subject_label = QLabel('Subject:', self)
        self.subject_label.setGeometry(50, 120, 100, 30)
        self.subject = QComboBox(self)
        self.subject.setGeometry(150, 120, 100, 30)
        self.subject.addItems(['0', '1', '2', '3', '4', '5'])

        self.song_duration_label = QLabel('Song Duration:', self)
        self.song_duration_label.setGeometry(50, 200, 100, 30)
        self.song_time = QComboBox(self)
        self.song_time.setGeometry(150, 200, 100, 30)
        self.song_time.addItems(['30', '60'])

        # Status label
        self.status_label = QLabel("Ready", self)
        self.status_label.setGeometry(50, 300, 500, 30)

        # Break label
        self.break_label = QLabel("", self)
        self.break_label.setGeometry(50, 350, 500, 30)

        # Image label
        self.image_label = QLabel(self)
        self.image_label.setGeometry(200,400, 400, 400)  # Adjust position and size as needed

        self.break_timer.timeout.connect(self.update_break_time)
        self.track_timer.timeout.connect(self.play_next_track)

    def update_visualizer(self, eeg_data):
        # Update the visualizer with EEG data
        self.eeg_visualizer.update_plot(eeg_data)

    def load_tracks(self):
        tracks = []
        for file in os.listdir(self.tracks_folder):
            if file.endswith(".mp3"):
                tracks.append(os.path.join(self.tracks_folder, file))

        return tracks

    def play_music(self):
        pygame.mixer.init()
        self.play_next_track()

    def play_next_track(self):
        if self.current_track_index < len(self.tracks):
            self.break_label.clear()
            self.track_timer.stop()
            track = self.tracks[self.current_track_index]
            image_path = os.path.join(self.image_folder, f"img{self.current_track_index}.jpg")
            self.set_image(image_path)
            pygame.mixer.music.load(track)
            pygame.mixer.music.play()
            self.status_label.setText(f"Playing: {track}")
            # Start EEG data acquisition when a track starts playing
            if self.data_acquisition_thread is None or not self.data_acquisition_thread.isRunning():
                self.data_acquisition_thread = DataAcquisitionThread(serial_port="COM5",
                                                                     board_id=2,
                                                                     subject=int(self.subject.currentText()),
                                                                     current_track_index=self.current_track_index)

                self.data_acquisition_thread.start()

            self.current_track_index += 1
            self.song_duration = int(self.song_time.currentText())
            self.break_timer.start(self.song_duration * 1000)  # Start break timer
        else:
            self.status_label.setText("All tracks played")
            self.break_label.setText("Ready for another round?")
            if self.data_acquisition_thread and self.data_acquisition_thread.isRunning():
                self.data_acquisition_thread.stop()

    def update_break_time(self):
        self.break_timer.stop()
        self.stop_music()
        self.break_label.setText(f"Taking a break")
        self.track_timer.start(self.break_duration * 1000)

        # Stop EEG data acquisition during the break
        if self.data_acquisition_thread and self.data_acquisition_thread.isRunning():
            self.data_acquisition_thread.stop()

    def pause_music(self):
        pygame.mixer.music.pause()
        self.status_label.setText("Paused")

    def stop_music(self):
        pygame.mixer.music.stop()
        self.status_label.setText("Stopped")
        self.break_timer.stop()
        self.break_label.clear()
        self.image_label.setPixmap(QtGui.QPixmap())
        # Stop EEG data acquisition when music is stopped
        if self.data_acquisition_thread and self.data_acquisition_thread.isRunning():
            self.data_acquisition_thread.stop()

    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(350, 350))  # Adjust size as needed


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
