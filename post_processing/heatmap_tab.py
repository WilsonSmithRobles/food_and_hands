import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from loguru import logger
from threading import Thread

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, QFileDialog
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from .bocados import (extract_fps_from_log, 
                      remove_values_not_equal, 
                      convert_non_zero_to_255,
                      convert_values_above_5_percent_to_0,
                      compute_mask_width)

class HeatmapTab(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_dir = ""
        self.initUI()

    def initUI(self):
        heatmap_layout = QVBoxLayout()

        self.heatmap_label = QLabel()
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        heatmap_layout.addWidget(self.heatmap_label) 

        self.heatmap_progress_bar = QProgressBar()
        self.heatmap_progress_bar.setAlignment(Qt.AlignCenter)
        heatmap_layout.addWidget(self.heatmap_progress_bar)

        post_process_button_layout = QHBoxLayout()
        self.post_process_dir_label = QLabel("No directory selected")
        select_button = QPushButton("Select directory (process output)")
        post_process_button = QPushButton("Post Process!")
        post_process_button_layout.addWidget(self.post_process_dir_label)
        post_process_button_layout.addWidget(select_button)
        post_process_button_layout.addWidget(post_process_button)
        heatmap_layout.addLayout(post_process_button_layout)

        # Initially, hide
        self.setLayout(heatmap_layout)
        self.hide()

        # Conexiones con funciones
        select_button.clicked.connect(self.select_folder)
        post_process_button.clicked.connect(self.post_process_button)
        

    def select_folder(self):
        self.post_process_dir_label.setText("No directory selected")
        self.selected_dir = ""
        file = str(QFileDialog.getExistingDirectory(self, "Select directory (process output)"))
        if file:
            if not os.path.exists(os.path.join(file, "log.log")):
                logger.error("Log file does not exist.")
                return
            if not os.path.exists(os.path.join(file, "EgoHOS_Masks/")):
                logger.error("EgoHOS masks folder does not exist.")
                return
            if not os.path.exists(os.path.join(file, "FoodSeg_Masks/")):
                logger.error("FoodSeg masks folder does not exist.")
                return
            self.post_process_dir_label.setText(file)
            self.selected_dir = file

    def post_process_button(self):        
        # Start the task in a separate thread
        thread = Thread(target=self.post_process)
        thread.start()

    
    def post_process(self):
        if(self.selected_dir == ""):
            logger.error("Invalid directory for post processing")
            return

        masks_dir = os.path.join(self.selected_dir, "EgoHOS_Masks")
        log_file = os.path.join(self.selected_dir, "log.log")
        hand_tag = 2

        # Primero revisar si se puede extraer los fps
        fps = extract_fps_from_log(log_file)
        if fps == None:
            logger.error("Invalid fps count at the log file.")
            return
        msecs_frame = 2 * 1000 / fps    # · 2 porque las máscaras se guardan skipeando un frame (doblando el tiempo)

        height = 108
        width = 192
        grid_hand_count = np.zeros((height, width), dtype=np.int64)
        
        mask_files = os.listdir(masks_dir)
        mask_pngs = [filename for filename in mask_files if filename.endswith(".png")]
        mask_pngs.sort(key=lambda x: int(x.split(".")[0]))

        superpixel_height = 10
        superpixel_width = 10
        
        percent_analyzed = 0
        self.heatmap_progress_bar.setValue(percent_analyzed)
        percent_increase_per_frame = 100 / len(mask_pngs)

        bocado_frames_timeout = 1000 / msecs_frame  # 1s de timeout para contar otro bocado si se dan las condiciones
        logger.info(f"bocado frames: {bocado_frames_timeout}")
        ultimo_bocado = 0
        bocado_count = 0
        for mask_filename in mask_pngs:
            frame_number = int(mask_filename.split(".")[0])
            percent_analyzed = percent_increase_per_frame * frame_number
            self.heatmap_progress_bar.setValue(percent_analyzed)

            EgoHOS_Mask = cv2.imread(os.path.join(masks_dir, mask_filename), cv2.IMREAD_GRAYSCALE)
            
            hand_mask = remove_values_not_equal(EgoHOS_Mask, hand_tag)
            hand_mask = convert_non_zero_to_255(hand_mask)

            hand_width = compute_mask_width(hand_mask)
            bottom_hand_mask = convert_values_above_5_percent_to_0(hand_mask)
            bottom_hand_width = compute_mask_width(bottom_hand_mask)

            if bottom_hand_width > 0.95 * hand_width:   # Bocado detectado
                if frame_number - ultimo_bocado > bocado_frames_timeout:    # Cuantificar solo si hace rato que se cuantifica uno.
                    bocado_count += 1
                    logger.info(f"Bocado detectado en frame {frame_number}. Segundo {frame_number * msecs_frame / 1000}")
                ultimo_bocado = frame_number


            for i in range(grid_hand_count.shape[0]):
                for j in range(grid_hand_count.shape[1]):
                    # Get the region corresponding to the superpixel
                    superpixel_region = hand_mask[i*superpixel_height:(i+1)*superpixel_height, 
                                                    j*superpixel_width:(j+1)*superpixel_width]
                    
                    # Check if any value in the superpixel region is equal to 255
                    if np.any(superpixel_region == 255):
                        # If any 255 value is found, set the count to 1
                        grid_hand_count[i, j] += 1
            
        # Create the heatmap using imshow
        plt.imshow(grid_hand_count, cmap='hot', interpolation='nearest')
        values_gotten = np.unique(grid_hand_count)
        logger.info(f"Values gotten: {values_gotten}")
        # Add color bar to indicate the value range
        plt.colorbar()

        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Heatmap of 1s Count in Superpixels')

        # Show the plot
        plt.show()

        logger.info(f"Cuenta de cucharadas: {bocado_count}")