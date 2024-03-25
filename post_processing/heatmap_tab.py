import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from loguru import logger
from threading import Thread

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, QFileDialog, QMenu
from PySide6.QtCore import Qt, Signal, Slot

from .bocados import (extract_fps_from_log, 
                      remove_values_not_equal, 
                      convert_non_zero_to_255,
                      convert_values_above_5_percent_to_0,
                      compute_mask_width)
from .custom_widgets.heatmap_widget import HeatmapWidget
from .custom_widgets.error_dialog import ErrorDialog

class HeatmapTab(QWidget):
    progress_bar_update = Signal(float)

    def __init__(self):
        super().__init__()
        self.selected_dir = ""
        self.right_handed = True
        self.progress_bar_update[float].connect(self.update_progress_bar)
        self.initUI()

    def initUI(self):
        heatmap_layout = QVBoxLayout()

        self.heatmap = HeatmapWidget(self)
        heatmap_layout.addWidget(self.heatmap) 

        self.heatmap_progress_bar = QProgressBar()
        self.heatmap_progress_bar.setAlignment(Qt.AlignCenter)
        heatmap_layout.addWidget(self.heatmap_progress_bar)

        post_process_button_layout = QHBoxLayout()
        self.post_process_dir_label = QLabel("No directory selected")
        select_button = QPushButton("Select directory (process output)")
        post_process_button = QPushButton("Post Process!")

        # Elegir zurdo o diestro
        self.dropdown_button = QPushButton('Right hand', self)
        menu = QMenu(self)
        right_hand_action = menu.addAction('Right handed')
        left_hand_action = menu.addAction('Left handed')
        right_hand_action.triggered.connect(self.right_hand_selected)
        left_hand_action.triggered.connect(self.left_hand_selected)
        self.dropdown_button.setMenu(menu)

        post_process_button_layout.addWidget(self.post_process_dir_label)
        post_process_button_layout.addWidget(select_button)
        post_process_button_layout.addWidget(self.dropdown_button)
        post_process_button_layout.addWidget(post_process_button)
        heatmap_layout.addLayout(post_process_button_layout)

        # Initially, hide
        self.setLayout(heatmap_layout)
        self.hide()

        # Conexiones con funciones
        select_button.clicked.connect(self.select_folder)
        post_process_button.clicked.connect(self.post_process_button)
        

    @Slot(float)
    def update_progress_bar(self, progress):
        self.heatmap_progress_bar.setValue(progress)

    def right_hand_selected(self):
        self.right_handed = True
        self.dropdown_button.setText('Right handed')

    def left_hand_selected(self):
        self.right_handed = False
        self.dropdown_button.setText('Left handed')

    def select_folder(self):
        self.post_process_dir_label.setText("No directory selected")
        self.selected_dir = ""
        file = str(QFileDialog.getExistingDirectory(self, "Select directory (process output)"))
        if file:
            if not os.path.exists(os.path.join(file, "log.log")):
                ErrorDialog("Log file does not exist.")
                return
            if not os.path.exists(os.path.join(file, "EgoHOS_Masks/")):
                ErrorDialog("EgoHOS masks folder does not exist.")
                return
            if not os.path.exists(os.path.join(file, "FoodSeg_Masks/")):
                ErrorDialog("FoodSeg masks folder does not exist.")
                return
            self.post_process_dir_label.setText(file)
            self.selected_dir = file

    def post_process_button(self):        
        # Start the task in a separate thread
        thread = Thread(target=self.post_process)
        thread.start()

    def post_process(self):
        hand_tag = 2 if self.right_handed else 1
        if not os.path.exists(self.selected_dir):
            ErrorDialog("Invalid directory for post processing")
            return
        
        if (os.path.exists(os.path.join(self.selected_dir, "post_process.log")) 
            and os.path.exists(os.path.join(self.selected_dir, "heatmap.jpg"))):
            saved_image = cv2.imread(os.path.join(self.selected_dir, "heatmap.jpg"), cv2.IMREAD_GRAYSCALE)
            grid_hand_count = saved_image.astype(np.int64)
            self.heatmap.plot(grid_hand_count)
            return
        
        if not os.path.exists(os.path.join(self.selected_dir, "log.log")):
            ErrorDialog("Log file does not exist.")
            return
        log_file = os.path.join(self.selected_dir, "log.log")
        
        if not os.path.exists(os.path.join(self.selected_dir, "EgoHOS_Masks/")):
            ErrorDialog("EgoHOS masks folder does not exist.")
            return
        masks_dir = os.path.join(self.selected_dir, "EgoHOS_Masks")

        if not os.path.exists(os.path.join(self.selected_dir, "FoodSeg_Masks/")):
            ErrorDialog("FoodSeg masks folder does not exist.")
            return


        # Primero revisar si se puede extraer los fps
        fps = extract_fps_from_log(log_file)
        if fps == None:
            ErrorDialog("Invalid fps count at the log file.")
            return
        msecs_frame = 2 * 1000 / fps    # · 2 porque las máscaras se guardan skipeando un frame (doblando el tiempo)

        result_logger = logger.bind(application="food_and_hands_post_process")
        result_logger.add(sink=os.path.join(self.selected_dir, "post_process.log"), rotation="10 MB")

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
        result_logger.info(f"bocado frames: {bocado_frames_timeout}")
        ultimo_bocado = 0
        bocado_count = 0
        for mask_filename in mask_pngs:
            frame_number = int(mask_filename.split(".")[0])
            percent_analyzed = percent_increase_per_frame * frame_number
            # self.heatmap_progress_bar.setValue(percent_analyzed)
            self.progress_bar_update.emit(percent_analyzed)

            EgoHOS_Mask = cv2.imread(os.path.join(masks_dir, mask_filename), cv2.IMREAD_GRAYSCALE)
            
            hand_mask = remove_values_not_equal(EgoHOS_Mask, hand_tag)
            hand_mask = convert_non_zero_to_255(hand_mask)

            hand_width = compute_mask_width(hand_mask)
            bottom_hand_mask = convert_values_above_5_percent_to_0(hand_mask)
            bottom_hand_width = compute_mask_width(bottom_hand_mask)

            if bottom_hand_width > 0.95 * hand_width:   # Bocado detectado
                if frame_number - ultimo_bocado > bocado_frames_timeout:    # Cuantificar solo si hace rato que se cuantifica uno.
                    bocado_count += 1
                    result_logger.info(f"Bocado detectado en frame {frame_number}. Segundo {frame_number * msecs_frame / 1000}")
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
        
        cv2.imwrite(os.path.join(self.selected_dir, "heatmap.jpg"), grid_hand_count)
        values_gotten = np.unique(grid_hand_count)
        result_logger.info(f"Valores obtenidos: {values_gotten}")
        result_logger.info(f"Cuenta de cucharadas: {bocado_count}")
        self.heatmap.plot(grid_hand_count)