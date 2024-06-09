import os
import numpy as np
import cv2
import json
from fastai.vision.all import load_learner, Path, Image
import matplotlib.pyplot as plt
from loguru import logger
from threading import Thread

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, QFileDialog, QMenu
from PySide6.QtCore import Qt, Signal, Slot

from .bocados import (extract_fps_from_log,
                      img_tfms_inf)
from .custom_widgets.heatmap_widget import HeatmapWidget
from .custom_widgets.error_dialog import ErrorDialog

class HeatmapTab(QWidget):
    progress_bar_update = Signal(float)
    post_processing_signal = Signal(bool)

    def __init__(self):
        super().__init__()
        self.selected_dir = ""
        self.progress_bar_update[float].connect(self.update_progress_bar)
        self.post_processing_status = False
        self.post_processing_signal[bool].connect(self.update_post_processing_status)
        self.model = load_learner(Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model1.pkl")))
        self.stop_post_processing = False
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

        # Elegir modelo
        self.dropdown_button = QPushButton('Model 1', self)
        menu = QMenu(self)
        model1_action = menu.addAction('Model 1')
        model2_action = menu.addAction('Model 2')
        model3_action = menu.addAction('Model 3')
        model4_action = menu.addAction('Model 4')
        model5_action = menu.addAction('Model 5')
        model1_action.triggered.connect(self.load_model1)
        model2_action.triggered.connect(self.load_model2)
        model3_action.triggered.connect(self.load_model3)
        model4_action.triggered.connect(self.load_model4)
        model5_action.triggered.connect(self.load_model5)
        self.dropdown_button.setMenu(menu)

        post_process_button_layout.addWidget(self.post_process_dir_label)
        post_process_button_layout.addWidget(select_button)
        post_process_button_layout.addWidget(self.dropdown_button)
        post_process_button_layout.addWidget(post_process_button)
        stop_analysis_button = QPushButton("Stop post processing")
        post_process_button_layout.addWidget(stop_analysis_button)
        heatmap_layout.addLayout(post_process_button_layout)

        # Initially, hide
        self.setLayout(heatmap_layout)
        self.hide()

        # Conexiones con funciones
        select_button.clicked.connect(self.select_folder)
        post_process_button.clicked.connect(self.post_process_button_sign_emit)
        stop_analysis_button.clicked.connect(self.stop_analysis)

    def load_model1(self):
        self.model = load_learner(Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model1.pkl")))
        self.dropdown_button.setText('Model 1')

    def load_model2(self):
        self.model = load_learner(Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model2.pkl")))
        self.dropdown_button.setText('Model 2')

    def load_model3(self):
        self.model = load_learner(Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model3.pkl")))
        self.dropdown_button.setText('Model 3')

    def load_model4(self):
        self.model = load_learner(Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model4.pkl")))
        self.dropdown_button.setText('Model 4')

    def load_model5(self):
        self.model = load_learner(Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model5.pkl")))
        self.dropdown_button.setText('Model 5')

        
    def stop_analysis(self):
        if not self.post_processing_status:
            ErrorDialog("Not post processing anything!")
            return
        
        self.stop_post_processing = True
        ErrorDialog("Post process stopping soon...")
        

    @Slot(float)
    def update_progress_bar(self, progress):
        self.heatmap_progress_bar.setValue(progress)

    @Slot(bool)
    def update_post_processing_status(self, new_status : bool):
        self.post_processing_status = new_status

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

    def post_process_button_sign_emit(self):        
        # Start the task in a separate thread
        thread = Thread(target=self.post_process)
        thread.start()

    @Slot()
    def post_process(self):
        if self.post_processing_status:
            logger.error("Already post processing")
            return
        
        if not os.path.exists(self.selected_dir):
            ErrorDialog("Invalid directory for post processing")
            return
        
        if (os.path.exists(os.path.join(self.selected_dir, "post_process.log")) 
            and os.path.exists(os.path.join(self.selected_dir, "heatmap.jpg"))):
            saved_image = cv2.imread(os.path.join(self.selected_dir, "heatmap.jpg"), cv2.IMREAD_GRAYSCALE)
            grid_hand_count = saved_image.astype(np.int64)
            self.heatmap.plot(grid_hand_count)
            return
        
        log_file = os.path.join(self.selected_dir, "log.log")
        if not os.path.exists(log_file):
            ErrorDialog("Log file does not exist.")
            return
        
        masks_dir = os.path.join(self.selected_dir, "EgoHOS_Masks")
        if not os.path.exists(masks_dir):
            ErrorDialog("EgoHOS masks folder does not exist.")
            return

        if not os.path.exists(os.path.join(self.selected_dir, "FoodSeg_Masks/")):
            ErrorDialog("FoodSeg masks folder does not exist.")
            return


        # Primero revisar si se puede extraer los fps
        fps = extract_fps_from_log(log_file)
        if fps == None:
            ErrorDialog("Invalid fps count at the log file.")
            return
        msecs_frame = 2 * 1000 / fps    # · 2 porque las máscaras se guardan skipeando un frame (doblando el tiempo)

        self.post_processing_signal.emit(True)

        json_log = []

        result_logger = logger.bind(application="food_and_hands_post_process")
        result_logger.add(sink=os.path.join(self.selected_dir, "post_process.log"), rotation="10 MB")
        
        mask_files = os.listdir(masks_dir)
        mask_pngs = [filename for filename in mask_files if filename.endswith(".png")]
        mask_pngs.sort(key=lambda x: int(x.split(".")[0]))

        np_image, hand_mask = img_tfms_inf(Path(os.path.join(masks_dir, mask_pngs[0])))
        height = int(np_image.shape[0] / 10)
        width = int(np_image.shape[1] / 10)
        grid_hand_count = np.zeros((height, width), dtype=np.int64)

        superpixel_height = 10
        superpixel_width = 10
        
        percent_analyzed = 0
        self.heatmap_progress_bar.setValue(percent_analyzed)
        percent_increase_per_frame = 100 / len(mask_pngs)

        bocado_count = 0
        self.stop_post_processing = False
        for mask_filename in mask_pngs:
            if self.stop_post_processing:
                return
            
            frame_number = int(mask_filename.split(".")[0])
            percent_analyzed = percent_increase_per_frame * (frame_number + 1)
            self.progress_bar_update.emit(percent_analyzed)

            np_image, hand_mask = img_tfms_inf(Path(os.path.join(masks_dir, mask_filename)))
            predictions = self.model.predict(hand_mask)

            json_frame_log = {
                "frame_number" : frame_number,
                "mask_id" : mask_filename,
                "bocado" : False
            }

            if (predictions[0] == "bite"):
                bocado_count += 1
                result_logger.info(f"Bocado detectado en frame {frame_number}. Segundo {frame_number * msecs_frame / 1000}")
                json_frame_log["bocado"] = True

            json_log.append(json_frame_log)

            for i in range(grid_hand_count.shape[0]):
                for j in range(grid_hand_count.shape[1]):
                    # Get the region corresponding to the superpixel
                    superpixel_region = np_image[i*superpixel_height:(i+1)*superpixel_height, 
                                                    j*superpixel_width:(j+1)*superpixel_width]
                    
                    # Check if any value in the superpixel region is equal to 255
                    if np.any(superpixel_region == 255):
                        # If any 255 value is found, set the count to 1
                        grid_hand_count[i, j] += 1
        
        cv2.imwrite(os.path.join(self.selected_dir, "heatmap.jpg"), grid_hand_count)
        result_logger.info(f"Cuenta de frames detectados como bocados: {bocado_count}")
        self.heatmap.plot(grid_hand_count)

        with open(os.path.join(self.selected_dir, "post_process_log.json"), "w") as json_file:
            json.dump(json_log, json_file)

        self.post_processing_signal.emit(False)
        