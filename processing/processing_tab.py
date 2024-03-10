import cv2
import os
import numpy as np
from loguru import logger
from threading import Thread

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, QFileDialog, QTableWidget, QTableWidgetItem
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtCore import Qt
from .error_dialog import ErrorDialog


from IPC.FoodSegUtils import analyze_FoodSeg_mask, colorize_FoodSeg_Mask
from IPC.EgoHOS_Utils import analyze_egoHOS_mask, colorize_egoHOS_mask
from IPC.client_protocols import FoodNhands_Client
from IPC.utils import create_directory, ThreadWithReturnValue
from IPC.defs import foodseg_categories, hands_categories


class ProcessingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_video = ""
        self.selected_output_dir = ""
        self.analyzing_video = False
        self.stop_analyzing_video = False
        self.initUI()

    def initUI(self):
        processing_layout = QVBoxLayout()

        # Layout de mostrado de imágenes + leyenda
        display_layout = QHBoxLayout()
        display_layout_proportions = [3, 1]
        self.video_images_label = QLabel()
        self.video_images_label.setAlignment(Qt.AlignCenter)
        self.video_images_label.setMinimumSize(1,1)
        self.video_images_label.setScaledContents(True)
        display_layout.addWidget(self.video_images_label, display_layout_proportions[0])

        # Tabla con la leyenda de las imágenes
        table_proportions = [20,80]
        self.table = QTableWidget()
        self.table.setRowCount(1)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Color", "Tag"])
        # self.table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        
        # Calculate column widths as percentages of the table width
        proportions_width = sum(table_proportions)
        table_width = self.table.size().width() / int(sum(display_layout_proportions))

        # Set section sizes for columns based on the calculated percentages
        for i, relative_width in enumerate(table_proportions):
            section_size = int((relative_width / proportions_width) * table_width)
            self.table.horizontalHeader().resizeSection(i, section_size)

        display_layout.addWidget(self.table, display_layout_proportions[1])
        processing_layout.addLayout(display_layout) 

        # Barra de progreso del análisis
        self.analysis_progress_bar = QProgressBar()
        self.analysis_progress_bar.setAlignment(Qt.AlignCenter)
        processing_layout.addWidget(self.analysis_progress_bar)

        # Layout para botones de selección de vídeo + botón de análisis
        processing_toolbar_layout = QHBoxLayout()
        self.video_path_label = QLabel("No video selected")
        processing_toolbar_layout.addWidget(self.video_path_label)
        choose_video_button = QPushButton("Select input video")
        processing_toolbar_layout.addWidget(choose_video_button)
        analyze_button = QPushButton("Analyze!")
        processing_toolbar_layout.addWidget(analyze_button)
        processing_layout.addLayout(processing_toolbar_layout)

        # Layout para botón de selección de directorio salida + stop análisis
        output_toolbar_layout = QHBoxLayout()
        self.output_dir_label = QLabel("No output dir selected")
        output_toolbar_layout.addWidget(self.output_dir_label)
        choose_outdir_button = QPushButton("Select output directory")
        output_toolbar_layout.addWidget(choose_outdir_button)
        stop_analysis_button = QPushButton("Stop analysis")
        output_toolbar_layout.addWidget(stop_analysis_button)
        processing_layout.addLayout(output_toolbar_layout)

        # Conexión de señales y slots
        choose_video_button.clicked.connect(self.select_video)
        choose_outdir_button.clicked.connect(self.select_folder)
        analyze_button.clicked.connect(self.analyze_video)
        stop_analysis_button.clicked.connect(self.stop_analysis)

        # Esconder todas las tabs al principio.
        self.setLayout(processing_layout)
        self.hide()


    def select_video(self):
        file_filter = 'Video Files (*.mp4 *.avi *.mkv)'
        response, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a video file',
            dir=os.getcwd(),
            filter=file_filter
        )
        if response:
            self.video_path_label.setText(response)
            self.selected_video = response
        else:
            self.video_path_label.setText("No video selected")
            self.selected_video = ""


    def select_folder(self):
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if file:
            self.output_dir_label.setText(file)
            self.selected_output_dir = file
        else:
            self.output_dir_label.setText("No output dir selected")
            self.selected_output_dir = ""


    def stop_analysis(self):
        if not self.analyzing_video:
            ErrorDialog("Not analyzing anything!")
            return
        
        self.stop_analyzing_video = True
        ErrorDialog("Analysis stoppnig soon...")


    def analyze_video(self):
        if self.analyzing_video:
            ErrorDialog("Already analyzing other video!")
            return
        
        video_path = self.selected_video
        out_dir = self.selected_output_dir

        # Start the task in a separate thread
        thread = Thread(target=self.food_n_hands_gui, args=("127.0.0.1", 33334, "127.0.0.1", 33333, video_path, False, True, out_dir,))
        thread.start()


    def food_n_hands_gui(self, FoodSegHOST : str, FoodSegPORT : int, EgoHOS_HOST : str, EgoHOS_PORT : int, video_path : str, left : bool, right : bool, out_dir : str):
        cap = cv2.VideoCapture(video_path)
        encoding = "BGR"
        if not cap.isOpened():
            ErrorDialog("Error: Unable to open video file.")
            return

        frame_number = 1
        ret, frame = cap.read()
        if not ret:
            ErrorDialog("Error: Unable to read frames from video.")
            return
        
        if not os.path.exists(out_dir):
            ErrorDialog("Error: Carpeta seleccionada no existe")
            return

        self.analyzing_video = True
        output_dir = self.selected_output_dir
        
        egohos_masks_dir = create_directory(output_dir, "EgoHOS_Masks")
        foodseg_masks_dir = create_directory(output_dir, "FoodSeg_Masks")
        original_imgs_dir = create_directory(output_dir, "Original_Images")

        result_logger = logger.bind(application="food_and_hands")
        result_logger.add(sink=os.path.join(output_dir, "log.log"), rotation="10 MB")

        frame_height, frame_width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(os.path.join(output_dir, "output_video.avi"), fourcc, fps, (frame_width, frame_height))
        result_logger.info(f"\nVideo path: {video_path}. Fps del vídeo: {fps}. Shape: ({frame_width}, {frame_height})")

        while not self.stop_analyzing_video:
            frame_number += 1
            ret, frame = cap.read()
            if not ret:
                break

            if not frame_number % 2 == 0:
                continue

            cv2.imwrite(os.path.join(original_imgs_dir, f"{frame_number/2}.png"), frame)
            FoodSegHOST = "127.0.0.1"
            FoodSegPORT = 33334
            EgoHOS_HOST = "127.0.0.1"
            EgoHOS_PORT = 33333

            FoodSeg_Thread = ThreadWithReturnValue(target=FoodNhands_Client, args=(FoodSegHOST, FoodSegPORT, frame, encoding,))
            EgoHOS_Thread = ThreadWithReturnValue(target=FoodNhands_Client, args=(EgoHOS_HOST, EgoHOS_PORT, frame, encoding,))
            EgoHOS_Thread.start()
            FoodSeg_Thread.start()

            FoodSeg_Mask = FoodSeg_Thread.join()
            EgoHOS_Mask = EgoHOS_Thread.join()

            if FoodSeg_Mask is None or EgoHOS_Mask is None:
                logger.error("Error at masks retrieval.")
                ErrorDialog("Error at masks retrieval.")
                return
            
            cv2.imwrite(os.path.join(foodseg_masks_dir, f"{frame_number/2}.png"), FoodSeg_Mask)
            cv2.imwrite(os.path.join(egohos_masks_dir, f"{frame_number/2}.png"), EgoHOS_Mask)

            try:
                FoodSeg_Mask_color = colorize_FoodSeg_Mask(frame, FoodSeg_Mask)
                EgoHOS_Mask_color = colorize_egoHOS_mask(frame, EgoHOS_Mask)

            except Exception as e:
                logger.error(f"Error at colirizing masks: {str(e)}")
                ErrorDialog(f"Error at colirizing masks: {str(e)}")
                return
            
            frame_log = f"\nIn frame {frame_number} we find:"
            ingredients_log, food_found = analyze_FoodSeg_mask(FoodSeg_Mask)
            egoHOS_log, egoHOS_tags_found = analyze_egoHOS_mask(EgoHOS_Mask, left_hand=left, right_hand=right)

            frame_log += ingredients_log + egoHOS_log
            result_logger.info(frame_log)

            # Señalizar los tags de esta imagen en la tabla
            self.show_tags_in_table(food_found, egoHOS_tags_found)

            masked_image_comb = frame.copy()
            masked_image_comb[FoodSeg_Mask_color > 0] = FoodSeg_Mask_color[FoodSeg_Mask_color > 0]
            masked_image_comb[EgoHOS_Mask_color > 0] = EgoHOS_Mask_color[EgoHOS_Mask_color > 0]

            # Convert the OpenCV frame to a QImage
            height, width, _ = masked_image_comb.shape
            bytes_per_line = 3 * width
            q_image = QImage(masked_image_comb.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(width, height, aspectMode=Qt.AspectRatioMode.KeepAspectRatio, mode=Qt.TransformationMode.SmoothTransformation)

            # Create a QPixmap from the QImage
            width = self.video_images_label.width()
            height = self.video_images_label.height()
            self.video_images_label.setPixmap(scaled_pixmap)

            try:
                out.write(masked_image_comb)
            
            except Exception as e:
                result_logger.error(f'{str(e)}')
                ErrorDialog(f'{str(e)}')
                return

        # Release the video capture object
        cap.release()
        self.stop_analyzing_video = False
        self.analyzing_video = False


    def show_tags_in_table(self, food_tags_found, egohos_tags_found):
        # Clear all existing rows
        self.table.clearContents()
        self.table.setRowCount(len(food_tags_found) + len(egohos_tags_found) - 2)   # - 2 porque en ambos tags tenemos el background.
        self.table.setColumnCount(2)

        # Set default values for the new row
        i = -1
        for food_tag in food_tags_found:
            if food_tag == 0:
                continue
            i += 1
            food_tag_name = QTableWidgetItem(foodseg_categories[food_tag]["tag"])
            food_tag_color = QTableWidgetItem()
            food_tag_color.setBackground(QColor.fromRgb(foodseg_categories[food_tag]["color"][2],
                                                   foodseg_categories[food_tag]["color"][1],
                                                   foodseg_categories[food_tag]["color"][0]))
            self.table.setItem(i, 0, food_tag_color)
            self.table.setItem(i, 1, food_tag_name)
        
        for ego_tag in egohos_tags_found:
            if ego_tag == 0:
                continue
            i += 1
            egohos_tag_name = QTableWidgetItem(hands_categories[ego_tag]["tag"])
            egohos_tag_color = QTableWidgetItem()
            egohos_tag_color.setBackground(QColor.fromRgb(hands_categories[ego_tag]["color"][2],
                                                   hands_categories[ego_tag]["color"][1],
                                                   hands_categories[ego_tag]["color"][0]))
            self.table.setItem(i, 0, egohos_tag_color)
            self.table.setItem(i, 1, egohos_tag_name)
        

        table_proportions = [20,80]
        proportions_width = sum(table_proportions)
        table_width = int(self.table.size().width())

        # Set section sizes for columns based on the calculated percentages
        for i, relative_width in enumerate(table_proportions):
            section_size = int((relative_width / proportions_width) * table_width)
            self.table.horizontalHeader().resizeSection(i, section_size)

        self.table.show()
