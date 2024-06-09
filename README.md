# food_and_hands
TFM sobre segmentación de comida y manos en vídeos ego-centric


## Instalación:
### Inicializar submódulos en diferentes entornos virtuales. FoodSeg y EgoHOS.
- Para entorno virtual de FoodSeg seguir guía de instalación [get_started.md](external_libs/FoodSeg103/docs/get_started.md#installation)
- Para entorno virtual de EgoHOS seguir guía de instalación en EgoHOS [README.md](external_libs/EgoHOS/README.md)
Nota: Añadir los siguientes directorios en la raíz del submódulo EgoHOS:
```md
server_imgs/
├── images/
├── pred_cb/
├── pred_obj1/
└── pred_twohands/
```
### Crear un nuevo entorno virtual "food_n_hands" para el código que usa los servidores para analizar los vídeos o imágenes + GUI.
```sh
conda create -n food_n_hands python==3.12 -y
```
- Instalar [Pytorch](https://pytorch.org/).
- Instalar [FastAI](https://docs.fast.ai/).
- Instalar [food_n_hands_req.txt](food_n_hands_req.txt)
```sh
python -m pip install -r food_n_hands_req.txt
```

### Incluir modelos en carpeta post_processing dentro de una carpeta */models*.
Se incluye este [enlace](https://drive.google.com/drive/folders/15c9Lgvp7lNaPlAHmkhIQK32wpc_uIvZX?usp=sharing) para la descarga de los modelos.

## Cómo usar:
### Terminal 1:
```sh
conda activate FoodSeg
cd external_libs/FoodSeg103
python FoodSeg_Server.py
```

### Terminal 2:
```sh
conda activate EgoHOS
cd external_libs/EgoHOS/mmsegmentation
python EgoHOS_Server.py
```

### Terminal 3:
```sh
conda activate food_n_hands
python gui.py
```

Abrirá la interfaz gráfica para utilizar el proyecto.


## Citations:
Thanks to these 2 awesome original repositories:

### [FoodSeg103](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1)
```
@inproceedings{wu2021foodseg,
	title={A Large-Scale Benchmark for Food Image Segmentation},
	author={Wu, Xiongwei and Fu, Xin and Liu, Ying and Lim, Ee-Peng and Hoi, Steven CH and Sun, Qianru},
	booktitle={Proceedings of ACM international conference on Multimedia},
	year={2021}
}
```

### [EgoHOS](https://github.com/owenzlz/EgoHOS)
```
@inproceedings{zhang2022fine,
  title={Fine-Grained Egocentric Hand-Object Segmentation: Dataset, Model, and Applications},
  author={Zhang, Lingzhi and Zhou, Shenghao and Stent, Simon and Shi, Jianbo},
  booktitle={European Conference on Computer Vision},
  pages={127--145},
  year={2022},
  organization={Springer}
}
```

