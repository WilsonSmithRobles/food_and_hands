# food_and_hands
TFM sobre segmentación de comida y manos en vídeos ego-centric


## Instalación:
1. Inicializar submódulos en diferentes entornos virtuales. FoodSeg, EgoHOS y food_n_hands para el código que usa los servidores para analizar los vídeos o imágenes.
2. Instalar los submódulos en sus respectivos entornos virtuales siguiendo sus guías de instalación.


## Cómo usar:
```sh
conda activate FoodSeg
cd external_libs/FoodSeg103
python FoodSeg_Server.py >/dev/null 2>&1 &

conda activate EgoHOS
cd ../external_libs/EgoHOS/mmsegmentation
python EgoHOS_Server.py >/dev/null 2>&1 &
# Lanzamos servidores en background.

conda activate food_n_hands
cd ../../IPC
python food_n_hands.py
```