# TFM: Reconocimiento de lenguaje aplicado como *Tiny ML* en una placa Coral Dev de Google

Proyecto para el TFM del máster en Ciencia de Datos de la Universidad de Valencia (UV), sobre el reconocimiento de palabras de activación utilizando redes neuronales e implementando los modelos en una placa Coral Dev de Google, dispositivo para Tiny ML.

## Estructura

En la carpeta "Code" se incluye todo lo que se ha ido haciendo durante el trabajo. Explico a continuación de manera muy resumida qué es lo que hay en cada archivo y carpeta:

* Coral Dev Board: archivos para poder llevar a cabo las ejecuciones en la placa. Están los modelos que se han utilizado, los archivos con las etiquetas, los requirements a instalar en la placa, y los scripts a ejecutar en la placa para llevar a cabo inferencia. La carpeta data está vacía: habría que añadir algunos audios de ejemplo en inglés y/o en español para llevarlos a la placa.

* Data: carpeta donde se deben introducir los datos y sobre la que se irán generando los datos modificados. En el repositorio la carpeta está vacía, solo tiene la estructura de carpetas que está codificada en los notebooks. Se debe rellenar convenientemente si se quiere ejecutar los notebooks.

* Imagenes: imágenes de los modelos que se han generado. De cada modelo hay una imagen obtenida con el software [Netron](https://netron.app/)

* models: modelos y archivos generados en cada una de las partes del trabajo. En la carpeta "Otros" está un notebook a ejecutar en Colab que compila un modelo TFLite para la TPU, y un modelo preentrenado en formato TFLite.

* notebooks:

    * calculos_CDB: estimaciones de los tiempos de ejecución de los diferentes modelos en la placa
    * modelos_simples: creación, entrenamiento y adaptación a la placa de los diferentes modelos generados
    * TL_model_maker_speech_recognition_ENG_COLAB: transfer learning sobre el conjunto de datos en inglés. Notebook principalmente desarrollado por TensorFlow
    * TL_model_maker_speech_recognition_ESP_COLAB: transfer learning sobre el conjunto de datos en español. Notebook principalmente desarrollado por TensorFlow
    * tratamiento_datos_speech_commands: tratamiento, análisis y transformación del conjunto de datos *Speech commands*
    * tratamiento_datos_transfer_learning: estandarización y análisis del conjunto de datos recopilado para llevar a cabo transfer learning

## Referencias

A las referencias que se nombran en la bibliografía de la memoria hay que añadir unas series de vídeos del canal de YouTube [The Sound of AI](https://www.youtube.com/channel/UCZPFjMe1uRSirmSpznqvJfQ) que me han ayudado mucho en la realización del trabajo:

* [Audio Signal Processing for ML](https://www.youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)
* [Audio Data Augmentation](https://www.youtube.com/playlist?list=PL-wATfeyAMNoR4aqS-Fv0GRmS6bx5RtTW)
* [Deep Learning (Audio) Application: From Design to Deployment](https://www.youtube.com/playlist?list=PL-wATfeyAMNpCRQkKgtOZU_ykXc63oyzp)
* [Deep Learning (for Audio) with Python](https://www.youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf)
