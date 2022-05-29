# Algoritmo de detección de anomalías en mamografías

Este proyecto persigue el objetivo de crear una herramienta para detectar las distintas anomalías que se puedan encontrar en una mamografía.

Para la detección de anomalías se utiliza la red YOLOv5, a la que ayudamos a un mejor entrenamiento gracias al procesado de imágenes. También se incluye el código para crear un servidor que permita subir imágenes y, con los mejores resultados de la etapa de entrenamiento, devolver las anomalías en las imágenes subidas. La base de datos que se ha utlizado para crear este proyecto ha sido la de [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM).

Este proyecto puede servir como base para generar una inteligencia artificial accesible y gratuita que ayude en la lucha contra el cancer de mama. A continuación, se detalla la estructura del repositorio.

* <code>server</code>: contiene todos los archivos que forman parte de la aplicación web que se aloja en un servidor
  * <code>aux_functions</code>: funciones auxiliares que se utilizan en varias partes del servidor.
  * <code>detect</code>: proceso que realiza la detección de anomalías en las imágenes que se suben al servidor.
  * <code>static</code>: contiene librerías de CSS y JavaScript que se utilizan en la aplicación.
  * <code>templates</code>: front-end de la aplicación web.
  * <code>weights</code>: contiene los pesos
  * <code>app.py</code>: back-end de la aplicación web.
  * <code>Dockerfile</code>: archivo de docker que se usa para desplegar toda la aplicación en el servidor.
  * <code>process_image.py</code>: funciones necesarias para el preprocesado de las imágenes que se suben al servidor.
* <code>train</code>: contiene todos los ficheros necesarios para realizar el entrenamiento de los datos.
  * <code>documents</code>: aquí se encuentra tanto el archivo de configuración para YOLOv5 como los archivos de hiperparámetros usados durante las distintas pruebas realizadas en la etapa de entrenamiento.
  * <code>image_preprocesing</code>: se encarga de realizar todas las transformaciones necesarias a las imágenes en formato PNG y dejarlas listas para el entrenamiento.
  * <code>preprocesing_cbid-ddsm</code>: recoge las imágenes descargadas en formato DICOM de web de CBIS-DDSM y las convierte a PNG para sus posteriores transformaciones.
  * <code>Train.ipynb</code>: notebook encargado de realizar el entrenamiento del modelo YOLOv5.
* <code>utils</code>: contiene un fichero _constants.py_ con constantes que se utilizan en el proyecto.
