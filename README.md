# Algoritmo de detección de anomalías en mamografías

Este proyecto persigue el objetivo de crear una herramienta para detectar las distintas anomalías que se puedan encontrar en una mamografía.

Para la detección de anomalías se utiliza la red YOLOv5, a la que ayudamos a un mejor entrenamiento gracias al procesado de imágenes. También se incluye el código para crear un servidor que permita subir imágenes y, con los mejores resultados de la etapa de entrenamiento, devolver las anomalías en las imágenes subidas. La base de datos que se ha utlizado para crear este proyecto ha sido la de CBIS-DDSM: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM

Este proyecto puede servir como base para generar una inteligencia artificial accesible y gratuita que ayude en la lucha contra el cancer de mama. A continuación, se detalla la estructura del repositorio.

* server: contiene todos los archivos que forman parte de la aplicación web que se aloja en un servidor
* train: contiene todos los ficheros necesarios para realizar el entrenamiento de los datos.
  * documents: aquí se encuentra tanto el archivo de configuración para YOLOv5 como los archivos de hiperparámetros usados durante las distintas pruebas realizadas en la etapa de entrenamiento.
* utils: contiene un fichero _constants.py_ con constantes que se utilizan en el proyecto.
