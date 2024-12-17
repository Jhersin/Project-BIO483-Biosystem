import cv2
import numpy as np

def recortar_centro(imagen, alto_deseado=320, ancho_deseado=320):
    """
    Recorta el centro de una imagen.

    Args:
        imagen (numpy.ndarray): La imagen de entrada con dimensiones (Canales, Alto, Ancho).
        alto_deseado (int): Alto deseado para la imagen recortada.
        ancho_deseado (int): Ancho deseado para la imagen recortada.

    Returns:
        numpy.ndarray: La imagen recortada con dimensiones (Canales, alto_deseado, ancho_deseado).
    """
    alto_original, ancho_original = imagen.shape[1], imagen.shape[2]
    inicio_alto = (alto_original - alto_deseado) // 2
    inicio_ancho = (ancho_original - ancho_deseado) // 2
    fin_alto = inicio_alto + alto_deseado
    fin_ancho = inicio_ancho + ancho_deseado
    imagen_recortada = imagen[:, inicio_alto:fin_alto, inicio_ancho:fin_ancho]
    return imagen_recortada