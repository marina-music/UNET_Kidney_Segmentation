import numpy as np

selected_class=['background', 'leftkidney', 'rightkidney']
selected_class_indice = [0,1,2]
selected_class_rgb=[[0, 0, 0], [0,255, 0], [0, 0, 255]]

def colour_code_segmentation(image):
    colour_code=np.array(selected_class_rgb)
    x=colour_code[image.astype(int)]
    return x
