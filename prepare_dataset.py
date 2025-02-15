import cv2
import fitz
from PIL import Image 
import numpy as np
from augraphy import *
import os
from tqdm import tqdm 
import random

PDF_DIR = 'pdfs'
INPUTS_DIR = 'inputs'
OUTPUTS_DIR = 'outputs'

paperfactory = PaperFactory(texture_path="",
                            generate_texture=1,
                            generate_texture_background_type="random",
                            generate_texture_edge_type="random",
                            texture_enable_color=0,
                            blend_texture=1,
                            blend_generate_texture = 0,
                            blend_texture_path='',
                            blend_texture_background_type="random",
                            blend_texture_edge_type="random",
                            blend_method="ink_to_paper")

pipeline = AugraphyPipeline(ink_phase=[], paper_phase=[paperfactory], post_phase=[])

for file in tqdm(os.listdir(os.path.join(PDF_DIR))):
    file_index = int(file.split('.')[0])
    file_image = fitz.open(os.path.join(PDF_DIR, file))
    #Get 300dpi Image matrix of PDF
    pix = file_image[0].get_pixmap(matrix = fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    image = image.convert('L')
    image = np.asarray(image)
    #Downloaded PDFs mot in appropriate A4 format, so we cut
    image = image[:3508,202:]

    for i in range(10):
        new_file_index = file_index * 10 + i
        noise = np.random.normal(0, 30, image.shape).astype(np.uint8)
        noised = cv2.add(image, noise)
        blur_ksize = random.choice([5, 7, 9])
        blurred = cv2.GaussianBlur(noised, (blur_ksize, blur_ksize), 0)
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, random.randint(15, 20))
        augmented_iamge = pipeline(thresholded)
        cv2.imwrite(os.path.join(OUTPUTS_DIR, str(new_file_index) + '.png'), image)
        cv2.imwrite(os.path.join(INPUTS_DIR, str(new_file_index) + '.png'), augmented_iamge)
