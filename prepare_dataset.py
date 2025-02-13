import cv2
import fitz
from PIL import Image 
import numpy as np
from augraphy import *
import os
import tqdm


PDF_DIR = 'pdfs'
INPUTS_DIR = 'inputs'
OUTPUTS_DIR = 'outputs'

ink_phase = [
    OneOf(
        [
            BleedThrough(),
            ColorShift(),
            DepthSimulatedBlur(),
            DoubleExposure(),
            Hollow(),
            InkBleed(),
            InkMottling(),
            Letterpress(),
            LowInkRandomLines()
        ]
    )
]

paper_phase = [
    OneOf(
        [
            BadPhotoCopy(),
            BindingsAndFasteners(),
            Brightness(),
            DelaunayTessellation(),
            DirtyDrum(),
            DirtyRollers(),
            Moire(),
            PatternGenerator(),
        ]
    )
]

post_phase = [
    OneOf(
        [
            Folding(),
            GlitchEffect(),
            WaterMark(),
            ShadowCast(),
            ReflectedLight(),
            InkShifter()
        ]
    )
]

pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

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
        augmented_iamge = pipeline(image)
        cv2.imwrite(os.path.join(INPUTS_DIR, str(new_file_index) + '.png'), image)
        cv2.imwrite(os.path.join(OUTPUTS_DIR, str(new_file_index) + '.png'), augmented_iamge)
