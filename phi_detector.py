#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from chris_plugin import chris_plugin, PathMapper
import pydicom
from PIL import Image
import pytesseract
import numpy as np
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
import re

phi_patterns = {
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "Phone": r"\b(?:\+?1\s*[-.]?)?\(?\d{3}\)?[-.]?\s?\d{3}[-.]?\d{4}\b",
        "Email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
        "Date (MM/DD/YYYY)": r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/\d{4}\b",
        "MRN": r"\b\d{6,7}\b",
        "ZIP Code": r"\b\d{5}(?:-\d{4})?\b",
        "IP Address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

__version__ = '1.0.4'

DISPLAY_TITLE = r"""
       _              _     _      _      _            _             
      | |            | |   (_)    | |    | |          | |            
 _ __ | |______ _ __ | |__  _   __| | ___| |_ ___  ___| |_ ___  _ __ 
| '_ \| |______| '_ \| '_ \| | / _` |/ _ \ __/ _ \/ __| __/ _ \| '__|
| |_) | |      | |_) | | | | || (_| |  __/ ||  __/ (__| || (_) | |   
| .__/|_|      | .__/|_| |_|_| \__,_|\___|\__\___|\___|\__\___/|_|   
| |            | |         ______                                    
|_|            |_|        |______|                                   
"""


parser = ArgumentParser(description='A ChRIS plugin to detect text in a DICOM file',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--pattern', default='**/*.txt', type=str,
                    help='input file filter glob')
parser.add_argument('-V', '--version', action='version',
                    version=f'%(prog)s {__version__}')
parser.add_argument('-f', '--fileFilter', default='dcm', type=str,
                    help='input file filter glob')
parser.add_argument('-t', '--outputType', default='dcm', type=str,
                    help='output file type(extension only)')


# The main function of this *ChRIS* plugin is denoted by this ``@chris_plugin`` "decorator."
# Some metadata about the plugin is specified here. There is more metadata specified in setup.py.
#
# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title='My ChRIS Plugin',
    category='',                 # ref. https://chrisstore.co/plugins
    min_memory_limit='1000Mi',    # supported units: Mi, Gi
    min_cpu_limit='2000m',       # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0              # set min_gpu_limit=1 to enable GPU
)
def main(options: Namespace, inputdir: Path, outputdir: Path):
    """
    *ChRIS* plugins usually have two positional arguments: an **input directory** containing
    input files and an **output directory** where to write output files. Command-line arguments
    are passed to this main method implicitly when ``main()`` is called below without parameters.

    :param options: non-positional arguments parsed by the parser given to @chris_plugin
    :param inputdir: directory containing (read-only) input files
    :param outputdir: directory where to write output files
    """

    print(DISPLAY_TITLE)

    # Typically it's easier to think of programs as operating on individual files
    # rather than directories. The helper functions provided by a ``PathMapper``
    # object make it easy to discover input files and write to output files inside
    # the given paths.
    #
    # Refer to the documentation for more options, examples, and advanced uses e.g.
    # adding a progress bar and parallelism.
    mapper = PathMapper.file_mapper(inputdir, outputdir, glob=f"**/*.{options.fileFilter}",fail_if_empty=False)
    for input_file, output_file in mapper:
        # Read each input file from the input directory that matches the input filter specified
        dcm_img = read_input_dicom(input_file)

        # check if a valid image file is returned
        if dcm_img is None:
            continue

        # Save the file in o/p directory in the specified o/p type\
        if options.outputType == "dcm":
            save_dicom(dcm_img, output_file)
        else:
            save_as_image(dcm_img, output_file, options.outputType)
        print("\n\n")
        
def read_input_dicom(input_file_path):
    """
    1) Read an input dicom file
    """
    ds = None
    try:
        print(f"Reading input file : {input_file_path.name}")
        ds = pydicom.dcmread(str(input_file_path))
        if 'PixelData' not in ds:
            print("No pixel data in this DICOM.")
            return None
    except Exception as ex:
        print(f"unable to read dicom file: {ex} \n")
        return None
    image = dicom_to_image(ds)

    # Perform OCR
    extracted_text = pytesseract.image_to_string(image)

    print(f"Extracted Text: {extracted_text}")
    if detect_phi_nltk(extracted_text):
        return ds


    return None

def dicom_to_image(ds):
    pixel_array = ds.pixel_array  # This is usually a NumPy array

    # Check shape
    print("DICOM pixel_array shape:", pixel_array.shape)

    # Choose the middle slice if it's 3D
    if pixel_array.ndim == 3:
        pixel_array = pixel_array[pixel_array.shape[0] // 2]

    # Normalize and convert to uint8 if necessary
    if pixel_array.dtype != np.uint8:
        pixel_array = (255 * (pixel_array - np.min(pixel_array)) / np.ptp(pixel_array)).astype(np.uint8)

    # Convert to PIL Image
    image = Image.fromarray(pixel_array)
    return image

def detect_phi_nltk(text):
    entities = []
    tokens = word_tokenize(text)
    for token in tokens:
        for phi_type, pattern in phi_patterns.items():
            if re.match(pattern, token):
                entities.append((token, phi_type))
    pos_tags = pos_tag(tokens)
    ner_tree = ne_chunk(pos_tags)
    for chunk in ner_tree:
        if hasattr(chunk, 'label'):
            entity = " ".join(c[0] for c in chunk)
            entity_type = chunk.label()
            entities.append((entity, entity_type))
    target_entities = ['Date (MM/DD/YYYY)', 'MRN', 'PERSON']
    matching_tuples = [
        tpl for tpl in entities
        if any(isinstance(item, str) and item in target_entities for item in tpl)
    ]
    if matching_tuples:
        print("Possible PHI detected.")
        print(matching_tuples)
        return True
    return False

def save_dicom(dicom_file, output_path):
    """
    Save a dicom file to an output path
    """
    print(f"Saving dicom file: {output_path.name}")
    dicom_file.save_as(str(output_path))

def save_as_image(dcm_file, output_file_path, file_ext):
    """
    Save the pixel array of a dicom file as an image file
    """
    pixel_array_numpy = dcm_file.pixel_array
    output_file_path = str(output_file_path).replace('dcm', file_ext)
    print(f"Saving output file as {output_file_path}")
    print(f"Photometric Interpretation is {dcm_file.PhotometricInterpretation}")

    # Prevents color inversion happening while saving as images
    if 'YBR' in dcm_file.PhotometricInterpretation:
        print(f"Explicitly converting color space to RGB")
        pixel_array_numpy = convert_color_space(pixel_array_numpy, "YBR_FULL", "RGB")

    cv2.imwrite(output_file_path,cv2.cvtColor(pixel_array_numpy,cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
