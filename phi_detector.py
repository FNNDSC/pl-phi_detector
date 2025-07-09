#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from chris_plugin import chris_plugin, PathMapper
import pydicom
from PIL import Image
import pytesseract
import numpy as np
import spacy

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
import re



# Load the small English model
nlp = spacy.load("en_core_web_sm")

phi_patterns = {
    "NAME": r"([^,]+),\s*(.+)" ,#r"[A-Z][a-z]+(?: [A-Z][a-z]+)*",  # Matches capitalized names
    "DATE": r"\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}", # Matches various date formats
    "MEDICAL_RECORD_NUMBER": r"\bMRN\d{7}\b", # Matches MRNs with a specific pattern
    # Add more patterns for other PHI categories as needed
}

__version__ = '1.0.0'

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


parser = ArgumentParser(description='!!!CHANGE ME!!! An example ChRIS plugin which '
                                    'counts the number of occurrences of a given '
                                    'word in text files.',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--pattern', default='**/*.txt', type=str,
                    help='input file filter glob')
parser.add_argument('-V', '--version', action='version',
                    version=f'%(prog)s {__version__}')
parser.add_argument('-f', '--fileFilter', default='dcm', type=str,
                    help='input file filter glob')


# The main function of this *ChRIS* plugin is denoted by this ``@chris_plugin`` "decorator."
# Some metadata about the plugin is specified here. There is more metadata specified in setup.py.
#
# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title='My ChRIS Plugin',
    category='',                 # ref. https://chrisstore.co/plugins
    min_memory_limit='100Mi',    # supported units: Mi, Gi
    min_cpu_limit='1000m',       # millicores, e.g. "1000m" = 1 CPU core
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
        
def read_input_dicom(input_file_path):
    """
    1) Read an input dicom file
    """
    ds = None
    try:
        print(f"Reading input file : {input_file_path.name}")
        ds = pydicom.dcmread(str(input_file_path))
    except Exception as ex:
        print(f"unable to read dicom file: {ex} \n")
        return None
    pixel_array = ds.pixel_array

    # Convert pixel array to an image format suitable for OCR
    # (e.g., scale and convert to 8-bit grayscale if necessary)
    image_scaled = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
    image_8bit = Image.fromarray(image_scaled.astype(np.uint8))

    # Perform OCR
    extracted_text = pytesseract.image_to_string(image_8bit)

    for text in extracted_text.splitlines():
        print(f"Extracted Text: {text}")
        detect_phi_spacy(text)
        #detect_phi_nltk(text)

    return ds

def detect_phi_spacy(text):
    doc = nlp(text.replace(",", ""))

    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")

def detect_phi_nltk(text):
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('maxent_ne_chunker_tab')

    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('averaged_perceptron_tagger')
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    ner_tree = ne_chunk(pos_tags)
    entities = []
    for chunk in ner_tree:
        if hasattr(chunk, 'label'):
            entity = " ".join(c[0] for c in chunk)
            entity_type = chunk.label()
            entities.append((entity, entity_type))
        else:
            # Check for PHI patterns in the remaining tokens
            for token in chunk:
                for phi_type, pattern in phi_patterns.items():
                    if re.match(pattern, token[0]):
                        entities.append((token[0], phi_type))
    print(entities)



if __name__ == '__main__':
    main()
