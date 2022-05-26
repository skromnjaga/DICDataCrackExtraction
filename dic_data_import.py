import fnmatch
import zipfile
from io import StringIO

import numpy as np


def import_xy_from_DaVis_txt(file):
    '''
    Import 2d x and y coordinates from .txt file imported from Davis
    '''
    # Read fist line of file
    first_line = file.readline()
    if isinstance(first_line, bytes):
        first_line = first_line.decode()
    text_data = first_line.split(' ')

    # Import coordinates data
    width = int(text_data[3])
    height = int(text_data[4])
    scale_x = float(text_data[6])
    offset_x = float(text_data[7])
    scale_y = float(text_data[10])
    offset_y = float(text_data[11])

    # Create mesh fo X and Y
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xx, yy = np.meshgrid(x, y)
    xx = xx * scale_x + offset_x
    yy = yy * scale_y + offset_y

    return xx, yy

def import_scalar_field_from_DaVis_txt(file, detect_bounding_box=True):
    '''
    Import 2d scalar field from .txt file imported from Davis
    '''
    file_data = file.read()
    if isinstance(file_data, bytes):
        file_data = file_data.decode()

    # Replace coma with dot
    file_data = file_data.replace(',', '.')

    # Load data in numpy array
    txt_data = np.loadtxt(StringIO(file_data), dtype=np.float32, delimiter='\t') #, converters = {0: lambda s: float(s.decode("UTF-8").replace(",", "."))})

    if detect_bounding_box:
        # Find bounding box for non zero data ROI
        # from https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
        rows = np.any(txt_data, axis=1)
        cols = np.any(txt_data, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

    if detect_bounding_box:
        return txt_data, (rmin, rmax, cmin, cmax)
    else:
        return txt_data, None

def import_data_from_davis(data_set_name, data_path, file_mask = 'B*.txt'):
    '''
    Import series of 2d scalar fields from Davis imported .txt files stored in zip archive
    and save imported data to .npz file if data_path folder
    '''
    imported_data = []

    file_name = data_path + '/' + data_set_name + '.zip'

    bounding_box_found = False

    # Open zip file with DaVis data
    with zipfile.ZipFile(file_name) as z:
        print(f'ZIP file { file_name } opened..')

        # Filter files to import with file_mask
        name_list = z.namelist()
        file_to_process = fnmatch.filter(name_list, file_mask)        

        # Detect X and Y coordinates in real dimensions
        with z.open(file_to_process[0], 'r') as file:
            xx, yy = import_xy_from_DaVis_txt(file)
             
        # Process each file
        for fn in file_to_process:
            with z.open(fn, 'r') as file:
                # file.readline() instead import_xy_from_DaVis_txt()???
                import_xy_from_DaVis_txt(file)
                data, bb = import_scalar_field_from_DaVis_txt(file)
                
                if bb is not None and not bounding_box_found:
                    bounding_box_found = True
                    bounding_box = bb
                    # Cut X and Y coordinates with detemined ROI
                    xx = xx[bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]
                    yy = yy[bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]

            if bounding_box_found:
                # Cut region of interest
                data = data[bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]

            imported_data.append(data)

            print(f'File { fn } imported..')

    # If any data exported, save it to file
    if len(imported_data) > 0:
        data_to_store = np.array(imported_data)
        file_name = data_path + '/' + data_set_name + '.npz'
        np.savez(file_name, x=xx, y=yy, data=data_to_store)
        print(f'Imported data saved to "{ file_name }" file')

    return xx, yy, data


if __name__ == '__main__':

    DATA_PATH = './data'

    DATA_SETS = ('R_6_28_12_2021', 'O16_12_08_2021', 'H_9_2_17_12_2021', 'R_Nemo_10_03_2022')

    # Index of data set to import
    DATA_SET_INDEX = 2

    xx, yy, data = import_data_from_davis(DATA_SETS[DATA_SET_INDEX], DATA_PATH)