from os import walk

import numpy as np

from keras import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.utils import to_categorical

from scripts.utils import readCsv, readMhd, getImgWorldTransfMats, convertToImgCoord

import traceback

# Get the mhd images
data = [file for file in list(walk('data'))[0][2] if file.endswith('.mhd')]
data = sorted(data)

# Read nodules csv
lines = readCsv('trainset_csv/trainNodules_gt.csv')
header = lines[0]
nodules = lines[1:]

# ML variables
X = []
Y = []

LNDb = {}

for filename in data:
    # Extract ID
    LNDbID = int(filename.lstrip('LNDb-').rstrip('.mhd').lstrip('0'))

    # Get nodule findings associated with ID
    LNDbNodules = [nodule for nodule in nodules if int(nodule[header.index('LNDbID')]) == LNDbID]

    # Read image
    [scan, spacing, origin, transfmat] = readMhd(f'data/{filename}')

    # Iterate through nodule findings
    for nodule in LNDbNodules:
        FindingID = int(nodule[header.index('FindingID')])
        RadID = nodule[header.index('RadID')]
        isNodule = bool(int(nodule[header.index('Nodule')]))
        Texture = float(nodule[header.index('Text')])

        # Get world coordinates
        x = float(nodule[header.index('x')])
        y = float(nodule[header.index('y')])
        z = float(nodule[header.index('z')])

        # Convert world coordinates to image coordinates
        ctr = np.array([x, y, z])
        transfmat_toimg, transfmat_toworld = getImgWorldTransfMats(spacing, transfmat)
        ctr = convertToImgCoord(ctr, origin, transfmat_toimg)

        # Determine nodule class
        if isNodule:
            if 0 < Texture < 2.3:
                Class = 1  # 'Ground Glass Opacities (GGO)'
            elif 2.3 <= Texture < 3.6:
                Class = 2  # 'Part Solid'
            elif 3.6 <= Texture:
                Class = 3  # 'Solid'
        else:
            Class = 0  # 'Not a Nodule'

        print(f"ID {LNDbID} - Finding {FindingID} - Class {Class} - Radiologists {RadID} - xyz {ctr}")

        # Read image
        img = np.array(scan[int(ctr[2])]).astype('float32')
        img = np.resize(img, (img.shape[0], img.shape[1], 1))
        if img.shape != (512, 512, 1):
            continue

        X.append(img)
        Y.append(Class)

# Split data into train and test sets
X = np.asarray(X)
Y = np.asarray(Y)
X_train, X_test = X[:218], X[218:]
Y_train, Y_test = Y[:218], Y[218:]

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Menu
while(1):

    print('Data Loading finished (press r + ENTER to run CNN) (press e + ENTER to exit)')

    k = input()
    if k == "r":
        try:
            # Run CNN
            f = open('CNN.py')
            source = f.read()
            exec(source)
        except Exception:
            print('CNN Fail')
            print(traceback.format_exc())
            pass
    elif k == "e":
        break