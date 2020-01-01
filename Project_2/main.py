from os import walk

import cv2 as cv
import numpy as np
from sklearn.linear_model import SGDClassifier

from Project.Project_2.scripts.utils import *

# Get the mhd images
data = [file for file in list(walk('data'))[0][2] if file.endswith('.mhd')]
data = sorted(data)

# Read nodules csv
lines = readCsv('trainset_csv/trainNodules_gt.csv')
header = lines[0]
nodules = lines[1:]

# Create Bag of Words
detector = cv.KAZE_create()
matcher = cv.FlannBasedMatcher()
bowTrainer = cv.BOWKMeansTrainer(100)
bowExtractor = cv.BOWImgDescriptorExtractor(detector, matcher)

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
                Class = 'Ground Glass Opacities (GGO)'
            elif 2.3 <= Texture < 3.6:
                Class = 'Part Solid'
            elif 3.6 <= Texture:
                Class = 'Solid'
            else:
                Class = None
        else:
            Class = 'Not a Nodule'

        print(f"ID {LNDbID} - Finding {FindingID} - Class {Class} - Radiologists {RadID} - xyz {ctr}")

        # Read image
        img = np.array(scan[int(ctr[2])]).astype('float32')
        if img.shape != (512, 512):
            continue

        # Extract image's key points and descriptor
        kp, des = detector.detectAndCompute(img, None)

        # Add image's descriptor to BoW
        if des is not None:
            bowTrainer.add(des)

            LNDb[f'{LNDbID}_{FindingID}'] = {}
            LNDb[f'{LNDbID}_{FindingID}']['image'] = img
            LNDb[f'{LNDbID}_{FindingID}']['class'] = Class
            LNDb[f'{LNDbID}_{FindingID}']['keypoints'] = kp
            LNDb[f'{LNDbID}_{FindingID}']['descriptor'] = des

print('LOOP FINISHED')

# Set BoW vocabulary
print('CREATING VOCABULARY')
bowExtractor.setVocabulary(bowTrainer.cluster())

print('COMPUTING')
# Compute images
for id, values in LNDb.items():
    LNDb[id]['bow'] = np.linalg.norm(bowExtractor.compute(values['image'], values['keypoints'], values['descriptor']))
    print(LNDb[id]['bow'])

# Classifier
clf = SGDClassifier()
