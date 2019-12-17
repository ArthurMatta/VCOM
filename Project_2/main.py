from os import walk

from matplotlib import pyplot as plt

from Project.Project_2.scripts.utils import *

# Get the mhd images
data = [file for file in list(walk('data'))[0][2] if file.endswith('.mhd')]
data = sorted(data)

# Get the mhd masks
masks = [file for file in list(walk('masks'))[0][2] if file.endswith('.mhd')]
masks = sorted(masks)

lines = readCsv('trainset_csv/trainNodules.csv')
header = lines[0]
nodules = lines[1:]

# Build a dict based on the LNDb ID
LNDb = {}
for d in data:
    id = d.strip('LNDb-').strip('.mhd').lstrip('0')
    m = [mask for mask in masks if mask.lstrip('LNDb-')[:4].lstrip('0') == id]
    n = [nodule for nodule in nodules if nodule[header.index('LNDbID')] == id]

    LNDb[id] = {}
    LNDb[id]['Image'] = d
    LNDb[id]['Masks'] = m
    LNDb[id]['Nodules'] = n

for LNDbID, LNDbValues in LNDb.items():
    LNDbImage = LNDbValues['Image']
    LNDbMasks = LNDbValues['Masks']
    LNDbNodules = LNDbValues['Nodules']

    [scan, spacing, origin, transfmat] = readMhd(f'data/{LNDbImage}')

    for Mask in LNDbMasks:
        [mask, spacing, origin, transfmat] = readMhd(f'masks/{Mask}')
        RadID = Mask[::-1][4]

        for nod in LNDbNodules:
            if nod[0] == LNDbID and nod[1] == RadID:
                FindingID = nod[2]
                ctr = np.array(
                    [float(nod[header.index('x')]), float(nod[header.index('y')]), float(nod[header.index('z')])])

                transfmat_toimg, transfmat_toworld = getImgWorldTransfMats(spacing, transfmat)
                ctr = convertToImgCoord(ctr, origin, transfmat_toimg)

                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(scan[int(ctr[2])])
                axs[1].imshow(mask[int(ctr[2])])
                plt.title(f'LNDbID: {LNDbID} | RadID: {RadID} | FindingID: {FindingID}')
                plt.show()
