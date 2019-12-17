from Project.Project_2.scripts.utils import *

LNDbID = 1
RadID = 1
FindingID = 1

# Read scan
[scan, spacing, origin, transfmat] = readMhd('data/LNDb-{:04}.mhd'.format(LNDbID))
print(spacing, origin, transfmat)

# Read segmentation mask
[mask, spacing, origin, transfmat] = readMhd('masks/LNDb-{:04}_rad{}.mhd'.format(LNDbID, RadID))
print(spacing, origin, transfmat)

# Read nodules csv
lines = readCsv('trainset_csv/trainNodules.csv')
header = lines[0]
nodules = lines[1:]

for nod in nodules:
    if int(nod[header.index('LNDbID')]) == LNDbID and int(nod[header.index('RadID')]) == RadID and int(
            nod[header.index('FindingID')]) == FindingID:
        ctr = np.array([float(nod[header.index('x')]), float(nod[header.index('y')]), float(nod[header.index('z')])])
        break

# Convert coordinates to image
transfmatToImg, transfmatToWorld = getImgWorldTransfMats(spacing, transfmat)
ctr = convertToImgCoord(ctr, origin, transfmatToImg)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(scan[int(ctr[2])])
axs[1].imshow(mask[int(ctr[2])])
plt.show()
