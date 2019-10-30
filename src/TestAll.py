from os import walk

from src.barDetection import image_detection

f = []
for (dirpath, dirnames, filenames) in walk("../TestFiles/"):
    f.extend(filenames)
    break

for x in filenames:
    print(f'Testing {x}')
    image_detection(f'../TestFiles/{x}')
