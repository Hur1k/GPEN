import glob
import os
import shutil

dirname = glob.glob(os.path.join("img_facescrub","*"))

for dir in dirname:
    name = os.path.split(dir)[-1]
    imgsPath = glob.glob(os.path.join(dir,"face","*.*"))
    for i in imgsPath:
        shutil.copy(i,os.path.join("img_facescrub_collection",f"{name}-{os.path.split(i)[-1][:-4]}.png"))
        pass


pass