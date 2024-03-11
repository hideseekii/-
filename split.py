import os
import random
import shutil
from shutil import copy2


def image_split():    
    
    datadir_normal = '/TOPIC/ProjectB/B_traing1'

    all_data = os.listdir(datadir_normal)
    num_all_data = len(all_data)
    print( "num_all_data: " + str(num_all_data) )
    index_list = list(range(num_all_data))


    trainDir = "/home/112060/project_b/train80"
    if not os.path.exists(trainDir):
        os.mkdir(trainDir)
    else :
        shutil.rmtree(trainDir)
        os.mkdir(trainDir)
            
    validDir = '/home/112060/project_b/val20'
    if not os.path.exists(validDir):
        os.mkdir(validDir)
    else :
        shutil.rmtree (validDir)
        os.mkdir(validDir)
            
        
    for i in index_list:
        dirName = os.path.join(datadir_normal, all_data[i])
        all_imge = os.listdir(dirName)
        imge_list = list(range(len(all_imge)))
        random.shuffle(imge_list)
        
        num = 0 
        for k in imge_list:
            filename = os.path.join(dirName, all_imge[k])
        
            if num < len(all_imge)*0.8:
                #print(str(fileName))
                tempdir = os.path.join(trainDir, all_data[i])
                if not os.path.exists(tempdir):
                    os.mkdir(tempdir)
                copy2(filename, tempdir)
            else:
                #print(str(fileName))
                tempdir = os.path.join(validDir, all_data[i])
                if not os.path.exists(tempdir):
                    os.mkdir(tempdir)
                copy2(filename, tempdir)
            num += 1
    
    
if __name__ == '__main__':      
    image_split()