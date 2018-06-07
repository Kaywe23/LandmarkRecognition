import pandas as pd
import os
import shutil
from shutil import copyfile
train_data = pd.read_csv('train.csv')
import urllib

temp = pd.DataFrame(train_data.landmark_id.value_counts().head(15000))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']
print temp
print temp.landmark_id.iloc[14950]
def createfolders():
    i = 0
    while i <= 14951:
        landmark = str(temp.landmark_id.iloc[i])
        path = r'./train_images/'+ landmark
        if not os.path.exists(path):
            os.makedirs(path)
        i+=1
#createfolders()


rootdirpics = './train/'
rootdirfolders = './train_images/'

def transformdata(path1, path2):

    n = 0
    while n <= 14951:
        t = train_data[(train_data.landmark_id == n)]
        print len(t.id)
        i = 1
        r =[]

        while i <= len(t.id):
            it = i - 1
            r.append(t.id.iloc[it])
            i += 1
                # print r


        for root, dirs, files in os.walk(rootdirpics):    # loop through startfolders
            for pic in files:
                p = os.path.splitext(pic)[0]   # all ids of pics (over 1 million pics)

                inpath = path1 + pic
                folder = str(n)
                outpath = path2 + folder
                if p in r:
                    print('move')
                    shutil.move(inpath, outpath)
        n+=1

#transformdata(rootdirpics, rootdirfolders)





