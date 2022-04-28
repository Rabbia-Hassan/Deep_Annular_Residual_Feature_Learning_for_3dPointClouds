
# coding: utf-8

# In[62]:


import h5py
import numpy as np
import os


# In[63]:


def str_format(pth,txt,var):
    if var >= 0 and var <=9:
        strmid="_000"+str(var)
    if var >= 10 and var <=99:
        strmid="_00"+str(var)
    if var >= 100 and var <=10000:
        strmid="_0"+str(var)
    txt1=".txt"
    str0=txt+strmid
    #print("str0 is ",str0)
    str1=str0+txt1
    #print("str1 is ",str1)
    str2=pth+str1
    print("str2 is ",str2)
    return str2


# In[64]:


def str_format_test_and_train(txt,var):
    if var >= 0 and var <=9:
        strmid="_000"+str(var)
    if var >= 10 and var <=99:
        strmid="_00"+str(var)
    if var >= 100 and var <=10000:
        strmid="_0"+str(var)
    #txt1=".xyz"
    #str0=strmid
    str0=txt+strmid
    #print("str0 is ",str0)
    #str1=str0+txt1
    #print("str1 is ",str1)
    #str2=pth+str1
    #print("str0 is ",str0)
    return str0


# In[65]:


def getDataFiles(list_filename):
        return [line.rstrip() for line in open(list_filename)]


# In[66]:


def _get_data_filename(list_train,current_file_idx):
        return self.h5_files[self.file_idxs[self.current_file_idx]]


# In[67]:


def read_h5(filename_val):
    hf = h5py.File(filename_val, 'r')
    n1 = hf.get('data')  
    n2 = np.array(n1)
    normal=hf.get('normal')
    n2normal = np.array(normal)
    nall=np.concatenate((n2,n2normal),axis=2)
    labels=hf.get('label')
    labels = np.array(labels)
    return nall,labels


# In[76]:


def write(data,n3label,var,loop):
    for i in range(0,loop):
        nall=data[i]
        #print("nall shape is :", nall.shape)
        #print("I am inside write function for i =", i)
        #ip=input("next")
        #print(nall[0])
        if n3label[i]== 0:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/airplane/","airplane",var[0])
            var[0]=var[0]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 1:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/bathtub/","bathtub",var[1])
            var[1]=var[1]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 2:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/bed/","bed",var[2])
            var[2]=var[2]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 3:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/bench/","bench",var[3])
            var[3]=var[3]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 4:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/bookshelf/","bookshelf",var[4])
            var[4]=var[4]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 5:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/bottle/","bottle",var[5])
            var[5]=var[5]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 6:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/bowl/","bowl",var[6])
            var[6]=var[6]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 7:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/car/","car",var[7])
            var[7]=var[7]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 8:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/chair/","chair",var[8])
            var[8]=var[8]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 9:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/cone/","cone",var[9])
            var[9]=var[9]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 10:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/cup/","cup",var[10])
            var[10]=var[10]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 11:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/curtain/","curtain",var[11])
            var[11]=var[11]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 12:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/desk/","desk",var[12])
            var[12]=var[12]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 13:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/door/","door",var[13])
            var[13]=var[13]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 14:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/dresser/","dresser",var[14])
            var[14]=var[14]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 15:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/flower_pot/","flower_pot",var[15])
            var[15]=var[15]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 16:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/glass_box/","glass_box",var[16])
            var[16]=var[16]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 17:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/guitar/","guitar",var[17])
            var[17]=var[17]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 18:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/keyboard/","keyboard",var[18])
            var[18]=var[18]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 19:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/lamp/","lamp",var[19])
            var[19]=var[19]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 20:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/laptop/","laptop",var[20])
            var[20]=var[20]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 21:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/mantel/","mantel",var[21])
            var[21]=var[21]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 22:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/monitor/","monitor",var[22])
            var[22]=var[22]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 23:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/night_stand/","night_stand",var[23])
            var[23]=var[23]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 24:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/person/","person",var[24])
            var[24]=var[24]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 25:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/piano/","piano",var[25])
            var[25]=var[25]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 26:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/plant/","plant",var[26])
            var[26]=var[26]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 27:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/radio/","radio",var[27])
            var[27]=var[27]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 28:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/range_hood/","range_hood",var[28])
            var[28]=var[28]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 29:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/sink/","sink",var[29])
            var[29]=var[29]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 30:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/sofa/","sofa",var[30])
            var[30]=var[30]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 31:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/stairs/","stairs",var[31])
            var[31]=var[31]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 32:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/stool/","stool",var[32])
            var[32]=var[32]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 33:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/table/","table",var[33])
            var[33]=var[33]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 34:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/tent/","tent",var[34])
            var[34]=var[34]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 35:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/toilet/","toilet",var[35])
            var[35]=var[35]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 36:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/tv_stand/","tv_stand",var[36])
            var[36]=var[36]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 37:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/vase/","vase",var[37])
            var[37]=var[37]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 38:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/wardrobe/","wardrobe",var[38])
            var[38]=var[38]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
        if n3label[i]== 39:
            stri=str_format("data/modelnet40_ply_hdf5_2048/modelnet40/xbox/","xbox",var[39])
            var[39]=var[39]+1
            np.savetxt(stri, nall, delimiter=',') 
            print('saved')
    return var


# In[77]:


def create_dir(list_filename,subfolder_names):
    #os.makedirs('modelnet40')
    #os.mkdir (list_filename,"hello")
    #print("created :")
    #ip=input("Next ?")
    DATA_DIR = os.path.join(list_filename, 'modelnet40')
    os.mkdir(DATA_DIR)
    #subfolder_names = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard','lamp','laptop','mantel','monitor','night_stand','person','piano','plant','radio','range_hood','sink','sofa','stairs','stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
    for subfolder_name in subfolder_names:
        os.makedirs(os.path.join(DATA_DIR, subfolder_name))
    #print("value of DATA_DIR is :",DATA_DIR)
    


# In[78]:


def shape_names(list_filename,subfolder_names):
    #os.mkdir (os.path.join(list_filename,"modelnet40"))
    list_filename=list_filename+'/modelnet40/shape_names.txt'
    f= open(list_filename,"w+")
    for subfolder_name in subfolder_names:
        f.write(subfolder_name + '\n')
    f.close()
    


# In[79]:


def train_and_test(list_filename,subfolder_names):
    counts1_train = [1] * 40
    counts2_train=[625,106,515,173,572,335,64,197,889,167,79,137,200,109,200,149,171,155,145,124,149,284,465,200,88,231,239,104,115,128,680,124,90,392,163,344,267,475,87,103]
    #counts2_train=[77,201,117,347,395,149,181,221,267,241,135,105,118,254,82]
    #counts2_test=[725,156,615,193,672,435,84,297,989,99,157,286,129,286,169,271,255,165,144,169,384,565,286,108,331,339,124,215,148,780,144,110,492,183,444,367,575,107,123]
    counts2_test=[725,156,615,193,672,435,84,297,989,187,99,157,286,129,286,169,271,255,165,144,169,384,565,286,108,331,339,124,215,148,780,144,110,492,183,444,367,575,107,123]
    print('counts2_test s shape is :', len(counts2_test))
    
    list_filename_train=list_filename+'/modelnet40/modelnet40_train.txt'
    train_file = open(list_filename_train, "w")
    list_filename_test=list_filename+'/modelnet40/modelnet40_test.txt'
    test_file = open(list_filename_test, "w")
    for i in range(0,40):
        temp1=counts1_train[i]
        temp2=counts2_train[i]
        tempstr=subfolder_names[i]
        for j in range(temp1,temp2+1):
            str1=str_format_test_and_train(tempstr,j)
            train_file.write(str1)
            train_file.write("\n")

   

    for i in range(0,40):
        temp1=counts2_train[i]
        temp2=counts2_test[i]
        tempstr=subfolder_names[i]
        for j in range(temp1+1,temp2+1):
            str1=str_format_test_and_train(tempstr,j)
            test_file.write(str1)
            test_file.write("\n")
    test_file.close()
    train_file.close()


# In[80]:



def main():
    #my_list = os.listdir('C:\\Users\\rabbi\\3D Objects\\python notebooks\\github\\repo\\modelnet40_ply_hdf5_2048\\')
    print("Hello World!")
    list_filename='/home/rabbia/code/H5/repo/Deep_Annular_Residual_Feature_Learning_for_3dPointClouds/data/modelnet40_ply_hdf5_2048/train_files.txt'

    base_dir='/home/rabbia/code/H5/repo/Deep_Annular_Residual_Feature_Learning_for_3dPointClouds/data/modelnet40_ply_hdf5_2048/'
    subfolder_names = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard','lamp','laptop','mantel','monitor','night_stand','person','piano','plant','radio','range_hood','sink','sofa','stairs','stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

    
    create_dir(base_dir,subfolder_names)
    shape_names(base_dir,subfolder_names)
    train_and_test(base_dir,subfolder_names)
    list_train=getDataFiles(list_filename)
    #ip=input("directory created")
    
    #print(list_train)
    current_file_idx =0
    var = [1] * 40

    for i in range(0,5):
        #_get_data_filename(list_train,)
        current_file_val=list_train[current_file_idx]
        print("value is :",current_file_val)
        #ip=input("next")
        data,label=read_h5(current_file_val)
        current_file_idx=current_file_idx+1
        var=write(data,label,var,data.shape[0])
        #ip=input("out of the function Next :")

    list_filename_test='/home/rabbia/code/H5/repo/Deep_Annular_Residual_Feature_Learning_for_3dPointClouds/data/modelnet40_ply_hdf5_2048/test_files.txt'
    list_test=getDataFiles(list_filename_test)
    print(list_test)
    for i in range(0,2):
        current_file_val_test=list_test[i]
        data,label=read_h5(current_file_val_test)
        var=write(data,label,var,data.shape[0])
        #ip=input("out of the function Next :")
    print("done, Written")
        


# In[81]:


if __name__ == "__main__":
    main()

