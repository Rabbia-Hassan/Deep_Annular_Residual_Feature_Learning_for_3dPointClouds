

import os
import evaluate
import pandas as pd
import csv
#import matplotlib.pyplot as plt
import numpy as np
def call():
    substring1="Try"
    substring="eval accuracy"
    substring_new1="eval mean mIoU1"
    substring_new2="eval mean mIoU (all shapes)"
    #arr=[substring,substring1,substring_new1,substring_new2]
    i=0
    substring2="Record"+".csv"
    print("Value of substring2 is :",substring2)
    #ip=input("Next :")
    print("The info extracted from log file has been sved into the file with the name :",substring2)
    file2= open(substring2,"w")
    writer = csv.writer(file2,delimiter=',')
    writer.writerow([substring1,substring,substring_new1,substring_new2])
    file2.close()
    for i in range(1,50):
        print("hello ,I am running ",i,"th instance :")
        #val='log_eval'+i
        val='log/log_eval'+str(i)
        #val1='log'+val
        #val1='log/'+val
        #val='log_eval'+str(i)
        #val=val+'.txt'
        print("val is :",val)
        #ip=input("Next :")
        #os.system("python evaluate.py %val")
        #os.system("python evaluate.py --log_dir log/log_eval+")
        cmdstring = "python evaluate.py --log_dir %s" % (val)
        print("cmdstring  is :",cmdstring )
        #ip=input("Next :")
        os.system(cmdstring)
        print("value of val is :",val)
        #ip=input("Next :?")
        #file2= open("dump/log_evaluate.txt","r")
        #write_file(substring2)
def write_file(substring2):
    print("Hello :")
    substring1="Try"
    substring="eval accuracy"
    substring_new="eval mean mIoU1"
    substring_new2="eval mean mIoU (all shapes)"
    arr=[substring_new,substring,substring1,substring_new2]
    tokens=[]
    tokens1=[]
    y_new=[]
    y_new1=[]
    mylines = []                                # Declare an empty list.
    #i=0
    #substring2=substring+".csv"
    i=0
    #substring2=substring+".csv"
    #print("The info extracted from log file has been sved into the file with the name :",substring2)
    #file2= open(substring2,"w")
    #writer = csv.writer(file2,delimiter=',')
    #writer.writerow([substring1,substring])
    #file2.close()
    file3= open("Record.csv","a")
    #print("file is opened :")
    writer1 = csv.writer(file3,delimiter=',')
    #writer = csv.writer(file2,delimiter=',')
    #print([substring1,substring])
    count=0
    count1=0
    #file2.write("eval mean Loss \n") 
    with open ('log/log_eval/log_train.txt', 'rt') as myfile:    # Open lorem.txt for reading text.
        print(" have opened file :")
        for myline in myfile:                   # For each line in the file,
            mylines.append(myline.rstrip('\n')) # strip newline and add to list.
            if myline.find(substring) is not -1:
		#print("inside if :")
                myline2=myline
                myline3=myline
                tokens=myline.split(':')
            #tokens1=tokens.rstrip('\n')
            #print("token is :",tokens[1])
            
            #file2.write(tokens[1])
                y=float(tokens[1])
                y_new.insert(count,y)
            #print([count,y])
                print("Y_new is :",y_new)
                #writer1.writerow([count,y])
                save1=y
    myfile.close()
    with open ('log/log_eval/log_train.txt', 'rt') as myfile1:    # Open lorem.txt for reading text.
        print(" have opened file :")
        for myline in myfile1:                   # For each line in the file,
            mylines.append(myline.rstrip('\n')) # strip newline and add to list.
            if myline.find(substring_new) is not -1:
                print("hello hahahah")
               #print("inside if :")
                myline3=myline
                tokens1=myline.split(':')
            #tokens1=tokens.rstrip('\n')
            #print("token is :",tokens[1])
            
            #file2.write(tokens[1])
                y1=float(tokens1[1])
                y_new1.insert(count1,y1)
            #print([count,y])
                print("Y_new1 is :",y_new)
                save2=y1
    myfile1.close()
    with open ('log/log_eval/log_train.txt', 'rt') as myfile2:    # Open lorem.txt for reading text.
        print(" have opened file :")
        for myline in myfile2:                   # For each line in the file,
            mylines.append(myline.rstrip('\n')) # strip newline and add to list.
            if myline.find(substring_new2) is not -1:
                print("hello hahahah")
               #print("inside if :")
                myline3=myline
                tokens2=myline.split(':')
            #tokens1=tokens.rstrip('\n')
            #print("token is :",tokens[1])
            
            #file2.write(tokens[1])
                y2=float(tokens2[1])
                y_new1.insert(count1,y2)
            #print([count,y])
                print("Y_new1 is :",y_new)
                save3=y2
    writer1.writerow([count1,save1,save2,save3])   
            #writer.writerow(y)
            #else:
		#print("shut up :")
            #file2.write(tokens[1])
    i=i+1
    count=count+1
    count1=count1+1

        
    #print(mylines1)
    #print(y_new)
    file3.close()
    myfile.close()
    myfile2.close()
    #graph(substring2,substring1,substring)


def main():
    call()

if __name__ == "__main__":
    main()


