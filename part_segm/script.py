
import os
import evaluate
import pandas as pd
import csv
#import matplotlib.pyplot as plt
import numpy as np
def call():
    substring1="Try"
    substring="eval mean mIoU"
    arr=[substring,substring1]
    i=0
    substring2="Record"+".csv"
    print("Value of substring2 is :",substring2)
    #ip=input("Next :")
    print("The info extracted from log file has been sved into the file with the name :",substring2)
    file2= open(substring2,"w")
    writer = csv.writer(file2,delimiter=',')
    writer.writerow([substring1,substring])
    file2.close()
    for i in range(1,30):
        print("hello ,I am running ",i,"th instance :")
        os.system("python evaluate.py")
		#file2= open("dump/log_evaluate.txt","r")
        write_file(substring2)
def write_file(substring2):
    print("Hello :")
    substring1="Try"
    #substring="eval accuracy"
    substring="eval mean mIoU"
    arr=[substring,substring1]
    tokens=[]
    y_new=[]
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
    #file2.write("eval mean Loss \n") 
    with open ('log/log_eval/log_train.txt', 'rt') as myfile:    # Open lorem.txt for reading text.
        print(" have opened file :")
        for myline in myfile:                   # For each line in the file,
            mylines.append(myline.rstrip('\n')) # strip newline and add to list.
            if myline.find(substring) is not -1:
		print("inside if :")
                myline2=myline
                tokens=myline.split(':')
            #tokens1=tokens.rstrip('\n')
            #print("token is :",tokens[1])
            
            #file2.write(tokens[1])
                y=float(tokens[1])
                y_new.insert(count,y)
            #print([count,y])
                print("Y_new is :",y_new)
                writer1.writerow([count,y])
            #writer.writerow(y)
            #else:
		#print("shut up :")
            #file2.write(tokens[1])
            i=i+1
            count=count+1

        
    #print(mylines1)
    #print(y_new)
    file3.close()
    myfile.close()
    #graph(substring2,substring1,substring)


def main():
    call()

if __name__ == "__main__":
    main()


