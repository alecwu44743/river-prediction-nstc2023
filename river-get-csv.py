import pandas as pd
import csv
import os

file_path = "./"

filename = "river.csv"
csv_header = ['x', 'y']
river_data = []

def isSpace(c):
    if c == ' ':
        return True
    else:
        return False
    

def writeCSV(data):
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(data)


def get_data(data):
    is_x = 1
    
    temp = ""
    temp_ll = []
    for i in range(len(data)):
        if not isSpace(data[i]):
            temp += data[i]
            # print(temp)
            # print(type(temp))
        if (isSpace(data[i]) and not isSpace(data[i+1]) and not temp == "") or (i == (len(data) - 1)):
            if is_x == 1:
                # print("temp1: " + temp)
                temp_ll.append(str(float(temp)))
                # print(temp_ll)
                is_x = 0
                temp = ""
            else:
                # print("temp2: " + temp)
                temp_ll.append(str(float(temp)))
                # print(temp_ll)
                is_x = 1
                temp = ""
                writeCSV(temp_ll)
                # print(river_data)
                temp_ll.clear()


if __name__ == '__main__':
    for river_file in os.listdir():
        if river_file.startswith("T"):
            
            file = open(river_file, 'r')
            filename = river_file + ".csv"
            
            lines = file.readlines()
            with open(filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(csv_header)
            
            line_index = 0
            for data in lines:
                if line_index >= 2:
                    if len(data) < 2:
                        break
                    
                #     print(ll.strip())
                #     print(len(ll))
                    get_data(data.strip())
                
                    
                line_index += 1
            
            
            dataFrame = pd.read_csv(filename)
            dataFrame.sort_values('x' , axis=0, ascending=True,inplace=True, na_position='first')
            
            print(dataFrame)
            
            os.remove(filename)
            dataFrame.to_csv(filename, index=False)
