import pandas as pd
import csv
import os

root_path = "../Tamsui-River-DATA"
save_path = "./result"

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
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    for dirdir in os.listdir(root_path):
        # if dirdir is .DS_Store to continue
        if dirdir == ".DS_Store":
            continue
        
        now_to_path = "../Tamsui-River-DATA/" + dirdir
        for river_file in os.listdir(now_to_path):
            now_to_file = now_to_path + "/" + river_file
            if not river_file == "T9612010.A" and not os.path.isdir(now_to_file) and not river_file.endswith(".DS_Store") and not river_file.endswith(".git") and not river_file.endswith(".gitignore") and not river_file.endswith(".zip") and not river_file.endswith(".CSV") and not river_file.endswith(".csv") and not river_file.endswith(".gitignore") and not river_file.endswith(".ZIP") and not river_file.endswith(".EXE") and not river_file.endswith(".txt") and not river_file.endswith(".TXT") and not river_file.endswith(".doc") and not river_file.endswith(".INT") and not river_file.endswith(".OUT") and not river_file.endswith(".DAT") and not river_file.endswith(".py"):
                
                filename = save_path + "/" + river_file + ".csv"
                file = open(now_to_file, 'r')
                
                print("Processing " + now_to_file + "...")
                
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
                dataFrame.drop_duplicates(subset=['x'], keep='first', inplace=True)
                
                # print(dataFrame)
                
                os.remove(filename)
                dataFrame.to_csv(filename, index=False)
