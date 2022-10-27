import pandas as pd
import csv
import os

# T 58 12 008

root_path = "./result_non-duplicated"
save_path = "./river_lowest_ponit.csv"

# start df.iloc[0, 1]

if __name__ == "__main__":
    riv = [ x for x in range(90, -1, -1) ]
    df = pd.DataFrame({ "year" : riv})
    
    for i in range(58, 110):
        df[i] = ""
    
    # with open(save_path, 'a') as csvfile:
    #     csvwriter = csv.writer(csvfile) 
    #     csvwriter.writerow(csv_header)
        
    for dirdir in os.listdir(root_path):
        # if dirdir is .DS_Store to continue
        if dirdir == ".DS_Store":
            continue
        
        print("Processing: " + "./lab/" + dirdir)
        
        ans = os.path.splitext(dirdir)[0]
        
        # print(ans, end="")
        if ans.rfind(".") != -1:
            ans = ans.replace(ans[ans.rfind("."):], "")
        
        if ans.rfind("-") != -1:
            ans = ans.replace(ans[ans.rfind("-"):], "")
            
        if ans.rfind("A") != -1:
            ans = ans.replace(ans[ans.rfind("A"):], "")

        ans = ans.replace("T", "")

        year = ""
        reach = ""
        
        if ans[0] == "1":
            year = int(ans[0:3])
            reach = int(ans[6:])
        else:
            year = int(ans[0:2])
            reach = int(ans[5:])

        col_pos = year - 58 + 1
        row_pos = abs(reach - 90)
        
        pd_read = pd.read_csv(root_path + "/" + dirdir)
        _col = pd_read["y"]
        _lowest = _col.min()
        
        # print(" -> " + str(year) + " " + str(reach) + " " + str(_lowest))
        
        if df.iloc[row_pos, col_pos] == "" or df.iloc[row_pos, col_pos] > _lowest:
            df.iloc[row_pos, col_pos] = _lowest


    df.to_csv(save_path, index=False)