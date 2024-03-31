# file_list = ["blink.txt", "ear.txt", "head.txt", "mar.txt", "perclos.txt", "yawn.txt"]
#
# for file_name in file_list:
#     file_prefix='./files/'
#     file_road=file_prefix+file_name
#     with open(file_road, "w") as file:
#         file.truncate(0)
import numpy as np
arr = []
with open("./files/ear.txt", "r") as f:
    name = [line.strip() for line in f if line.strip()]
    for num in name:
        arr.append(round(float(num), 3)*1000)
avg = np.mean(arr)
avg = round(avg, 2)
print(avg)