import pitch
import os
import csv

row_list = [["speaker_id", "pitch"]]
folder = 'd:/Voice-Cloning/vctk-inputs'
i=0
for filename in os.listdir(folder): 
    print(i)
    path = folder + "/" + filename
    p = pitch.find_pitch(path)
    print(p)
    row_list.append([filename, p])
    i=i+1

with open('input_pitch.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)

print('Saved successfully')