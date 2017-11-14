import csv

with open('data.csv', 'r') as f:
	reader = csv.reader(f)
	data = list(reader)

prev_proc = data[0][0]
prev_grid = data[0][1]
prev_step = data[0][2]
sum = 0.0
for i in data:
	if(prev_proc == i[0] and prev_grid == i[1] and prev_step == i[2]):
		sum = sum + float(i[4])
	else:
		print(prev_proc,prev_grid,prev_step,str(sum),sep='\t')
		sum = float(i[4])
		prev_proc = i[0]
		prev_grid = i[1]
		prev_step = i[2]
print(prev_proc,prev_grid,prev_step,str(sum),sep='\t')


input("Press Enter to continue...")
