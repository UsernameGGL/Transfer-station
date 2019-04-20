import csv
# 读入label
label_file = open('prodLabels.csv', 'r')
label_reader = csv.reader(label_file)
# print(list(label_reader)[0:5])
reader_list = list(label_reader)
# print(len(reader_list))
print(reader_list[0:10])
create_label = True
for line in reader_list:
	if line == []:
		continue
	if create_label:
		label = [list(map(int, line))]
		create_label = False
	else:
		label.append(list(map(int, line)))

print(len(label))
print(label[0:10])

