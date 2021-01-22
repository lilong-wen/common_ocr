# generate dictionary from label file


label_file = './20K/formulas.txt'
output_file = './dictionary.txt'

total_item = []
with open(label_file, 'r') as label_f:
    label_lines = label_f.readlines()

    for label_line in label_lines:
        label = label_line.split('\t')[1].strip()
        total_item += label.split(' ')

total_item_set = set(total_item)

with open(output_file, 'w') as out_f:
    for num, item in enumerate(total_item_set):
        out_f.write(r''+item + "\t" + str(num) + "\n")
