def get_list_from_file(input_file_name, col=0):
    '''get a list from a file column'''
    x_list = []
    input_file = open(input_file_name, 'r')
    for line in input_file:
        x = line.split()[col]
        x_list.append(x)
    input_file.close()
    return x_list
