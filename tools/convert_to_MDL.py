import os


def convert(fileprefix, src_folder, output_folder):

    pos_train = open(os.path.join(src_folder, fileprefix + '-pos-train.txt')).readlines()
    neg_train = open(os.path.join(src_folder, fileprefix + '-neg-train.txt')).readlines()

    pos_test = open(os.path.join(src_folder, fileprefix + '-pos-test.txt')).readlines()
    neg_test = open(os.path.join(src_folder, fileprefix + '-neg-test.txt')).readlines()

    with open(os.path.join(output_folder, fileprefix + '_train'), 'w') as f:
        for sample in pos_train:
            f.write('1,{}'.format(sample))

        for sample in neg_train:
            f.write('0,{}'.format(sample))

    with open(os.path.join(output_folder, fileprefix + '_test'), 'w') as f:
        for sample in pos_test:
            f.write('{}'.format(sample))

        for sample in neg_test:
            f.write('{}'.format(sample))

    with open(os.path.join(output_folder, fileprefix + '_goldstandard'), 'w') as f:
        for sample in pos_test:
            f.write('1\n')

        for sample in neg_test:
            f.write('0\n')


if __name__ == '__main__':

    DATA_MDL = 'data_MDL'
    DATA_MDL_NORMALIZED = 'data_MDL_normalized'

    if not os.path.exists(DATA_MDL):
        os.makedirs(DATA_MDL)

    if not os.path.exists(DATA_MDL_NORMALIZED):
        os.makedirs(DATA_MDL_NORMALIZED)

    file_list = ['01-9bZkp7q19f0',
                 '04-CevxZvSJLk8',
                 '07-KQ6zr6kCPj8',
                 '08-uelHwf8o7_U',
                 '09-pRpeEdMmmQ0']

    for fileprefix in file_list:
        convert(fileprefix, 'data_split', DATA_MDL)
        convert(fileprefix, 'data_split_normalized', DATA_MDL_NORMALIZED)
