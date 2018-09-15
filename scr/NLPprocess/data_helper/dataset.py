import pickle
import numpy as np
import pandas as pd

from conf import preprocess_config

result = preprocess_config.result

def dataset_split(data_part1_path, data_part2_path, output_path, ratio=0.1, shuffle=True):
    data_part1 = pickle.load(open(data_part1_path, 'rb'))
    data_part2 = pickle.load(open(data_part2_path, 'rb'))

    # shuffle
    if shuffle:
        data_part1 = data_part1.reindex(np.random.permutation(data_part1.index))
    # split
    split_length = int(len(data_part1) * ratio)
    data_part2 = data_part2[:split_length]

    selected = result + ["vid"]
    non_selected = (set(data_part2.columns) - set(selected))
    data_part2 = data_part2.drop(non_selected, axis=1)

    data = pd.merge(data_part1, data_part2, how='right', on="vid")

    pickle.dump(data, open(output_path, 'wb'), pickle.HIGHEST_PROTOCOL)

def view_dataset(data_path):
    data = pickle.load(open(data_path, 'rb'))
    print(data['y1'])
    print(data['y2'])
    print(data['y3'])
    print(data['y4'])
    print(data['y5'])



def view_dataset_columns(data_path, column):
    data = pickle.load(open(data_path, 'rb'))[column]
    data_clean = data.dropna(axis=0, how='any')

    total_length, count = 0.0, 0
    for text in set(data_clean):
        if not isinstance(text, str):
            continue
        text = text.strip()
        count += 1
        total_length += len(text)
        print(text)

    print("Mean sentence length: {}".format(total_length / count))
    print("Length of data: {}".format(len(data)))
    print("Length of data not null: {}".format(sum(data.notnull())))
    print("Length of unique {}".format(len(set(data_clean))))

    print("Column: {}".format(column))

def write_result_columns(data_path1, data_path2):
    data1 = pickle.load(open(data_path1, 'rb'))
    data2 = pickle.load(open(data_path2, 'rb'))

    selected = ['y1', 'y2', 'y3', 'y4', 'y5', 'vid']
    data1 = data1[selected]
    data2 = data2[selected]

    data1_vid = data1['vid']
    data2 = data2.loc[data2.vid.isin(data1_vid)]

    data1 = data1.sort_values(by='vid')
    data2 = data2.sort_values(by='vid')

    data1.to_csv('../data/check/train_data_part1_full.csv', index=False)
    data2.to_csv('../data/check/0416-trainData2.csv', index=False)

if __name__ == "__main__":
    data_part1_path ="../data/dataset/train_data_part1.pickle"
    data_part1_simple_path = "../data/dataset/train_data_part1_simple.pickle"
    data_part1_full_path = "../data/dataset/train_data_part1_full.pickle"
    data_part2_path = "../data/dataset/train_data_part2.pickle"
    data_part2_origin_path = "../data/dataset/0429-testData2.pkl"
    data_part2_test_path = "../data/backup/test_data.pickle"
    output_path = "../data/dataset/train_data_part1_test.pickle"
    data_test_path = "../data/dataset/test_data_part1.pickle"
    # dataset_split(data_part1_path, data_part2_test_path, output_path, ratio=1)
    # view_dataset(output_path)
    view_dataset_columns(data_part1_full_path,'0709')
    # write_result_columns(output_path, data_part2_path_origin)

    # view_dataset(data_part2_path)
