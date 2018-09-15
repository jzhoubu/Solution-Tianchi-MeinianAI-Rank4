
def load_vocab_file_to_list(vocab_path):
    vocab = []

    lines = open(vocab_path, 'r', encoding='utf-8').readlines()
    for line in lines:
        line = line.strip()
        if line:
            vocab.append(line)
    vocab = list(set(vocab))

    return vocab

def rewrite_dictionary(output_path, *args):
    """
        将主语词典与描述词词典生成同一个词典
    :param noun_file:
    :param desc_file:
    :param output_path:
    :return:
    """

    words = []

    for arg in args:
        words.extend(open(arg, 'r', encoding='utf-8').readlines())

    words = sorted(words)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for word in words:
            word = word.strip()
            if word:
                if word.encode('UTF-8').isalpha():
                    output_file.write(word.upper())
                    output_file.write("\n")
                    output_file.write(word.lower())
                    output_file.write("\n")
                else:
                    output_file.write(word)
                    output_file.write("\n")

if __name__ == '__main__':
    noun_file = "../data/dictionary/noun.dic"
    desc_file = "../data/dictionary/desc.dic"
    output_file = "../data/dictionary/dict.dic"
    rewrite_dictionary(output_file, noun_file, desc_file)
