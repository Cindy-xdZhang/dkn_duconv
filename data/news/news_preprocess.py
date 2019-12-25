import re
import os
import gensim
import numpy as np
"""
news/news_preprocess.py的步骤：
1.统计词频，entity 频率，并进行频率过滤，WORD_FREQ_THRESHOLD = 2,ENTITY_FREQ_THRESHOLD = 1.
2.建立类似一般的class voc词汇表和实体表，给每一个单词建立Index,使得每一个word和实体分别有了该项目世界里的ID,let's call it 统一ID.
另外还把这个统一ID转化关系写到：kg/entity2index.txt
3.建立localmap：把实体名中非英文字符去掉，然后小写化。然后根据实体名的组成单词建立单词2实体的map
4.利用23步骤的结果完成数据转化内：把raw txt的记录中，新闻标题的字符串替换成word idx串;
entity从原来的一个人工标注的ID和文字换成与标题的每个word，相关联的entity的统一ID串，空处为0（根据实体名的组成单词和实体建立的关联;
换了一下label记录项在一行中的位置。
"""
PATTERN1 = re.compile('[^A-Za-z]')#正则匹配范式：开头符号+字母串[A-Z a-z]匹配A-Z，a至z的所有字母就等于可以用来控制只能输入英文了
PATTERN2 = re.compile('[ ]{2,}')#正则匹配范式：长度为2到任意的空格
WORD_FREQ_THRESHOLD = 2
ENTITY_FREQ_THRESHOLD = 1
MAX_TITLE_LENGTH = 10
WORD_EMBEDDING_DIM = 50

word2freq = {}
entity2freq = {}
word2index = {}
entity2index = {}
corpus = []


def count_word_and_entity_freq(files):
    """
    Count the frequency of words and entities in news titles in the training and test files
    :param files: [training_file, test_file]
    :return: None
    """
    for file in files:
        reader = open(file, encoding='utf-8')
        for line in reader:
            array = line.strip().split('\t')
            # txt记录行：
            # 0	 north stonington private road ordinance finalized	0	14784:Stonington（uid，news title，label，entity）
            news_title = array[1]
            entities = array[3]

            # count word frequency
            for s in news_title.split(' '):
                if s not in word2freq:
                    word2freq[s] = 1
                else:
                    word2freq[s] += 1

            # count entity frequency
            for s in entities.split(';'):
                entity_id = s[:s.index(':')]
                if entity_id not in entity2freq:
                    entity2freq[entity_id] = 1
                else:
                    entity2freq[entity_id] += 1

            corpus.append(news_title.split(' '))
        reader.close()

#类似一般的class voc 建立词汇表，给每一个单词建立Index，可以变成onehot
# 使得每一个word和实体分别有了该项目世界里的ID,let's call it 统一ID
def construct_word2id_and_entity2id():
    """
    Allocate each valid word and entity a unique index (start from 1)
    :return: None
    """
    cnt = 1  # 0 is for dummy word
    for w, freq in word2freq.items():
        if freq >= WORD_FREQ_THRESHOLD:
            word2index[w] = cnt
            cnt += 1
    print('- word size: %d' % len(word2index))

    writer = open('../kg/entity2index.txt', 'w', encoding='utf-8')
    cnt = 1
    for entity, freq in entity2freq.items():
        if freq >= ENTITY_FREQ_THRESHOLD:
            entity2index[entity] = cnt#entity 是raw txt的entityID（比较混乱）， 把rawID 对应到新的从1开始的index
            writer.write('%s\t%d\n' % (entity, cnt))  # for later use
            cnt += 1
    writer.close()
    print('- entity size: %d' % len(entity2index))

#根据raw txt最后一列，对实体名进行去空格、去非英文字符，小写化
#然后建立word 2 对应的entity的统一ID的map：local map（实体名的组成单词和实体的关联）
def get_local_word2entity(entities):
    """
    Given the entities information in one line of the dataset, construct a map from word to entity index
    E.g., given entities = 'id_1:Harry Potter;id_2:England', return a map = {'harry':index_of(id_1),
    'potter':index_of(id_1), 'england': index_of(id_2)}
    by the way, remove non-character word and transform words to lower case
    :param entities: entities information in one line of the dataset
    :return: a local map from word to entity index
    """
    local_map = {}

    for entity_pair in entities.split(';'):
        entity_id = entity_pair[:entity_pair.index(':')]
        entity_name = entity_pair[entity_pair.index(':') + 1:]

        # remove non-character word and transform words to lower case
        entity_name = PATTERN1.sub(' ', entity_name)
        entity_name = PATTERN2.sub(' ', entity_name).lower()

        # constructing map: word -> entity_index
        for w in entity_name.split(' '):
            entity_index = entity2index[entity_id]#raw ID换位统一ID
            local_map[w] = entity_index#word 2 对应的实体的统一ID

    return local_map


def encoding_title(title, entities):
    """
    Encoding a title according to word2index map and entity2index map
    :param title: a piece of news title
    :param entities: entities contained in the news title
    :return: encodings of the title with respect to word and entity, respectively
    """
    local_map = get_local_word2entity(entities)#建立word 2 对应的entity的统一ID的map：local map

    array = title.split(' ')
    word_encoding = ['0'] * MAX_TITLE_LENGTH
    entity_encoding = ['0'] * MAX_TITLE_LENGTH

    point = 0
    for s in array:
        if s in word2index:#WORD_FREQ_THRESHOLD = 2
            word_encoding[point] = str(word2index[s])
            if s in local_map:
                entity_encoding[point] = str(local_map[s])
            point += 1
        if point == MAX_TITLE_LENGTH:
            break
    word_encoding = ','.join(word_encoding)#1728,732,5895,151,289,1224,1225,0,0,0
    entity_encoding = ','.join(entity_encoding)#1728,732,5895,151,289,1224,1225,0,0,0
    return word_encoding, entity_encoding


def transform(input_file, output_file):
    reader = open(input_file, encoding='utf-8')
    writer = open(output_file, 'w', encoding='utf-8')
    for line in reader:
        array = line.strip().split('\t')
        user_id = array[0]
        title = array[1]
        label = array[2]
        entities = array[3]
        word_encoding, entity_encoding = encoding_title(title, entities)
        #把raw txt的记录中，新闻标题的字符串替换成word idx串，
        # entity从原来的一个人工标注的ID和文字，
        # 换成与标题的每个word，相关联的entity的统一ID串（根据实体名的组成单词和实体建立的关联）
        writer.write('%s\t%s\t%s\t%s\n' % (user_id, word_encoding, entity_encoding, label))
    reader.close()
    writer.close()


def get_word2vec_model():
    if not os.path.exists('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model'):
        print('- training word2vec model...')
        w2v_model = gensim.models.Word2Vec(corpus, size=WORD_EMBEDDING_DIM, min_count=1, workers=16)
        print('- saving model ...')
        w2v_model.save('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model')
    else:
        print('- loading model ...')
        w2v_model = gensim.models.word2vec.Word2Vec.load('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model')
    return w2v_model


if __name__ == '__main__':
    print('counting frequencies of words and entities ...')
    count_word_and_entity_freq(['raw_train.txt', 'raw_test.txt'])

    print('constructing word2id map and entity to id map ...')
    construct_word2id_and_entity2id()

    print('transforming training and test dataset ...')
    transform('raw_train.txt', 'train.txt')
    transform('raw_test.txt', 'test.txt')
    # 把raw txt的记录中，新闻标题的字符串替换成word idx串，
    # entity从原来的一个人工标注的ID和文字，
    # 换成与标题的每个word，相关联的entity的统一ID串（根据实体名的组成单词和实体建立的关联）

    print('getting word embeddings ...')
    embeddings = np.zeros([len(word2index) + 1, WORD_EMBEDDING_DIM])
    model = get_word2vec_model()
    for index, word in enumerate(word2index.keys()):
        embedding = model[word] if word in model.wv.vocab else np.zeros(WORD_EMBEDDING_DIM)
        embeddings[index + 1] = embedding
    print('- writing word embeddings ...')
    np.save(('word_embeddings_' + str(WORD_EMBEDDING_DIM)), embeddings)
