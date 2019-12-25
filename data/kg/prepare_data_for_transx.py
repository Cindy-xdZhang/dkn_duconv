def prepare_data(kg_in, triple_out, relation_out, entity_out):
    relation2index = {}
    entity2index = {}
    relation_list = []
    entity_list = []

    reader_kg = open(kg_in, encoding='utf-8')
    writer_triple = open(triple_out, 'w', encoding='utf-8')
    writer_relation = open(relation_out, 'w', encoding='utf-8')
    writer_entity = open(entity_out, 'w', encoding='utf-8')

    entity_cnt = 0
    relation_cnt = 0
    triple_cnt = 0

    print('reading knowledge graph ...')
    kg = reader_kg.read().strip().split('\n')

    print('writing triples to triple2id.txt ...')
    writer_triple.write('%d\n' % len(kg))
    for line in kg:
        array = line.split('\t')
        head = array[0]
        relation = array[1]
        tail = array[2]
        if head in entity2index:
            head_index = entity2index[head]
        else:
            head_index = entity_cnt
            entity2index[head] = entity_cnt
            entity_list.append(head)
            entity_cnt += 1
        if tail in entity2index:
            tail_index = entity2index[tail]
        else:
            tail_index = entity_cnt
            entity2index[tail] = entity_cnt
            entity_list.append(tail)
            entity_cnt += 1
        if relation in relation2index:
            relation_index = relation2index[relation]
        else:
            relation_index = relation_cnt
            relation2index[relation] = relation_cnt
            relation_list.append(relation)
            relation_cnt += 1
        writer_triple.write(
            '%d\t%d\t%d\n' % (head_index, tail_index, relation_index))
        triple_cnt += 1
    print('triple size: %d' % triple_cnt)

    print('writing entities to entity2id.txt ...')
    writer_entity.write('%d\n' % entity_cnt)
    for i, entity in enumerate(entity_list):
        writer_entity.write('%s\t%d\n' % (entity, i))
    print('entity size: %d' % entity_cnt)

    print('writing relations to relation2id.txt ...')
    writer_relation.write('%d\n' % relation_cnt)
    for i, relation in enumerate(relation_list):
        writer_relation.write('%s\t%d\n' % (relation, i))
    print('relation size: %d' % relation_cnt)

    reader_kg.close()

"""
把kg.txt 利用类似词汇表的方式，统计三元组、关系类型、实体的总数，
然后转成triple2id.txt,relation2id.txt,entity2id.txt。
WHY？？？？？？
triple2id.txt中三元组顺序变了：(head_index, tail_index, relation_index)
Q：kg.txt转成的entity2id.txt和news_preprocess 转出的entity2id一致吗？
news_preprocess在raw.txt中建立类似一般的class voc词汇表和实体表，
转出的entity2id负责把rawID转成统一ID，转换关系写在kg/entity2index.txt。
而prepare_data_for_transx也把kg.txt出现的实体rawID转换为自己的独立ID，暂称为KGid,写在kg/entity2id.txt，
目前看来KGid=rawID，故可以和统一ID对应。既是不一样，KGid和统一ID也可以通过rawID对应。
"""
if __name__ == '__main__':
    prepare_data(kg_in='kg.txt', triple_out='triple2id.txt', relation_out='relation2id.txt', entity_out='entity2id.txt')
