class Config(object):
    unlabeled = False
    lowercase = False
    train_parser = True
    feature_template = True #Template (word, pos features for s0, s1, s2, s3, b0, b1, b2, b3, lc1(s0), rc1(s0), lc1(s1), rc1(s1), lc1(b0) )
#    template_size = 6
    use_pos = False # TODO make this configurable
#   use_dep = True
#   use_dep = use_dep and (not unlabeled)
    data_path = './data'
    train_file = 'tel_train.conll'
    test_file = 'tel_dev.conll'
    dev_file = 'tel_test.conll'
    embedding_file = './data/irshad_tel_word_embeddings.txt' #word embeddings
    pos_embedding_file = './data/tel_AnnCorra_POS_embeddings_dim60' #TODO add case for random initialization when POS embeddings are not available
    output_dir = 'parser_output/'
    parsing_algo = 'standard' # can be 'eager', 'standard' TODO 'eager_tree' -> arc-eager with tree constraint
    model_path = ''
    
    use_bert = True
    bert_layers = [-4]
    mode = 'concat' # 'concat' or 'add' layers
    concat_bert_with_word_vec = False
    bert_train_file_vectors = './data/bert_vectors/bert_train_vectors.jsonl'
    bert_test_file_vectors = './data/bert_vectors/bert_dev_vectors.jsonl'
    bert_dev_file_vectors = './data/bert_vectors/bert_test_vectors.jsonl'

