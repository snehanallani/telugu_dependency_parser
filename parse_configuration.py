class Configuration:
    def __init__(self, sentence, pos, bert_vectors=[]):
        """ sentence is the list of words. The stack and buffer contain indices of words in the sentence."""
        self.stack = [0]
        self.buffer = [i + 1 for i in range(len(sentence)-1)]
        self.relations = [] # contains (head_index, child_index, label)
        self.sentence  = sentence
        self.pos = pos
        self.bert_vectors = bert_vectors

    def print_state(self):
        print('Stack ::: ', self.stack, ' Buffer ::: ', self.buffer, ' Arcs ::: ', self.relations)

    def get_state(self):
        return {'stack': self.stack.copy(), 'buffer': self.buffer.copy(), 'relations': self.relations.copy(), 'sentence': self.sentence, 'pos': self.pos, 'bert_vectors': self.bert_vectors}

