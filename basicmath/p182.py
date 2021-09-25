import numpy as np

docs = [
    'This is the first document',
    'This is the second document',
    'And the third one',
    'Is this the first document'
]

V = ['<NULL>', 'and', 'document', 'first', 'is', 'one', \
    'second', 'the', 'third', 'this']

N = max([len(doc) for doc in docs])
pre_docs = [doc + ['<NULL>']*(N-len(doc)) for doc in docs]
print(pre_docs)