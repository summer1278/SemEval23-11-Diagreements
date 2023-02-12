"""
Return a fixed-length sentence embedding,
using pre-trained word embedding
"""
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
import numpy as np

flair_embedding_options= ['glove','extvec','crawl',
'twitter','turian','news'] #
"""
'en-glove' (or 'glove') English GloVe embeddings
'en-extvec' (or 'extvec')   English Komninos embeddings
'en-crawl' (or 'crawl') English FastText embeddings over Web crawls
'en-twitter' (or 'twitter') English Twitter embeddings
'en-turian' (or 'turian')   English Turian embeddings (small)
'en' (or 'en-news' or 'news')   English FastText embeddings over news and wikipedia data
'ar'    Arabic  Arabic FastText embeddings
"""

def load_word_embedding(doc,document_embedding):
    sentence = Sentence(doc)#doc.text
    document_embedding.embed(sentence)
    sent_embedding = sentence.embedding.cpu().detach().numpy()
    # print(sent_embedding)
    return sent_embedding

if __name__ == '__main__':
    doc = 'child is in the host family. are you here?'
    embedding_option= flair_embedding_options[0]
    embedding_model = WordEmbeddings(embedding_option)
    document_embedding = DocumentPoolEmbeddings([embedding_model])
    load_word_embedding(doc,document_embedding)