# Self-trained philosophical word2vec embeddings

## ...obtained out of Hegel's and Kant's corpora.

### Corpora:

The texts of great German philosophers Hegel and Kant were chosen as the
corpora for training the model, namely:

* Kant, Kritik der reinen Vernunft, 1787
* Kant, Kritik der Urteilskraft, 1790
* Hegel, Phänomenologie des Geistes, 1807
* Hegel, Wissenschaft der Logik (Enzyklopädie), 1830

All four texts are taken both in 
German and in English (from gutenberg.org), so that the first two build Kant’s 
corpora (`DE_KANT_CORPUS.txt`, `ENG_KANT_CORPUS.txt`) and the other two Hegel’s ones 
(`DE_HEGEL_CORPUS.txt`, `ENG_HEGEL_CORPUS.txt`).

### Main Tasks:

1. From-scratch implementation of Skip-Gram architecture: *from raw .txt files to embeddings*

2. Comparison of embeddings' similarity of the most frequently encountered
philosophical concepts in the corpora of Hegel and Kant (one language, two
philosophers)

3. Comparison of similarity of English and German embeddings for the same
philosopher i.e. corpora (two languages, one philosopher)



