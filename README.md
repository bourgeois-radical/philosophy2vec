# Self-trained philosophical word2vec embeddings

## ...obtained out of Hegel's and Kant's corpora.

This project is a part of the means of assessment for the mandatory course 
"Machine Learning for Natural Language Understanding" of my NLP master's degree
program at Trier University in the winter semester 22/23. 

## Corpora:
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

## Main Tasks:
1. From-scratch implementation of Skip-Gram architecture: *from raw .txt files to embeddings*

2. Comparison of embeddings' similarity of the most frequently encountered
philosophical concepts in the corpora of Hegel and Kant (one language, two
philosophers)

3. Comparison of similarity of English and German embeddings for the same
philosopher i.e. corpora (two languages, one philosopher)

## Outcomes

### Task 1
[Top-k-accuracy](https://pytorch.org/docs/stable/generated/torch.topk.html) was used for model evaluation.

Check the results here:
* [Window size = 3, stop words included.](https://drive.google.com/drive/folders/1TCP6JXWiHNIi86XK45cPJnp_CsjQe0Sc?usp=sharing) (used in Outcomes and Inferences)
* [Window size = 4, without stop words.](https://drive.google.com/drive/folders/1pj_4nvWPVXE2CrqOw9xCQZ2vyuyZIQyc?usp=sharing) (bad results without stopwords due to relatively small corpora)

### Task 2
Comparison algorithm for one term looks as follows:
1. Take a term from Hegel’s vocabulary.
2. Find k-nearest terms for the selected term by means of cosine distance in
Hegel’s embeddings.
3. Take the same term from Kant’s vocabulary (if it is there at all).
4. Find k-nearest terms for it by means of cosine distance in Kant’s
embeddings.
5. Check, if there are some overlap-terms in k-nearest of Hegel and k-
nearest of Kant.


For example, we select the word “sei”, and find for it the word “Gott” in both Hegel’s
k-nearest to “sei” and Kant’s k-nearest to “sei”. Then the result of our function is a
tuple: ___(&#39;sei&#39;, &#39;gott&#39;)___ (see. `./notebooks/task_a`).

### German
The first 500 most frequent terms out of Hegel’s vocabulary were taken.
Given k-nearest=20, ___64 overlaps___ with Kant’s embeddings were obtained. A few interesting examples:

* _(&#39;wesen&#39;, &#39;existiert&#39;)_
* _(&#39;subjekt&#39;, &#39;prädikat&#39;)_
* _(&#39;ganze&#39;, &#39;mannigfaltige&#39;)_
* _(&#39;notwendig&#39;, &#39;kausalität&#39;)_

### English
The procedure is the same as for German. There are ___79 overlaps___ in total (given 500 terms). The most
interesting ones are:

* _(&#39;universal&#39;, &#39;entire&#39;)_
* _(&#39;law&#39;, &#39;rule&#39;)_
* _(&#39;world&#39;, &#39;intelligence&#39;)_
* _(&#39;animal&#39;, &#39;plant&#39;)_
* _(&#39;negative&#39;, &#39;positive&#39;)_

## Inferences

### Task 2
We observe fewer overlaps in German than it is in English corpora. One may assume the following reason: 
in general, German has more words than English, that's why density of occurence of a certain word in a German text can be
lower than of an English one. That implies a lesser chance to encounter this particular word in a context window 
of some center word.

It appears that Skip-Gram catches synonymous and antonymous patterns of common sense (e.g.:
`('law', 'rule')`, `('negative', 'positive')`, `('ganze', 'mannigfaltige')`) 
as well as of philosophical sense (e.g.: `('world', 'intelligence')`, `('subjekt', 'prädikat')`,
`('notwendig', 'kausalität')`) in the corpora, but the results for English and German embeddings hardly coincide with each other.

### Task 3
Given the results obtained in Task 1, one may assume that patterns, which would let conclude English and German 
embeddings are similar to some reasonable extent, are hardly to discover. The most evident explanation 
of this fact is the syntactical differences between the two languages. Moreover, in order to accomplish this task by
means of Python we would need a matching function between English and German
vocabularies. To write such a function is not a trivial task. But nevertheless, manual
cherry-picking can still show us some interesting results. One of the most frequent
words in English and German Hegel’s corpora is ___consciousness = Bewusstsein___. In
their 10-closest embeddings we found two similar words, namely: "individuality” (EN) and
“Person” (DE).

## How to use
`git clone https://github.com/bourgeois-radical/philosophy2vec.git`

Feel free to explore notebooks in `./notebooks/task_a` folder to train the model on texts of your choice and to check the results.

[Click here to download source texts, plots/results and obtained embeddings.](https://drive.google.com/drive/folders/1rWlO5mntEBYmmrJ30BiBFXxYXrCh8FpT?usp=sharing)
You can simply add these folders into project's root after the `git clone` command given above having been done.

## References
Mikolov, T., Chen, K., Corrado, G. & Dean J. (2013) Efficient estimation of word representations
in vector space. _International Conference on Learning Representations_. ICLR

Chernytska, O. (2021). Word2vec with PyTorch: Implementing the Original Paper.
https://towardsdatascience.com/word2vec-with-pytorch-implementing-original-paper-2cd7040120b0

## Weak Points
* No errors and exceptions inside the modules.
* Modest corpora. 

## TODO
* A matching function between English and German vocabularies.
* A function which prepares training data for CBOW architecture.


