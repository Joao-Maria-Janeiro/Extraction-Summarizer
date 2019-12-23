# Extraction Summarizer

## Objective
Our objective for this project is, given a large corpus of text, create a short summary of 30% the size of the original corpus.

## Motivation
Everyday we are bombarded with tons and tons of articles, tweets, posts, news, etc... in written form, there is no way we can actually read everything if everything is quite long so, in a hope of being able to create automatic summaries for given corpus, I thought of doing this summarizer using Machine Learning.

## Details about summarization
There are two main types of summarizations being done:
* Extraction summarization
* Abstraction summarization

In short, in extraction summarization our algorithm will get the most relevant sentences from the original corpus and, with that, create the summary.

The abstraction summary is more similar to what we, humans, do. So abstraction summary will actually generate a new text from the corpus it was fed.

Abstraction summarization is harder and it will be featured in my next post, but for now let's focus on extraction summarization.

### How will we go about doing this?
There are a lot of ways to go about doing this, I picked the one that, from what I saw, gets the best results. Check [this article](https://towardsdatascience.com/comparing-text-summarization-techniques-d1e2e465584e) for a comparison of good possible methods.

The idea is quite simple, so we get the corpus, from this corpus que tokenize it into sentences. From these sentences we generate word embeddings with context and from these we create sentence embeddings. After we have sentence embeddings we simply cluster our data, the number of clusters being 30% of the size of the original corpus. Now that we have some clusters we simply pick one sentence per cluster (the sentence that's closest to the center of the cluster, which in theory, will be the best representation of the cluster) and then simply display those sentences.

#### Sentence tokenization
For tokenizing our corpus into sentences we will make use of the sentence tokenizer in the NLTK library.

#### Embeddings
This is the crucial part of our project. Why do we need embeddings? Why not just use Bag of Words or Tf-Idf? Since our project will be clustering the data, it will be comparing the vectors we generate in order to see how similar they are, bag of words does not capture the meaning of the words in the vector it generates so the generated clusters would be as good as random. 

So are we going to use Word2Vec or GloVe? The answer is no. Despite the popularity of these embeddors, they are not enough for the good accuracy of this project. Yes they do have some inherent meaning in the vectors in relation to the words, so if you were to do "King" - "Man" + "Woman" it would give you the vector that represents the word "Queen" but this embeddors do not get context as they are pre-trained vectors, the same word used with a different meaning would have the same vector, for example:

“The man was accused of robbing a bank.” “The man went fishing by the bank of the river.”

Word2Vec would produce the same word embedding for the word “bank” in both sentences, so we need something a little better.

There are a lot of possibilities for this task but I chose BERT, the Google model.

What is bert exactly and why is it better than Word2Vec or GloVe?

As I mentioned before BERT takes the context of the words into account, it will generate an embedding based on next and previous context (bidirectional, will explain ahead), so BERT is pretty much a layer in your model, you still need to train it.

So how does it work and why is it so good? BERT is really good because it applies the bidirectional training of Transformer (which is an attention model). Before papers were only looking at text sequences from left to right or right to left, BERT is bidirectional. The transformer can learn relations between words and sub-words (sub-words are pieces of words like splitting “subword” to “sub” and “word”). It has two types of training:
* So what BERT will do is, before feeding it into the model, will mask about 15% of the words in each sentence and try to guess those words. 
* It also tries to guess the next sentence, so the model gets a pair of sentences and learns to see if they are subsequent in the original corpus. 

For more information regarding BERT check these articles, from [Bharat S Raj]("https://towardsdatascience.com/understanding-bert-is-it-a-game-changer-in-nlp-7cca943cf3ad") and [Rani Horev]("https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270").

So with this we have a token for each word in a sentence, so to generate a vector for a sentence we will simply average out all the words in a sentence and, thus, generating a single vector each sentence.

For this we will leverage this [repo]("https://github.com/imgarylai/bert-embedding")

#### Clustering

Now that we have a vector for each sentence we can use cosine similarity to compare how similar each sentence is to each other and generate clusters. In order to do all this we will simply use leverage SKlearn's Kmeans.

#### Generating the summary
As I mentioned before, generating the summary now is just getting the sentence that's closest to the center of each cluster and print those sentences.


So let's look at what our code currently generates:

The source text is:

"If Cristiano Ronaldo didn’t exist, would Lionel Messi have to invent him? The question of how much these two other-worldly players inspire each other is an interesting one, and it’s tempting to imagine Messi sitting at home on Tuesday night, watching Ronaldo destroying Atletico, angrily glaring at the TV screen and growling: 'Right, I’ll show him!' As appealing as that picture might be, however, it is probably a false one — from Messi’s perspective, at least. He might show it in a different way, but Messi is just as competitive as Ronaldo. Rather than goals and personal glory, however, the Argentine’s personal drug is trophies. Ronaldo, it can be said, never looks happy on the field of play unless he’s just scored a goal — and even then he’s not happy for long, because he just wants to score another one. And that relentless obsession with finding the back of the net has undoubtedly played a major role in his stunning career achievements. Messi, though, is a different animal, shown by the generosity with which he sets up team-mates even if he has a chance to shoot, regularly hands over penalty-taking duties to others and invariably celebrates a goal by turning straight to the player who passed him the ball with an appreciative smile. Rather than being a better player than Ronaldo, Messi’s main motivations — according to the people who are close to him — are being the best possible version of Lionel Messi, and winning as many trophies as possible. That theory was supported by Leicester boss Brendan Rodgers when I interviewed him for a book I recently wrote about Messi. Do Messi and Ronaldo inspire each other? 'Maybe subconsciously in some way they\'ve driven each other on,' said Rodgers. 'But I think both those players inherently have that hunger to be the best players they can be. With the very elite performers, that drive comes from within.' Messi and Ronaldo ferociously competing with each other for everyone else’s acclaim is a nice story for fans to debate and the media to spread, but it’s probably not particularly true."

Our generated summary is:

If Cristiano Ronaldo didn’t exist, would Lionel Messi have to invent him? Ronaldo, it can be said, never looks happy on the field of play unless he’s just scored a goal — and even then he’s not happy for long, because he just wants to score another one. Rather than goals and personal glory, however, the Argentine’s personal drug is trophies. Messi and Ronaldo ferociously competing with each other for everyone else’s acclaim is a nice story for fans to debate and the media to spread, but it’s probably not particularly true. 'Maybe subconsciously in some way they've driven each other on,' said Rodgers.

## Conclusions

I tried using lemmatization and stemming but that didn't change the performance of our model. There are not many things that we could do to improve the performance of our model, our results are quite satisfactory so we could say this is a good method and a good model.

