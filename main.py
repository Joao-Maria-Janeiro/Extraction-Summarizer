from bert_embedding import BertEmbedding
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from nltk.corpus import stopwords 

stop_words = set(stopwords.words('english')) 

corpus = "If Cristiano Ronaldo didn’t exist, would Lionel Messi have to invent him? The question of how much these two other-worldly players inspire each other is an interesting one, and it’s tempting to imagine Messi sitting at home on Tuesday night, watching Ronaldo destroying Atletico, angrily glaring at the TV screen and growling: 'Right, I’ll show him!' As appealing as that picture might be, however, it is probably a false one — from Messi’s perspective, at least. He might show it in a different way, but Messi is just as competitive as Ronaldo. Rather than goals and personal glory, however, the Argentine’s personal drug is trophies. Ronaldo, it can be said, never looks happy on the field of play unless he’s just scored a goal — and even then he’s not happy for long, because he just wants to score another one. And that relentless obsession with finding the back of the net has undoubtedly played a major role in his stunning career achievements. Messi, though, is a different animal, shown by the generosity with which he sets up team-mates even if he has a chance to shoot, regularly hands over penalty-taking duties to others and invariably celebrates a goal by turning straight to the player who passed him the ball with an appreciative smile. Rather than being a better player than Ronaldo, Messi’s main motivations — according to the people who are close to him — are being the best possible version of Lionel Messi, and winning as many trophies as possible. That theory was supported by Leicester boss Brendan Rodgers when I interviewed him for a book I recently wrote about Messi. Do Messi and Ronaldo inspire each other? 'Maybe subconsciously in some way they\'ve driven each other on,' said Rodgers. 'But I think both those players inherently have that hunger to be the best players they can be. With the very elite performers, that drive comes from within.' Messi and Ronaldo ferociously competing with each other for everyone else’s acclaim is a nice story for fans to debate and the media to spread, but it’s probably not particularly true."

# Tokenize sentences
sentences = sent_tokenize(corpus)

# # Cleanup the sentences
# cleaned_sents = []

# for sentence in sentences:
#     cleaned_sents.append([w for w in sentence if not w in stop_words])


# Create sentence embeddings with bert
bert = BertEmbedding()
results = bert(sentences)

# Average the word emebddings in each sentence to create a single array for a sentence
averaged = []
for sent in results:
    averaged.append(np.mean(sent[1], axis = 0, dtype=np.float64))


# Cluster the data
n_clusters = int(np.ceil(len(averaged) * 0.3)) # Our output summary will be 30% of the size of the input corpus
kmeans = KMeans(n_clusters=n_clusters)
kmeans = kmeans.fit(np.array(averaged))

# Summarization
avg = []
for j in range(n_clusters):
    idx = np.where(kmeans.labels_ == j)[0]
    avg.append(np.mean(idx))
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, averaged)
ordering = sorted(range(n_clusters), key=lambda k: avg[k])
summary = ' '.join([sentences[closest[idx]] for idx in ordering])

# Display the summary
print(summary)