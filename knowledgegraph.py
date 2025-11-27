import spacy
import re
from spacy.matcher import Matcher
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

nlp = spacy.load("en_core_web_trf")

def named_entity_recognition(sentence):

    doc = nlp(sentence)

    return doc.ents

def relation_extraction(sentence):

    doc = nlp(sentence)
    matcher = Matcher(nlp.vocab)
    pattern = [
        {'DEP':'ROOT'},
        {'DEP':'prep','OP':"?"},
        {'DEP':'agent','OP':"?"},
        {'POS':'ADJ','OP':"?"}
    ]
    matcher.add("relation_extraction", [pattern])
    matches = matcher(doc)
    span = doc[matches[-1][1]:matches[-1][2]]
    relation = span.text

    return relation

def text2triples(sentence):

    triples = []
    entities = named_entity_recognition(sentence)
    if len(entities) >= 2:
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                text_in_between = sentence[entities[i].start_char:entities[j].end_char]
                relation = relation_extraction(text_in_between)
                triples.append((entities[i], relation, entities[j]))

    return triples

def clean_twitter_text(text):

    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def save_kg_to_graphml(triples, filename="kg.graphml"):
    G = nx.DiGraph()
    for (subject, predicate, obj) in triples:
        subj_text = str(subject)
        obj_text = str(obj)
        G.add_node(subj_text)
        G.add_node(obj_text)
        G.add_edge(subj_text, obj_text, label=predicate)
    nx.write_graphml(G, filename)
    print(f"Knowledge graph saved to {filename}")

#text = """Meta just released TruthRL on Hugging Face A new RL framework directly optimizes LLM truthfulness. Using a ternary reward, it slashes hallucinations by 28.9% and boosts truthfulness by 21.1%, teaching models to recognize their knowledge boundaries. https://t.co/5bmpzHmoWx"""
#cleaned_text = clean_twitter_text(text)
#print(cleaned_text)
#text2triples(cleaned_text)


"""
sentences = [
    "Sir Arthur Conan Doyle was best known for his creation of the fictional detective Sherlock Holmes",
    "Elon Musk founded the SpaceX Inc.",
    "Mona Lisa was painted by Leonardo da Vinci.",
    "The Beatles were a famous rock band from Liverpool.",
    "Einstein formulated the theory of relativity.",
    "Mount Everest is the tallest mountain in the world.",
    "The Statue of Liberty was a gift from France to the United States."
]

triples = []
for sentence in sentences:
    triples += text2triples(clean_twitter_text(sentence))
triples
"""

df1, df2, df3 = pd.read_csv('data/collected.csv'), pd.read_csv('data/groundtruth.csv'), pd.read_csv('data/seed.csv')
df = pd.concat([df1, df2, df3], axis=0)

print('-'*50)
print(df.shape)
print(df.columns)


triples = []
for i, row in df.iterrows():
    triples += text2triples(clean_twitter_text(row['text']))

print('-'*50)
print(len(triples))
print('-'*50)
for i, t in enumerate(triples):
    print(i, t)

save_kg_to_graphml(triples)
