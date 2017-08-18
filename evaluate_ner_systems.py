#!/usr/bin/env
# -*- coding: utf-8 -*-

from __future__ import division

import csv
import io
import os
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from itertools import izip
from subprocess import call

from polyglot.text import Text
sys.path.insert(0, "MITIE/mitielib")
import mitie
import nltk
import spacy

sys.path.insert(0, "TwitterNER")
from NoisyNLP.features import sent2labels, sent2features, DictionaryFeatures, ClusterFeatures, preprocess_token, WordVectors
from NoisyNLP.models import CRFModel
from NoisyNLP.utils import load_sequences, Tag, process_glovevectors

TYPE_MAP = {"person": "PERSON",
"company": "ORGANIZATION",
"geo-loc": "LOCATION",
"band": "ORGANIZATION",
"musicartist": "ORGANIZATION",
"sportsteam": "ORGANIZATION",
"facility": "LOCATION",
"movie": None,
"tvshow": None,
"other": None,
"product": None,
"NONE": None}

POLYGLOT_TYPE_MAP = {"I-PER": "PERSON",
"I-ORG": "ORGANIZATION",
"I-LOC": "LOCATION"
}

SPACY_TYPE_MAP = {"PERSON": "PERSON",
"FACILITY": "LOCATION",
"ORG": "ORGANIZATION",
"GPE": "LOCATION",
"LOC": "LOCATION"}

FININ_TYPE_MAP = {"PER": "PERSON",
"LOC": "LOCATION",
"ORG": "ORGANIZATION"}

NLTK_TYPE_MAP = {"ORGANIZATION": "ORGANIZATION",
"PERSON": "PERSON",
"LOCATION": "LOCATION",
"DATE": None,
"TIME": None,
"MONEY": None,
"PERCENT": None,
"FACILITY": "LOCATION",
"GPE": "LOCATION",
"GSP": "LOCATION"
}

WNUT_TYPE_MAP = {
"group": "ORGANIZATION",
"corporation": "ORGANIZATION",
"location": "LOCATION",
"person": "PERSON",
"creative-work": None,
"product": None
}

GOOD_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION"]

TRAIN_FILE = "twitter_nlp/data/annotated/wnut16/data/train"
TEST_FILE = "twitter_nlp/data/annotated/wnut16/data/test"
DEV_FILE = "twitter_nlp/data/annotated/wnut16/data/dev"
UNTOKENIZED_TEST_FILE = "test_untokenized.txt"
OSU_NLP_OUTPUT_FILE = "test_untokenized_out.txt"
WNUT_TRAIN_FILE = "emerging.test.annotated"
HEGE_TRAIN_FILE = "hege.test.tsv"
FININ_TRAIN_FILE = "finin.train"
FININ_TEST_FILE = "finin.test"

UNDEFINED = "undefined"

TWITTER_NER_MODEL_FILE = "twitter_ner_model.pkl"
TWITTER_NER_WNUT_TRAINING_DATA_MODEL_FILE = "twitter_ner_wnut_training_data_model.pkl"
TWITTER_NER_HEGE_TRAINING_DATA_MODEL_FILE = "twitter_ner_hege_training_data_model.pkl"
TWITTER_NER_FININ_TRAINING_DATA_MODEL_FILE = "twitter_ner_finin_training_data_model.pkl"
TWITTER_NER_WNUT_AND_HEGE_TRAINING_DATA_MODEL_FILE = "twitter_ner_wnut_and_hege_training_data_model.pkl"

DICTIONARY_DIR = "TwitterNER/data/cleaned/custom_lexicons/"
WORDVEC_FILE_RAW = "glove.twitter.27B.200d.txt"
WORDVEC_FILE_PROCESSED = "glove.twitter.27B.200d.txt.processed.txt"
GIMPLE_TWITTER_BROWN_CLUSTERS_DIR = "50mpaths2"
TEST_ENRICHED_DATA_BROWN_CLUSTER_DIR = "brown_clusters"
BROWN_EXEC_PATH = "brown-cluster/wcluster"
BROWN_INPUT_DATA_PATH = "all_sequences.brown.txt"
TEST_ENRICHED_DATA_CLARK_CLUSTER_DIR = "clark_clusters"
CLARK_EXEC_PATH = "clark_pos_induction/src/bin/cluster_neyessenmorph"
CLARK_INPUT_DATA_PATH = "all_sequences.clark.txt"

def write_scores(row, gold, system):
    intersection_size = len(gold & system)
    precision = UNDEFINED
    if len(system):
        precision = intersection_size / len(system)
    recall = UNDEFINED
    if len(gold):
        recall = intersection_size / len(gold)
    f1_score = UNDEFINED
    if precision != UNDEFINED and recall != UNDEFINED and precision + recall:
        f1_score = 2 * precision * recall / (precision + recall)

    row.extend([precision, recall, f1_score])

def parse_label(label):
    state = label[0]
    if state == "O":
        entity_type = None
    else:
        entity_type = TYPE_MAP[label[2:]]
    return (state, entity_type)

def get_gold_entities():
    entities = set()
    untokenized = ""
    with io.open(TEST_FILE, "r", encoding="utf8") as test_data_in:
        entity_start = None
        previous_entity_type = None
        for line in test_data_in:
            stripped_line = line.strip()
            if stripped_line:
                (token, label) = stripped_line.split("\t")
                (state, entity_type) = parse_label(label)
                if entity_start is not None and state in ("O", "B"):
                    entities.add((entity_start, len(untokenized), previous_entity_type))
                    entity_start = None
                if untokenized and untokenized[-1] != "\n" and token != "'s":
                    untokenized += " "
                if state == "B" and entity_type is not None:
                    entity_start = len(untokenized)
                untokenized += token
                previous_entity_type = entity_type
            else:
                if entity_start is not None:
                    entities.add((entity_start, len(untokenized), previous_entity_type))
                if untokenized[-1] != "\n":
                    untokenized += "\n"
                entity_start = None
                previous_entity_type = None
        if entity_start is not None:
            entities.add((entity_start, len(untokenized), previous_entity_type))

    with io.open(UNTOKENIZED_TEST_FILE, "w", encoding="utf8") as test_data_out:
        test_data_out.write(untokenized)
    return (entities, untokenized)

def get_osu_nlp_entities(original_tweets):
    os.chdir("twitter_nlp")
    os.environ["TWITTER_NLP"] = "./"
    call(["python", "python/ner/extractEntities.py", os.path.join("..", UNTOKENIZED_TEST_FILE), "-o", OSU_NLP_OUTPUT_FILE, "--classify"])
    system_entities = set()
    with io.open(OSU_NLP_OUTPUT_FILE, "r", encoding="utf8") as annotated_file:
        original_index = 0
        previous_token_end = 0

        for line in annotated_file:
            previous_entity_type = None
            entity_start = None

            stripped_line = line.strip()
            if not stripped_line:
                continue
            labeled_tokens = stripped_line.split()
            for labeled_token in labeled_tokens:
                (token, label) = labeled_token.rsplit("/", 1)
                (state, entity_type) = parse_label(label)
                original_index = original_tweets.index(token, previous_token_end)
                if entity_start is not None and state in ("O", "B"):
                    system_entities.add((entity_start, previous_token_end, previous_entity_type))
                    entity_start = None
                if state == "B" and entity_type is not None:
                    entity_start = original_index
                previous_entity_type = entity_type
                previous_token_end = original_index + len(token)
            if entity_start is not None:
                system_entities.add((entity_start, previous_token_end, previous_entity_type))
    os.chdir("..")
    return system_entities

def get_stanford_entities(truecase=False, caseless=False, twitter_pos=False):
    os.chdir("stanford-corenlp-full-2017-06-09")
    annotators = "tokenize,ssplit,"
    if truecase:
        annotators += "truecase,"
    annotators += "pos,lemma,ner"
    command = ["./corenlp.sh", "-annotators", annotators, "-outputFormat", "xml", "-file",
               os.path.join("..", UNTOKENIZED_TEST_FILE),
               "-ssplit.newlineIsSentenceBreak", "always"]
    if truecase:
        command.extend(["-truecase.overwriteText", "true"])
    if caseless:
        command.extend(["-pos.model",
                        "edu/stanford/nlp/models/pos-tagger/english-caseless-left3words-distsim.tagger",
                        "-parse.model",
                        "edu/stanford/nlp/models/lexparser/englishPCFG.caseless.ser.gz",
                        "-ner.model",
                        "edu/stanford/nlp/models/ner/english.all.3class.caseless.distsim.crf.ser.gz,edu/stanford/nlp/models/ner/english.muc.7class.caseless.distsim.crf.ser.gz,edu/stanford/nlp/models/ner/english.conll.4class.caseless.distsim.crf.ser.gz"])
    if twitter_pos:
        command.extend(["-pos.model", "gate-EN-twitter.model"])
    call(command)
    system_entities = set()
    tree = ET.parse(UNTOKENIZED_TEST_FILE + ".xml")
    root = tree.getroot()
    sentences = root.find("document").find("sentences")

    for sentence in sentences.iter("sentence"):
        previous_entity_type = None
        entity_start = None

        tokens = sentence.find("tokens")
        for token_element in tokens.iter("token"):
            token = token_element.find("word").text
            raw_ner_tag = token_element.find("NER").text

            entity_type = "O"
            if raw_ner_tag in GOOD_ENTITY_TYPES:
                entity_type = raw_ner_tag

            if entity_type != previous_entity_type:
                if entity_start is not None:
                    system_entities.add((entity_start, previous_token_end, previous_entity_type))
                    entity_start = None
                if entity_type != "O":
                    entity_start = int(token_element.find("CharacterOffsetBegin").text)
            previous_entity_type = entity_type
            previous_token_end = int(token_element.find("CharacterOffsetEnd").text)

        if entity_start is not None:
            system_entities.add((entity_start, previous_token_end, previous_entity_type))

    os.chdir("..")
    return system_entities

def get_polyglot_entities(original_tweets):
    system_entities = set()
    original_index = 0
    previous_token_end = 0

    for tweet in original_tweets.split("\n"):
        entity_start = None

        stripped_tweet = tweet.strip()
        if not stripped_tweet:
            continue

        text = Text(stripped_tweet, hint_language_code='en')
        entities = [(e.start, e.end, POLYGLOT_TYPE_MAP[e.tag]) \
                    for e in text.entities]
        if entities:
            current_entity = entities.pop(0)
        else:
            current_entity = None
        for i, token in enumerate(text.words):
            original_index = original_tweets.index(token, previous_token_end)
            if entity_start is not None and i == current_entity[1]:
                system_entities.add((entity_start, previous_token_end, current_entity[2]))
                entity_start = None
                if entities:
                    current_entity = entities.pop(0)
                else:
                    current_entity = None
            if current_entity is not None and i == current_entity[0]:
                entity_start = original_index
            previous_token_end = original_index + len(token)
        if entity_start is not None:
            system_entities.add((entity_start, previous_token_end, current_entity[2]))
    return system_entities

def get_mitie_entities(original_tweets):
    system_entities = set()
    original_index = 0
    previous_token_end = 0
    ner = mitie.named_entity_extractor('MITIE/MITIE-models/english/ner_model.dat')

    original_tweets_clean = original_tweets.replace(u"’", "'")
    for tweet in original_tweets_clean.split("\n"):
        entity_start = None

        stripped_tweet = tweet.strip()
        if not stripped_tweet:
            continue

        tokens = mitie.tokenize(stripped_tweet)
        entities = ner.extract_entities(tokens)
        if entities:
            current_entity = entities.pop(0)
        else:
            current_entity = None
        for i, token in enumerate(tokens):
            unicode_token = token.decode('utf-8')
            original_index = original_tweets_clean.index(unicode_token, previous_token_end)
            if entity_start is not None and i == current_entity[0][-1] + 1:
                system_entities.add((entity_start, previous_token_end, current_entity[1]))
                entity_start = None
                if entities:
                    current_entity = entities.pop(0)
                else:
                    current_entity = None
            if current_entity is not None and i == current_entity[0][0]:
                entity_start = original_index
            previous_token_end = original_index + len(unicode_token)
        if entity_start is not None:
            system_entities.add((entity_start, previous_token_end, current_entity[1]))
    return system_entities

def get_nltk_entities(original_tweets):
    system_entities = set()
    original_index = 0
    previous_token_end = 0

    original_tweets_clean = original_tweets.encode('ascii', 'replace')
    for tweet in original_tweets_clean.split("\n"):
        entity_start = None

        stripped_tweet = tweet.strip()
        if not stripped_tweet:
            continue

        tagged_tweet = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(stripped_tweet)))
        for item in tagged_tweet:
            if type(item) == nltk.tree.Tree:
                entity_start = None
                for i, entity_piece in enumerate(item.leaves()):
                    token = entity_piece[0].replace("``", '"').replace("''", '"')
                    original_index = original_tweets_clean.index(token, previous_token_end)
                    if i == 0:
                        entity_start = original_index
                    if i == len(item.leaves()) - 1:
                        system_entities.add((entity_start, original_index + len(token), NLTK_TYPE_MAP[item.label()]))
                    previous_token_end = original_index + len(token)
            else:
                token = item[0].replace("``", '"').replace("''", '"')
                original_index = original_tweets_clean.index(token, previous_token_end)
                previous_token_end = original_index + len(token)
    return system_entities

def get_spacy_entities(original_tweets):
    nlp = spacy.load('en')
    system_entities = set()
    original_index = 0
    previous_token_end = 0

    for tweet in original_tweets.split("\n"):
        entity_start = None

        stripped_tweet = tweet.strip()
        if not stripped_tweet:
            continue

        doc = nlp(stripped_tweet)
        entities = [(e.start, e.end, SPACY_TYPE_MAP[e.label_]) \
                    for e in doc.ents \
                    if e.label_ in SPACY_TYPE_MAP]
        if entities:
            current_entity = entities.pop(0)
        else:
            current_entity = None
        for i, token_object in enumerate(doc):
            token = token_object.text
            original_index = original_tweets.index(token, previous_token_end)
            if entity_start is not None and i == current_entity[1]:
                system_entities.add((entity_start, previous_token_end, current_entity[2]))
                entity_start = None
                if entities:
                    current_entity = entities.pop(0)
                else:
                    current_entity = None
            if current_entity is not None and i == current_entity[0]:
                entity_start = original_index
            previous_token_end = original_index + len(token)
        if entity_start is not None:
            system_entities.add((entity_start, previous_token_end, current_entity[2]))
    return system_entities

def get_twitter_ner_features(sequences, dict_features, wv_model, gimple_brown_clusters,
                             test_enriched_data_brown_clusters,
                             test_enriched_data_clark_clusters):
    return [sent2features(sequence, vocab=None,
                          dict_features=dict_features, vocab_presence_only=False,
                          window=4, interactions=True, dict_interactions=True,
                          lowercase=False, dropout=0, word2vec_model=wv_model.model,
                          cluster_vocabs=[
                            gimple_brown_clusters,
                            test_enriched_data_brown_clusters,
                            test_enriched_data_clark_clusters
                          ])
            for sequence in sequences]

def get_twitter_ner_model(model_file_path, train_files,
                          dict_features, wv_model, gimple_brown_clusters,
                          test_enriched_data_brown_clusters,
                          test_enriched_data_clark_clusters):
    if os.path.exists(model_file_path):
        with open(model_file_path, "rb") as pickle_file:
            model = pickle.load(pickle_file)
    else:
        training_data = []
        for (train_file, encoding, type_map) in train_files:
            bieou_file = train_file + ".BIEOU.tsv"
            if not os.path.exists(bieou_file):
                sequences = load_sequences(train_file, sep="\t", encoding=encoding)
                write_sequences(sequences, bieou_file, to_bieou=True, type_map=type_map)
            training_data.extend(load_sequences(bieou_file))

        model = CRFModel()
        X_train = get_twitter_ner_features(training_data, dict_features, wv_model,
                                           gimple_brown_clusters,
                                           test_enriched_data_brown_clusters,
                                           test_enriched_data_clark_clusters)
        y_train = [sent2labels(sequence) for sequence in training_data]
        model.fit(X_train, y_train)
        with open(model_file_path, "wb") as pickle_file:
            pickle.dump(model, pickle_file)
    return model

def phrase_to_BIEOU(phrase):
    l = len(phrase)
    new_phrase = []
    for j, t in enumerate(phrase):
        new_tag = t.tag
        if l == 1:
            new_tag = "U%s" % t.tag[1:]
        elif j == l-1:
            new_tag = "E%s" % t.tag[1:]
        new_phrase.append(Tag(t.token, new_tag))
    return new_phrase

def to_BIEOU(seq, verbose=False):
    # TAGS B I E U O
    phrase = []
    new_seq = []
    for i, tag in enumerate(seq):
        if not phrase and tag.tag[0] == "B":
            phrase.append(tag)
            continue
        if tag.tag[0] == "I":
            phrase.append(tag)
            continue
        if phrase:
            if verbose:
                print("Editing phrase", phrase)
            new_phrase = phrase_to_BIEOU(phrase)
            new_seq.extend(new_phrase)
            phrase = []
        new_seq.append(tag)
    if phrase:
        if verbose:
            print("Editing phrase", phrase)
        new_phrase = phrase_to_BIEOU(phrase)
        new_seq.extend(new_phrase)
        phrase = []
    return new_seq

at_mention_re = re.compile(r"[@＠][a-zA-Z0-9_]+")

def write_sequences(sequences, filename, sep="\t", to_bieou=True, type_map=None):
    with io.open(filename, "w", encoding="utf8") as fp:
        for seq in sequences:
            if to_bieou:
                seq = to_BIEOU(seq)
            for tag in seq:
                new_tag = tag
                if type_map is not None and tag.tag[0] != "O":
                    new_label = "O"
                    if not at_mention_re.match(tag.token):
                        new_entity_type = type_map[tag.tag[2:]]
                        if new_entity_type is not None:
                            new_label = tag.tag[:2] + new_entity_type
                    new_tag = Tag(tag.token, new_label)
                fp.write(sep.join(new_tag) + u"\n")
            fp.write(u"\n")

def get_twitter_ner_entities(original_tweets, model_file_path, train_files):
    dict_features = DictionaryFeatures(DICTIONARY_DIR)
    all_sequences = load_sequences(DEV_FILE)
    for (train_file, encoding, type_map) in train_files:
        all_sequences.extend(load_sequences(train_file, sep="\t", encoding=encoding))
    all_tokens = [[t[0] for t in seq] for seq in all_sequences]
    if not os.path.exists(WORDVEC_FILE_PROCESSED):
        process_glovevectors(WORDVEC_FILE_RAW)
    wv_model = WordVectors(all_tokens, WORDVEC_FILE_PROCESSED)

    gimple_brown_cf = ClusterFeatures(GIMPLE_TWITTER_BROWN_CLUSTERS_DIR, cluster_type="brown")
    gimple_brown_cf.set_cluster_file_path(GIMPLE_TWITTER_BROWN_CLUSTERS_DIR)
    gimple_brown_clusters = gimple_brown_cf.read_clusters()

    for directory in (TEST_ENRICHED_DATA_BROWN_CLUSTER_DIR, TEST_ENRICHED_DATA_CLARK_CLUSTER_DIR):
        if not os.path.exists(directory):
            os.makedirs(directory)

    test_enriched_data_brown_cf = ClusterFeatures(TEST_ENRICHED_DATA_BROWN_CLUSTER_DIR,
                                                  cluster_type="brown", n_clusters=100)
    test_enriched_data_brown_cf.set_cluster_file_path()
    test_enriched_data_clark_cf = ClusterFeatures(TEST_ENRICHED_DATA_CLARK_CLUSTER_DIR,
                                                  cluster_type="clark", n_clusters=32)
    test_enriched_data_clark_cf.set_cluster_file_path()

    if not os.path.exists(test_enriched_data_brown_cf.cluster_file_path) or \
       not os.path.exists(test_enriched_data_clark_cf.cluster_file_path):
        preprocessed_tokens = [[preprocess_token(t[0], to_lower=True) for t in seq] 
                               for seq in all_sequences]

    if not os.path.exists(test_enriched_data_brown_cf.cluster_file_path):
        test_enriched_data_brown_cf.set_exec_path(BROWN_EXEC_PATH)
        test_enriched_data_brown_cf.gen_training_data(preprocessed_tokens, BROWN_INPUT_DATA_PATH)
        test_enriched_data_brown_cf.gen_clusters(BROWN_INPUT_DATA_PATH, TEST_ENRICHED_DATA_BROWN_CLUSTER_DIR)

    test_enriched_data_brown_clusters = test_enriched_data_brown_cf.read_clusters()

    if not os.path.exists(test_enriched_data_clark_cf.cluster_file_path):
        test_enriched_data_clark_cf.set_exec_path(CLARK_EXEC_PATH)
        test_enriched_data_clark_cf.gen_training_data(preprocessed_tokens, CLARK_INPUT_DATA_PATH)
        test_enriched_data_clark_cf.gen_clusters(CLARK_INPUT_DATA_PATH, TEST_ENRICHED_DATA_CLARK_CLUSTER_DIR)

    test_enriched_data_clark_clusters = test_enriched_data_clark_cf.read_clusters()

    model = get_twitter_ner_model(model_file_path, train_files,
                                  dict_features, wv_model, gimple_brown_clusters,
                                  test_enriched_data_brown_clusters,
                                  test_enriched_data_clark_clusters)

    test_data = load_sequences(TEST_FILE, sep="\t")
    predictions = model.predict(get_twitter_ner_features(test_data, dict_features,
                                                         wv_model, gimple_brown_clusters,
                                                         test_enriched_data_brown_clusters,
                                                         test_enriched_data_clark_clusters))

    system_entities = set()
    original_index = 0
    previous_token_end = 0
    previous_state = None
    for labeled_tokens, tweet_predictions in izip(test_data, predictions):
        entity_start = None
        for i in xrange(len(labeled_tokens)):
            token = labeled_tokens[i].token
            label = tweet_predictions[i]
            original_index = original_tweets.index(token, previous_token_end)
            state = label[0]
            if state in ("B", "U") or \
               (state in ("I", "E") and previous_state not in ("B", "I")):
                entity_start = original_index
            if state in ("E", "U") or \
               (state in ("B", "I") and (i == len(labeled_tokens) - 1 or tweet_predictions[i + 1][0] not in ("I", "E"))):
                entity_type = label[2:]
                if entity_type is not None:
                    system_entities.add((entity_start, original_index + len(token), entity_type))
                entity_start = None
            previous_token_end = original_index + len(token)
            previous_state = state
    return system_entities

def filter_by_type(entities, entity_type):
    return set([x for x in entities if x[2] == entity_type])

def write_all_scores(unmodified_csv_writer, no_mentions_csv_writer, system_name, gold_entities, system_entities):
    for csv_writer in (unmodified_csv_writer, no_mentions_csv_writer):
        corrected_gold_entities = gold_entities
        corrected_system_entities = system_entities
        if csv_writer == no_mentions_csv_writer:
            corrected_gold_entities = set([entity for entity in gold_entities if not at_mention_re.match(original_tweets[entity[0]:entity[1]])])
            corrected_system_entities = set([entity for entity in system_entities if not at_mention_re.match(original_tweets[entity[0]:entity[1]])])
        row = [system_name]
        write_scores(row, corrected_gold_entities, corrected_system_entities)
        for entity_type in GOOD_ENTITY_TYPES:
            write_scores(row,
                         filter_by_type(corrected_gold_entities, entity_type),
                         filter_by_type(corrected_system_entities, entity_type))
        csv_writer.writerow(row)

if __name__ == "__main__":
    (gold_entities, original_tweets) = get_gold_entities()
    header_row = ["System Name"] + ["%s %s" % (entity_type, metric) \
                                    for entity_type in ["Overall"] + GOOD_ENTITY_TYPES \
                                    for metric in ("Precision", "Recall", "F1 Score")]
    with open("evaluation.csv", "w") as evaluation_file:
        with open("evaluation_no_mentions.csv", "w") as evaluation_file_no_mentions:
            csv_writer = csv.writer(evaluation_file)
            no_mentions_csv_writer = csv.writer(evaluation_file_no_mentions)
            csv_writer.writerow(header_row)
            no_mentions_csv_writer.writerow(header_row)

            system_entities = get_osu_nlp_entities(original_tweets)
            write_all_scores(csv_writer, no_mentions_csv_writer, "OSU NLP", gold_entities, system_entities)

            system_entities = get_stanford_entities()
            write_all_scores(csv_writer, no_mentions_csv_writer, "Stanford", gold_entities, system_entities)

            system_entities = get_stanford_entities(truecase=True)
            write_all_scores(csv_writer, no_mentions_csv_writer, "Stanford (with truecasing)", gold_entities, system_entities)

            system_entities = get_stanford_entities(caseless=True)
            write_all_scores(csv_writer, no_mentions_csv_writer, "Stanford (with caseless models)", gold_entities, system_entities)

            system_entities = get_stanford_entities(twitter_pos=True)
            write_all_scores(csv_writer, no_mentions_csv_writer, "Stanford (with Twitter POS tagger)", gold_entities, system_entities)

            system_entities = get_polyglot_entities(original_tweets)
            write_all_scores(csv_writer, no_mentions_csv_writer, "Polyglot", gold_entities, system_entities)

            system_entities = get_spacy_entities(original_tweets)
            write_all_scores(csv_writer, no_mentions_csv_writer, "spaCy", gold_entities, system_entities)

            system_entities = get_mitie_entities(original_tweets)
            write_all_scores(csv_writer, no_mentions_csv_writer, "MITIE", gold_entities, system_entities)

            system_entities = get_nltk_entities(original_tweets)
            write_all_scores(csv_writer, no_mentions_csv_writer, "NLTK", gold_entities, system_entities)

            twitter_ner_entities = get_twitter_ner_entities(original_tweets, TWITTER_NER_MODEL_FILE, [(TRAIN_FILE, "utf-8", TYPE_MAP)])
            write_all_scores(csv_writer, no_mentions_csv_writer, "TwitterNER", gold_entities, twitter_ner_entities)

            system_entities = get_twitter_ner_entities(original_tweets,
                                                       TWITTER_NER_WNUT_TRAINING_DATA_MODEL_FILE,
                                                       [(TRAIN_FILE, "utf-8", TYPE_MAP),
                                                        (WNUT_TRAIN_FILE, "utf-8", WNUT_TYPE_MAP)])
            write_all_scores(csv_writer, no_mentions_csv_writer, "TwitterNER (with W-NUT 2017 training data)", gold_entities, system_entities)

            system_entities = get_twitter_ner_entities(original_tweets,
                                                       TWITTER_NER_HEGE_TRAINING_DATA_MODEL_FILE,
                                                       [(TRAIN_FILE, "utf-8", TYPE_MAP),
                                                        (HEGE_TRAIN_FILE, "utf-8", FININ_TYPE_MAP)])
            write_all_scores(csv_writer, no_mentions_csv_writer, "TwitterNER (with Hege training data)", gold_entities, system_entities)

            system_entities = get_twitter_ner_entities(original_tweets,
                                                       TWITTER_NER_FININ_TRAINING_DATA_MODEL_FILE,
                                                       [(TRAIN_FILE, "utf-8", TYPE_MAP),
                                                        (FININ_TRAIN_FILE, "cp1252", FININ_TYPE_MAP),
                                                        (FININ_TEST_FILE, "cp1252", FININ_TYPE_MAP)])
            write_all_scores(csv_writer, no_mentions_csv_writer, "TwitterNER (with Finin training data)", gold_entities, system_entities)

            system_entities = get_twitter_ner_entities(original_tweets,
                                                       TWITTER_NER_WNUT_AND_HEGE_TRAINING_DATA_MODEL_FILE,
                                                       [(TRAIN_FILE, "utf-8", TYPE_MAP),
                                                        (WNUT_TRAIN_FILE, "utf-8", WNUT_TYPE_MAP),
                                                        (HEGE_TRAIN_FILE, "utf-8", FININ_TYPE_MAP)])
            write_all_scores(csv_writer, no_mentions_csv_writer, "TwitterNER (with W-NUT 2017 and Hege training data)", gold_entities, system_entities)