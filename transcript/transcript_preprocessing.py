#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

lexicon_file1 = "./lexicons/Ratings_Warriner_et_al.csv"
df1 = pd.read_csv(lexicon_file1)

lexicon_file2 = "./lexicons/DepecheMood_english_token_full.tsv"
df2 = pd.read_csv(lexicon_file2, delimiter="\t")

column_names_lex1 = ["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]
column_names_lex2 = ["AFRAID", "AMUSED", "ANGRY", "ANNOYED", "DONT_CARE", "HAPPY", "INSPIRED", "SAD"]

out_dir = "../data/text/lexicons_features/"

for su in range(1,11):
    for st in range(1,11):
        emotions_story = []
        l1 = [0]*(len(column_names_lex1))
        l2 = [0]*(len(column_names_lex2))
        l1_prev = l1.copy()
        l2_prev = l2.copy()

        story_file = story_file = "Subject_"+str(su)+"_Story_"+str(st)

        df_story = pd.read_csv("../data/text/word_valence/"+story_file+".tsv", delimiter="\t", header=None)
        list_gold_valence = df_story[1].tolist()

        for _, row in df_story.iterrows():
            word = row[0]

            # lexicon 1
            q = df1.loc[df1["Word"] == word]
            if len(q) == 1: 
                l1 = [float(q[x]) for x in column_names_lex1]
                l1_prev = l1.copy()
            else: 
                l1 = l1_prev.copy()
                if len(l1) > 3:
                    print("l1:",l1)

            # lexicon 2
            q = df2.loc[df2["Unnamed: 0"] == word]
            if len(q) == 1: 
                l2 = [float(q[x]) for x in column_names_lex2]
                l2_prev = l2.copy()
            else: 
                l2 = l2_prev.copy()
                if len(l2) > 8:
                    print("l2:",l2)

            l_tot = [str(x) for x in l1+l2]
            emotions_story.append(",".join(l_tot))
            if len(l_tot) >= 12:
                print(l_tot)

        out_filename = out_dir + story_file + "_lex.csv"
        with open(out_filename, 'w') as out:
            out.write("\n".join(emotions_story))
            print(out_filename)