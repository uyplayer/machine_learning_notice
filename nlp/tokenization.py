#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/25 17:47
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : tokenization.py
# @Software: PyCharm

# dependency library
# system
import re
# nlp
from nltk.tokenize import word_tokenize


text = "مالىيە كۈچى توكيو دۇنيادىكى داڭلىق پۇل-مۇئامىلە مەركەزلىرىنىڭ بىرى ، نيۇ-يورك ۋە لوندون خەلقئارا پۇل-مۇئامىلە مەركەزلىرىنىڭ تەرەققىيات كۈچى جەھەتتە ئالدىنقى ئۈچ ئورۇندا تۇرىدۇ. ئۇنىڭ ئوچۇقلىقى ، پۇل-مۇئامىلە مۇلازىمىتى سەۋىيىسى ، پۇل-مۇئامىلە يېڭىلىق يارىتىش ئىقتىدارى ۋە باشقا تەرەپلىرى دۇنيانىڭ ئالدىنقى قاتارىدا.توكيو خەلقئارالىق سودا مەركىزى بولۇش سۈپىتى بىلەن 2300 دىن ئارتۇق چەتئەل مەبلىغى بىلەن تەمىنلىگەن شىركەتنى يىغىپ ، بۇ دۆلەتنىڭ تەخمىنەن% 76 گە تەڭ كېلىدۇ. بايلىق يەرشارى 500 نىڭ باش ئىش بېجىرىش ئورنى دۇنيادىكى ئالدىنقى قاتاردا تۇرىدۇ. بۇ نۇرغۇنلىغان كارخانا توپى يېڭى سانائەت توپىنى بارلىققا كەلتۈرۈپ ، توكيونىڭ ئىقتىسادىي ھاياتىي كۈچىنى ئىلگىرى سۈردى."

# split() function
print(text.split())

#  Regular Expressions (RegEx)
tokens = re.findall("[\w']+", text)
print(tokens)

# Sentence Tokenization
sentences = re.compile('[.!?] ').split(text)
print(sentences)

# Tokenization using NLTK
# Word Tokenization
# word_tokenize(text)
# print(word_tokenize(text))