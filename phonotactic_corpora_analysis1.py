#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Saliha Muradoglu
"""
import re
import numpy as np
import itertools
import pandas as pd
from legality_principle import LegalitySyllableTokenizer
from legality_principle_gbb import LegalitySyllableTokenizer_gbb
import matplotlib.pyplot as plt
import seaborn as sns
#%%#
def extract(file, lang):
    """
    file: dictionary file name without '.txt' extention
    lang: 3-letter iso code for language. E.g., 'ptj' for Pitjantjatjara. 
    """
    language = lang
    valid = {'ptj', 'wbp','wrm'}
    if lang not in valid:
        raise ValueError("Please use one of: 'ptj' for Pitjantjatjara, 'wrm':Warumungu or 'wbp for Walpiri." % valid)
    file_dump = open(file, 'r') 
    file_lines= file_dump.readlines()
    if language == 'wrm':
        w = ' \n'
    else:
        w = '\n'
    lexical_grp = [list(y) for x, y in itertools.groupby(file_lines, lambda z: z == w) if not x]
    return lexical_grp

def ptj_clean(lexical_list):    
    headword =[]
    pos=[]
    gloss=[]
    #Go over each lexeme
    for i in range(len(lexical_list)):
        lexeme = lexical_list[i]
        if [k for k in lexeme if '\lx ' in k] !=[]:
            l = [k for k in lexeme if '\lx ' in k]
            lx_clean = l[0].split('\lx ')[-1].split('\n')[0]
            headword.append(lx_clean)
        else: 
            headword.append('-')
        if [l for l in lexeme if '\ps ' in l] !=[]:
            p = [l for l in lexeme if '\ps ' in l]
            ps_clean = p[0].split('\ps ')[-1].split('\n')[0]
            pos.append(ps_clean)
        else: 
            pos.append('-')
        if [m for m in lexeme if '\de ' in m] !=[]:
            d = [m for m in lexeme if '\de ' in m]

            de_clean = d[0].split('\de ')[-1].split('\n')[0]
            gloss.append(de_clean)
        else: 
            gloss.append('-')    

    return headword, pos, gloss    

def wrm_clean(lexical_list):
    headword =[]
    pos=[]
    gloss=[]
    #Go over each lexeme
    for i in range(len(lexical_list)):
        lexeme = lexical_list[i]
        if [k for k in lexeme if '\\w ' in k] !=[]:
            l = [k for k in lexeme if '\\w ' in k]
            lx_clean = l[0].split('\\w ')[-1].split('\n')[0]
            headword.append(lx_clean)
        else: 
            headword.append('-')
        if [l for l in lexeme if '\\p ' in l] !=[]:
            p = [l for l in lexeme if '\\p ' in l]
            ps_clean = p[0].split('\\p ')[-1].split('\n')[0]
            pos.append(ps_clean)
        else: 
            pos.append('-')
        if [m for m in lexeme if '\\d ' in m] !=[]:
            d = [m for m in lexeme if '\\d ' in m]
            de_clean = d[0].split('\\d ')[-1].split('\n')[0]
            gloss.append(de_clean)
        else: 
            gloss.append('-')    
    return headword, pos, gloss    

def wbp_clean(lexical_list):
    me =[]
    pos=[]
    headword=[]
    gloss=[]
    #Go over each lexeme
    for i in range(len(lexical_list)):
        lexeme = lexical_list[i]
        if [k for k in lexeme if '\\me ' in k] !=[]:
            l = [k for k in lexeme if '\\me ' in k]
            lx_clean = l[0].split('\\me ')[-1].split('\n')[0]
            me.append(lx_clean)
        else: 
            me.append('-')
        if [l for l in lexeme if '\gl' in l] !=[]:
            d = [l for l in lexeme if '\gl' in l]
            de_clean = d[0].split('\gl ')[-1].split('\egl')[0]
            de_strip = de_clean.replace('\glo','')
            de_strip1 = de_strip.split('\[')[0]
            de = de_strip1.translate({ord(c): None for c in '@^""'})
            gloss.append(de)
        else: 
            gloss.append('-')
    for m in range(len(me)):
        if '*' in me[m]:
            headword.append(me[m].split('*')[0])
        else:
            headword.append(me[m].split(' (')[0])
        pos.append(me[m].split(' (')[1].split('):')[0])
    return headword, pos, gloss    


def gbb_import(lang):
    df = pd.read_csv(lang)
    return df

def vowel_count(headwords):
    #Count number of vowels
    countable=[]
    for k in range(len(headwords)):
    # initializing replace mapping 
        word = headwords[k]
        for items, initial in {'aa' : 'A', 'ii' : 'I','uu':'U'}.items():
            word = word.replace(items, initial)
        countable.append(word)
    count = []
    for h in range(len(countable)):
        count.append(countable[h].count('a')+countable[h].count('u')+countable[h].count('i')+countable[h].count('A')+countable[h].count('I')+countable[h].count('U')+countable[h].count('U'))
    return count, countable


def gbb_vowel_count(headword_ipa):
    #Count number of vowels
    count = []
    for h in range(len(headword_ipa)):
        count.append(headword_ipa[h].count('u')+headword_ipa[h].count('i')+headword_ipa[h].count('ɐ')+headword_ipa[h].count('ə'))
    return count

def orth2ipa(lang,headword):
    language = lang
    if language == 'ptj':
        ipas = {'ṯ':'ʈ', 'ṉ':'ɳ','ḻ':'ɭ','tj':'c','ny':'ɲ','ng':'ŋ','ly':'ʎ','ṟ':'ɻ','y':'j','au':'awu','ai':'ayi'}
    elif language == 'wrm':
        ipas = {'rr':'ɾ','rt':'ʈ','rn':'ɳ','rl':'ɭ','j':'c','ny':'ɲ','ng':'ŋ','ly':'ʎ'}
    elif language == 'wbp':
        ipas ={'rr':'ɾ', 'rt':'ʈ','rn':'ɳ','rl':'ɭ','rd':'ɽ','j':'c','ny':'ɲ','ly':'ʎ','ng':'ŋ'}
    IPA=[]
    for g in range(len(headword)):
    # initializing replace mapping 
        item = headword[g].lower()
        for word, ipa in ipas.items():
            item =item.replace(word, ipa)
        IPA.append(item)
    return IPA

def wbp_verb(headword, pos):
    npst = {'-mi':'', '-rni':'','-ni':'','-nyi':''}
    root = []
    for i in range(len(headword)):
        item = headword[i]
        if pos[i] == 'V':
            for word, non in npst.items():
                item = item.replace(word,non)
            root.append(item)
        else:
            root.append(headword[i])
    return root   

def ptj_vcompound(headword, pos):
    root_pv=[]
    lightverbs = ['katinyi','mananyi','rungkaṉi','nyanganyi','-mananyi','tjunanyi','tjingaṉi','punganyi','pupanyi','nyinanyi','ngaṟanyi','wakaṉi','pakaṉi','kulini','ngaṟanyi']  
    for i in range(len(headword)):
        for ii in range(len(lightverbs)):
            if lightverbs[ii] in headword[i] and len(headword[i]) > len(lightverbs[ii]):
                root_pv.append(headword[i].replace(lightverbs[ii],''))
                break
            else:
                root_pv.append(headword[i])
                break
    return root_pv

def ptj_verb(headword, pos):
    prs = {'-ni':'','-nyi':'', '-ṉi':''}
    root = []
    for i in range(len(headword)):
        item = headword[i]
        if pos[i] == 'verb':
            for word, non in prs.items():
                item = item.replace(word,non,1)
            root.append(item)

        else:
            root.append(headword[i])
        
    return root 

def ptj_multiple_replace(text):
    tense = {'ni':'-ni','nyi':'-nyi', 'ṉi':'-ṉi'}
    # Create a regular expression from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, tense.keys())) + r'$')
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: tense[mo.group()], text,1) 


def ptj_verb_segment(headword, pos):
    segment = []
    for i in range(len(headword)):
        item = headword[i]
        if pos[i] == 'verb':
            item = ptj_multiple_replace(item)
            segment.append(item)
        else:
            segment.append(headword[i])
        
    return segment 

def lightverb_replace(text):
    light = {'katinyi': '','mananyi':'','tjingaṉi':'','rungkaṉi':'','nyanganyi':'','-mananyi':'','tjunanyi':'','punganyi':'','pupanyi':'','nyinanyi':'','ngaṟanyi':'','wakaṉi':'','pakaṉi':'','kulini':'','ngaṟanyi':''}
    # Create a regular expression from the dictionary keys
    regex = re.compile('(%s)' % '|'.join(map(re.escape, light.keys())) + r'$')
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: light[mo.group()], text,1) 

def lightverb_space(headword):
    light_verbs=[]
    for i in range(len(headword)):
        light_verb = headword[i]
        # add a space between preverb and verb
        light_verb = light_verb.replace('mananyi',' mananyi')
        light_verb = light_verb.replace('rungkaṉi',' rungkaṉi')
        light_verb = light_verb.replace('nyanganyi',' nyanganyi')
        light_verb = light_verb.replace('-mananyi',' mananyi')
        light_verb = light_verb.replace('tjunanyi',' tjunanyi')
        light_verb = light_verb.replace('punganyi',' punganyi')
        light_verb = light_verb.replace('tjingaṉi',' tjingaṉi')
        light_verb = light_verb.replace('pupanyi',' pupanyi')
        light_verb = light_verb.replace('nyinanyi',' nyinanyi')
        light_verb = light_verb.replace('ngaṟanyi',' ngaṟanyi')
        light_verb = light_verb.replace('wakaṉi',' wakaṉi')
        light_verb = light_verb.replace('pakaṉi',' pakaṉi')
        light_verb = light_verb.replace('kulini',' kulini')
        light_verb = light_verb.replace('katinyi',' katinyi')
        light_verbs.append(light_verb)
    return  light_verbs

def lightverb_replace1(text):
    light1 = {'tjingaṉi':'','katinyi':'','mananyi':'','rungkaṉi':'','nyanganyi':'','-mananyi':'','tjunanyi':'','punganyi':'','pupanyi':'','nyinanyi':'','ngaṟanyi':'','wakaṉi':'','pakaṉi':'','kulini':'','ngaṟanyi':''}
    # Create a regular expression from the dictionary keys\
    inter=[]
    if len((re.split('(%s)' % '|'.join(map(re.escape, light1.keys())) + r'$',text)))> 2:
        inter.append(re.split('(%s)' % '|'.join(map(re.escape, light1.keys())) + r'$',text)[0])
        inter.append(re.split('(%s)' % '|'.join(map(re.escape, light1.keys())) + r'$',text)[1])
    else:
        pass
    return inter

def ptj_verb_comp(headword):
    root = []
    for i in range(len(headword)):
        item = headword[i]
        item = lightverb_replace(item)
        if len(item) > 0:
            root.append(item)
        else:
            root.append(headword[i])
    return root 

def ptj_pvv(headword, pos, gloss, trans):
    ptj_pvv_me =[] 
    ptj_pvv_pos=[]
    ptj_pvv_gl=[]
    ptj_pvv_pv=[]
    ptj_pvv_v=[]
    ptj_pvv_trans=[]
    for i in range(len(headword)):
        if pos[i] == 'verb':
            item = headword[i]
            interim = lightverb_replace1(item)
            if len(interim) > 0:
                if len(interim[0]) > 0:
                    ptj_pvv_me.append(headword[i])
                    ptj_pvv_pos.append(pos[i])
                    ptj_pvv_gl.append(gloss[i])
                    ptj_pvv_trans.append(trans[i])
                    ptj_pvv_pv.append(interim[0])
                    ptj_pvv_v.append(interim[1])
                else:
                    pass
            else:
                pass
    return ptj_pvv_me, ptj_pvv_pos, ptj_pvv_gl, ptj_pvv_trans, ptj_pvv_pv, ptj_pvv_v
def segment(headword):
    """
    Segment words. Assumes that the words are segemented with '-', 
    that headwords are already stripped of other inflectional values.
    """
    #Split word into segments
    seg1=[]
    seg2=[]
    for g in range(len(headword)):
        word = headword[g]
        segment = word.split('-')
        seg1.append(segment[0])
        if len(segment)>1:
            seg2.append(segment[1])
        else:
            seg2.append('-') 
    return seg1, seg2

def vcompound(headword, seg1, seg2):
    vmin=[]
    for i in range(len(headword)):
        if seg2[i] in headword:
            vmin.append(seg1[i])
        else:
            vmin.append(headword[i])
    return vmin


def redup(headword, seg1, seg2):
    """
    Find words with reduplications. Assumes that the words are segemented with '-', 
    that headwords are already stripped of other inflectional values.
    """
    #evaluate where word entails reduplication
    root =[]
    for i in range(len(headword)):
        if seg1[i] == seg2[i]:
            root.append(seg1[i])
        else:
            root.append(headword[i])
    return root

def syllable(ipa_list):
    syllable =[]
    unique =[]     
    LP = LegalitySyllableTokenizer(ipa_list)
    for i in range(len(ipa_list)):
        b=LP.tokenize(ipa_list[i])
        syllable.append(str(b))
        unique.extend(b)
    return syllable, unique

def syllable_gbb(ipa_list):
    syllable =[]
    unique =[]     
    LP = LegalitySyllableTokenizer_gbb(ipa_list)
    for i in range(len(ipa_list)):
        b=LP.tokenize(ipa_list[i])
        syllable.append(str(b))
        unique.extend(b)
    return syllable, unique
        
def stichframe(headword, IPA, syll_clean, pos, gloss, count,syll_pred,seg1,seg2):
    gf = pd.DataFrame(list(zip(headword, IPA,syll_clean, pos, gloss, count,syll_pred,seg1,seg2)), columns =['headword', 'IPA', 'OS', 'pos', 'gloss', 'syllable_count','syllable','const1','const2'])
    df = gf.drop(gf[gf['headword'] == '-'].index)
    df = df[df["headword"].str.startswith("-") == False]
    #df.to_csv(language+'_clean',index=False, header=True)
    return df
def prestichframe(headword, IPA,syll_clean, pos, gloss, count,syll_pred,seg1,seg2):
    gf = pd.DataFrame(list(zip(headword, IPA,syll_clean, pos, gloss, count,syll_pred,seg1,seg2)), columns =['headword', 'IPA', 'OS','pos', 'gloss', 'syllable_count','syllable','const1','const2'])
    df = gf.drop(gf[gf['headword'] == ''].index)
    df = df[df["headword"].str.startswith("") == False]
    #df.to_csv(language+'_clean',index=False, header=True)
    return gf    
    
def stichframe1(headword, segment, IPA,syll_clean, pos, gloss, count,syll_pred,seg1,seg2):
    gf = pd.DataFrame(list(zip(headword, segment, IPA, pos, gloss, count,syll_pred,seg1,seg2)), columns =['headword', 'verb segment', 'IPA', 'OS', 'pos', 'gloss', 'syllable_count','syllable','const1','const2'])
    df = gf.drop(gf[gf['headword'] == '-'].index)
    df = df[df["headword"].str.startswith("-") == False]
    #df.to_csv(language+'_clean',index=False, header=True)
    return df


 
def pos_filter(df,condition):
    string= condition
    df = df[df["pos"].eq(string) == True]
    return df

def Npos_filter(df,condition):
    string= condition
    df = df[~df["pos"].eq(string) == True]
    return df

def head_filter(df,condition):
    """The entered condition can include regex expressions. 
    For example '^y' for words beginning with 'y' or 'i$' for words ending in 'i'.
    """
    string= condition
    df = df[~df["headword"].str.contains(string, regex= True)]
    return df

def remove_items(test_list, item): 
    rem_list=[]
    for i in range(len(test_list)):
        rem_list.append(test_list[i].replace(item,''))
    return rem_list

def redup1(df):
    """
    Find words with reduplications. Assumes that the words are segemented with '-', 
    that headwords are already stripped of other inflectional values.
    """
    #evaluate where word entails reduplication
    condition1 = df['seg2'].eq(df['seg1'])
    gf = df.loc[~condition1]
    #Check if first segment exisits as a main-entry
    condition2 = df['seg1'].isin(df['headword'])
    qf = df.loc[condition1 & ~condition2]
    #combine non-reduplicate forms with unique-reduplications
    ff = gf.append(qf)
    return ff

def write_file(lang,headword,pos, gloss):
    gf = pd.DataFrame(list(zip(headword, pos, gloss)), columns =['headword','pos', 'gloss'])
    gf.to_csv(lang+'_clean',index=False, header=True)
 
#%%# Analysis functions
def clean_syllables(df):
    syll = df['syllable'].tolist()
    sy=[]
    for x in syll:
        x= x.replace("[", '').replace("]", '').replace("\'", '')
        sy.append([x])
    return sy   
    
def count_syllables(df):
    syll = df['syllable'].tolist()
    sy_len =[]
    sy=[]
    for x in syll:
        x= x.replace("[", '').replace("]", '').replace("\'", '')
        x = x.split(',')
        sy.append(x)
        sy_len.append(len(x))
    return sy, sy_len  
    
def plot_syllables(df, file_name):
    sy = df['syllable'].tolist()
    sy_len =[]
    for x in sy:
        sy_len.append(len(x))
    res = np.array(sy_len)
    unique, counts = np.unique(res, return_counts=True)
    sns.set()
    ax = sns.barplot(x=unique,y=counts, palette ="deep")
    ax.set(xlabel='Number of syllables', ylabel='Number of words')
    plt.savefig(file_name+'.png', dpi=600)
    return sy, sy_len

def count_syllable_position(syllable_list, syllable_length):        
    #maximum syllable length observed in dataset
    N = max(syllable_length)
    #create lists for each syllable length
    x= [ [] for _ in range(N) ]
    #Collect each syllable across each position, across all words
    for i in range(len(syllable_list)):
        item = syllable_list[i]
        j=0
        while j <= N:    
            if len(item) > j:
                x[j].append(item[j])
            else:
                pass    
            j +=1
    return x


def vowel_distribution(x):
    # Vowel distribution across syllable positions 
    #create lists for each syllable length
    xa= [ [] for _ in range(len(x)) ]
    #create lists for each syllable length
    xi= [ [] for _ in range(len(x)) ]
    #create lists for each syllable length
    xu= [ [] for _ in range(len(x)) ]
    #count 'aiu' vowels in each syllable position
    for k in range(len(x)):
        syllable = x[k]
        xa[k].append(sum(map(lambda x: 'a' in x, syllable)))
        xi[k].append(sum(map(lambda x: 'i' in x, syllable)))
        xu[k].append(sum(map(lambda x: 'u' in x, syllable)))
    syll= list(range(1,len(xu)+1))
    sy = pd.DataFrame(syll, columns =['Syllable'])
    Aframe = pd.DataFrame(xa, columns =['a'])
    Iframe = pd.DataFrame(xi, columns =['i'])
    Uframe = pd.DataFrame(xu, columns =['u'])
    frame = pd.concat([sy,Aframe, Iframe, Uframe], axis=1)
  
    return frame

def gbb_vowel_distribution(x):
    # Vowel distribution across syllable positions 
    #create lists for each syllable length
    xe= [ [] for _ in range(len(x)) ]
    #create lists for each syllable length
    xa= [ [] for _ in range(len(x)) ]
    #create lists for each syllable length
    xi= [ [] for _ in range(len(x)) ]
    #create lists for each syllable length
    xu= [ [] for _ in range(len(x)) ]
    #count 'aiu' vowels in each syllable position
    for k in range(len(x)):
        syllable = x[k]
        xa[k].append(sum(map(lambda x: 'ɐ' in x, syllable)))
        xi[k].append(sum(map(lambda x: 'i' in x, syllable)))
        xu[k].append(sum(map(lambda x: 'u' in x, syllable)))
        xe[k].append(sum(map(lambda x: 'ə' in x, syllable)))
    syll= list(range(1,len(xu)+1))
    sy = pd.DataFrame(syll, columns =['Syllable'])
    Aframe = pd.DataFrame(xa, columns =['ɐ'])
    Iframe = pd.DataFrame(xi, columns =['i'])
    Uframe = pd.DataFrame(xu, columns =['u'])
    Eframe = pd.DataFrame(xe, columns =['ə'])
    frame = pd.concat([sy,Aframe, Iframe, Uframe, Eframe], axis=1)
    return frame

def syllable_matrix(syllable_list):
    syl_matrix = pd.DataFrame(syllable_list).fillna('').add_prefix('sy')
    return syl_matrix

def find_vowel(string):
    match=re.findall(r'[aiuəɐ]', string)
    if len(match) ==1:
        return match[0]
    else:
        return ''
    
def find_consonant(string):
    match=re.findall('(?=[aiu])(.*)',string)
    if len(match) ==1:
        return match[0]
    else:
        return ''
    
def vowel_matrix(sy_matrix):
    vowel_matrix=sy_matrix.applymap(find_vowel)
    return vowel_matrix

def consonant_matrix(sy_matrix):
    consonant_matrix=sy_matrix.applymap(find_consonant)
    return consonant_matrix


def poa_distribution(consonant_matrix, vowel):
    # Generate all posible vowel+consonant combinations
    labial  = ['p','m','w']
    alvelor = ['t','n','l','r','ɾ']
    retroflex = ['ʈ','ɳ','ɭ	','ɽ']	
    palatal = ['c','ɲ','ʎ','y']
    velar = ['k','ŋ']
    vowel_labial=  [vowel+ s for s in labial]
    vowel_alvelor=  [vowel+ s for s in alvelor]
    vowel_retroflex=  [vowel+ s for s in retroflex]
    vowel_palatal=  [vowel+ s for s in palatal]
    vowel_velar=  [vowel+ s for s in velar]
    #create lists for each syllable length
    vc_lab = consonant_matrix.isin(vowel_labial).sum()
    vc_alv = consonant_matrix.isin(vowel_alvelor).sum()
    vc_ret = consonant_matrix.isin(vowel_retroflex).sum()
    vc_pal = consonant_matrix.isin(vowel_palatal).sum()
    vc_vel = consonant_matrix.isin(vowel_velar).sum()
    return vc_lab, vc_alv, vc_ret, vc_pal, vc_vel

def word_length(lang_syllables, lang_syllable_len):
    #find max word length possible
    length = max(lang_syllable_len)
    #create lists for each word length
    word_length= [ [] for _ in range(length)]
    #Collect words of same length together
    for k in range(len(lang_syllables)):
        word = lang_syllables[k]
        for j in range(length):
            if len(word) -1  == j:
                word_length[j].append(word)
            else:
                pass
    return word_length

def count_vowels(wbp_word):
    #create lists for each syllable length
    wfa= [ [] for _ in range(len(wbp_word)) ]
    #create lists for each syllable length
    wfi= [ [] for _ in range(len(wbp_word)) ]
    #create lists for each syllable length
    wfu= [ [] for _ in range(len(wbp_word)) ]
    for k in range(len(wbp_word)):
        num_words = len(wbp_word[k])
        for i in range(num_words):
            word = wbp_word[k][i][k]
            wfa[k].append(word.count('a'))
            wfi[k].append(word.count('i'))
            wfu[k].append(word.count('u'))
    a_count=[]
    i_count=[]
    u_count=[]
    syll= list(range(1,len(wfa)+1))
    for j in range(len(wfa)):
        a_count.append(wfa[j].count(1))
        i_count.append(wfi[j].count(1))
        u_count.append(wfu[j].count(1))
    wf_frame = pd.DataFrame(list(zip(syll,a_count, i_count,u_count)), columns=['Syllable','a','i','u'])
    return wf_frame


def gbb_count_vowels(gbb_word):
    #create lists for each syllable length
    wfa= [ [] for _ in range(len(gbb_word)) ]
    #create lists for each syllable length
    wfi= [ [] for _ in range(len(gbb_word)) ]
    #create lists for each syllable length
    wfu= [ [] for _ in range(len(gbb_word)) ]
    #create lists for each syllable length
    wfe= [ [] for _ in range(len(gbb_word)) ]
    for k in range(len(gbb_word)):
        num_words = len(gbb_word[k])
        for i in range(num_words):
            word = gbb_word[k][i][k]
            wfa[k].append(word.count('ɐ'))
            wfi[k].append(word.count('i'))
            wfu[k].append(word.count('u'))
            wfe[k].append(word.count('ə'))
    a_count=[]
    i_count=[]
    u_count=[]
    e_count=[]
    syll= list(range(1,len(wfa)+1))
    for j in range(len(wfa)):
        a_count.append(wfa[j].count(1))
        i_count.append(wfi[j].count(1))
        u_count.append(wfu[j].count(1))
        e_count.append(wfe[j].count(1))
    wf_frame = pd.DataFrame(list(zip(syll,a_count, i_count,u_count,e_count)), columns=['Syllable','ɐ','i','u','ə'])
    return wf_frame


def word_template(lang,headword):
    language = lang
    if language == 'Pitjantjatjara':
        vowel = {'a':'V','i':'V','u':'V'}
        consonant = {'ʈ':'C', 'ɳ':'C','ɭ':'C','c':'C','ɲ':'C','ŋ':'C','ʎ':'C','ɻ':'C','j':'C','p':'C','m':'C','w':'C','t':'C','n':'C','l':'C','r':'C','k':'C'}
    elif language == 'Warumungu':
        vowel = {'a':'V','i':'V','u':'V'}
        consonant = {'p':'C','m':'C','w':'C','t':'C','n':'C','l':'C','r':'C','ɾ':'C','ʈ':'C','ɳ':'C','ɭ':'C','c':'C','ɲ':'C','ʎ':'C','y':'C','k':'C','ŋ':'C'}
    elif language == 'Warlpiri':
        vowel = {'a':'V','i':'V','u':'V'}
        consonant = {'p':'C','m':'C','w':'C','t':'C','n':'C','l':'C','r':'C','ɾ':'C','ʈ':'C','ɳ':'C','ɭ':'C','ɽ':'C','c':'C','ɲ':'C','ʎ':'C','y':'C','k':'C','ŋ':'C'}
    elif language == 'Kaytetye':
        vowel = {'ɐ':'V','i':'V','u':'V','ə':'V'}
        consonant = {'p':'C','j':'C','ɰ':'C','m':'C','ḻ':'C','ṉ':'C','ṯ':'C','w':'C','t':'C','n':'C','l':'C','r':'C','ɹ':'C','ʈ':'C','ɳ':'C','ɭ':'C','c':'C','ɲ':'C','ʎ':'C','y':'C','k':'C','ŋ':'C'}
    word_template = []     
    for g in range(len(headword)):
    # initializing replace mapping 
        item = headword[g].lower()
        for word, C in consonant.items():
            item =item.replace(word, C)
        for word, V in vowel.items():
            item =item.replace(word, V)
        word_template.append(item)                                                                         
    return word_template

def VC_clusters(headword, word_template):
    onset=[]
    coda=[]
    #find all VC clusters
    for i in range(len(word_template)):
        v =[m.start() for m in re.finditer('VC', word_template[i])] 
        #sort VC cluters into coda/onset
        for j in range(len(v)):
            num = v[j]
            if word_template[i][num:num+3] == 'VCV':
                onset.append((headword[i][num:num+2]))
            else: 
                coda.append((headword[i][num:num+2]))
    return onset, coda

def poa_labeller(VC):
    poa=[]
    labial  = ['p','m','w']
    alvelor = ['t','n','l','r','ɾ','ɹ']
    retroflex = ['ʈ','ɳ','ɭ','ɽ','ḻ','ṉ','ṯ']	
    palatal = ['c','ɲ','ʎ','y','j']
    velar = ['k','ŋ','ɰ']
    for k in range(len(VC)):
        if any(n in VC[k] for n in labial):
            poa.append('labial')
        elif any(n in VC[k] for n in alvelor):
            poa.append('alvelor')
        elif any(n in VC[k] for n in retroflex):
            poa.append('retroflex')
        elif any(n in VC[k] for n in palatal):
            poa.append('palatal')
        elif any(n in VC[k] for n in velar):
            poa.append('velar')
    return poa

def VC_spliter(VC):
    vlist = []
    clist =[]
    for i in range(len(VC)):
        vlist.append(VC[i][0:1])
        clist.append(VC[i][1:2])
    return vlist,clist

def poa_dist(onset,coda,onset_poa,coda_poa):
    onset_lab = list(len(onset)*['onset'])
    coda_lab = list(len(coda)*['coda'])
    df = pd.DataFrame(onset, columns=['VC'])
    df[['V']] = df['VC'].str.slice(0, 1)
    df[['C']] = df['VC'].str.slice(1, 2)
    df[['placement']] = onset_lab
    df[['poa']] = onset_poa
    gf = pd.DataFrame(coda, columns=['VC'])
    gf[['V']] = gf['VC'].str.slice(0, 1)
    gf[['C']] = gf['VC'].str.slice(1, 2)
    gf[['placement']] = coda_lab
    gf[['poa']] = coda_poa
    ff = pd.concat([df,gf])
    return ff    
def drop_dup(frame):
    frame = frame.loc[frame.astype(str).drop_duplicates(subset=['syllable']).index]
    return frame


def drop_Eng(frame):
    frame = frame[frame["gloss"].str.contains("from English")==False]
    return frame



#%%#
#PTJ EXAMPLE
def ptj_augment_replace(text):
    aug = {'nga-nyi':'-nga-nyi','na-nyi':'-na-nyi','nga-ni':'-nga-ni', 'na-ni':'-na-ni', 'na-ṉi':'-na-ṉi', 'nga-ṉi':'-nga-ṉi'}
    # Create a regular expression from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, aug.keys()))+ r'$')
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: aug[mo.group()], text,1) 


def ptj_augment_segment(headword, pos):
    segment = []
    for i in range(len(headword)):
        item = headword[i]
        if pos[i] == 'verb':
            item = ptj_augment_replace(item)
            segment.append(item)
        else:
            segment.append(headword[i])
        
    return segment 

def ptj_incho_replace(text):
    aug = {'rinyi':'-rinyi','ringanyi':'-ringanyi'}
    # Create a regular expression from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, aug.keys()))+ r'$')
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: aug[mo.group()], text,1) 


def ptj_incho_segment(headword, pos):
    segment = []
    for i in range(len(headword)):
        item = headword[i]
        if pos[i] == 'verb':
            item = ptj_incho_replace(item)
            segment.append(item)
        else:
            segment.append(headword[i])
        
    return segment 
