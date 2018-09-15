# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 21:28:17 2017

@author: lzzz
"""

class TrieNode(object):
    def __init__(self):
        self.data = {}
        self.is_word = False
        
class Trie(object):
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for letter in word:
            child = node.data.get(letter)
            if not child:
                node.data[letter] = TrieNode()
            node = node.data[letter]
        node.is_word = True
    
    def search(self, word):
        node = self.root
        for letter in word:
            node = node.data.get(letter)
            if not node:
                return False
        return node.is_word
    
    def starts_with(self, prefix):
        node = self.root
        for letter in prefix:
            node = node.data.get(letter)
            if not node:
                return False
        return True
    
    def starts_with_or_is(self, prefix):
        node = self.root
        for letter in prefix:
            node = node.data.get(letter)
            if not node:
                return -1
        if node.is_word:
            return 0
        return 1
    
    def get_start(self, prefix):
        
        def _get_key(pre, pre_node):
            words_list = []
            if pre_node.is_word:
                words_list.append(pre)
            for x in pre_node.data.keys():
                words_list.append(_get_key(pre + str(x), pre_node.data.get(x)))
            return words_list
        
        words = []
        if not self.starts_with(prefix):
            return words
        if self.search(prefix):
            words.append(prefix)
            return words
        node = self.root
        for letter in prefix:
            node = node.data.get(letter)
        return _get_key(prefix, node)
    
    def print_tree(self):
        
        def tranverse(prefix, node):
            if node.data.__len__() == 0:
                print(prefix)
            else:
                for x in node.data.keys():
                    tranverse(prefix + str(x), node.data.get(x))
                if node.is_word:
                    print(prefix)
        
        node = self.root
        tranverse('', node)
