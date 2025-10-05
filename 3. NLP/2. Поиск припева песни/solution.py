import regex as re
from nltk import PorterStemmer
from typing import List, Tuple, Set
from Levenshtein import distance as levenshtein_distance

class Solution:
    def __init__(self):
        self.stemmer = PorterStemmer()
        
    def detect(self, tracks: List[Tuple[List[str], str]]) -> List[Set[Tuple[int, int]]]:            
        return [self.find_chorus(self.preprocess_text(lines)) for lines, title in tracks]
    
    def preprocess_text(self, lines: List[str]) -> List[str]:
        processed_lines = []
        
        for line in lines:
            line = line.lower()
            line = re.sub(r'\[.*?\]|\(.*?\)', '', line)
            line = re.sub(r'[^\p{L}\p{N}\s]', '', line)
            words = line.split()
            stemmed_words = [self.stemmer.stem(word) for word in words]
            processed_lines.append(" ".join(stemmed_words))
        
        return processed_lines
    
    def find_chorus(self, lines: List[str]) -> Set[Tuple[int, int]]:
        num_lines = len(lines)
        max_size = min(10, num_lines // 4)
        
        best_size = 0
        best_line_index = -1
        max_ones_count = 0
        best_similarity_matrix = []

        for size in range(2, max_size + 1):
            num_blocks = num_lines - size + 1
            
            similarity_matrix = [[0] * num_blocks for _ in range(num_blocks)]
            
            for i in range(num_blocks):
                for j in range(i + 1, num_blocks):
                    block_i = ' '.join(lines[i:i + size])
                    block_j = ' '.join(lines[j:j + size])
                    lev_distance = levenshtein_distance(block_i, block_j)
                    similarity_score = lev_distance / max(len(block_i), len(block_j))
                    
                    if similarity_score <= 0.075:
                        similarity_matrix[i][j] = similarity_matrix[j][i] = 1
            
            for i in range(num_blocks):
                ones_count = sum(similarity_matrix[i])
                
                if ones_count > max_ones_count or (ones_count == max_ones_count and size > best_size):
                    best_size = size
                    best_line_index = i
                    max_ones_count = ones_count
                    best_similarity_matrix = similarity_matrix

        if best_line_index == -1:
            return set()
        
        chorus_intervals = set()
        best_similarity_matrix[best_line_index][best_line_index] = 1
        
        for j in range(len(best_similarity_matrix[best_line_index])):
            if best_similarity_matrix[best_line_index][j] == 1:
                chorus_intervals.add((j, j + best_size - 1))
        
        return chorus_intervals