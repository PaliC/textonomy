# from Levenshtein import ratio
from nltk.stem import PorterStemmer

# Takes a bunch of strings and sorts them into groups of similar strings according to Levenshtein distance.
# Tolerance is a float from 0 to 1. 
def fuzzy_group(strings, tol):
    
    groups = {}
    
    for i in range(len(strings)):
        curr = strings[i]
        append_to = curr

        for key, key_group in groups.items():
            if ratio(curr, key) >= tol:
                append_to = key
                break
                
        if groups.get(append_to) == None:
            groups[append_to] = [curr]
        else:
            groups[append_to].append(curr)
    
    return groups

# fuzzy_group was too slow, so this function just compares everything in strings to 
# one keyword, which is linear and much faster
def fuzzy_group_key(strings, keyword, tol):
    
    matches = [keyword]
    
    for i in range(len(strings)):
        curr = strings[i]
        append_to = curr

        if ratio(curr, keyword) >= tol:
            matches.append(curr) 
    
    return matches

# An answer to the terrible runtime we had before! Sort everything alphabetically,
# and if adjacent words are in the same group, group them! (this is real shitty)
def linear_fuzzy_group(strings, tol):

    strings.sort()
    
    groups = {}

    curr = strings[0]
    curr_group_key = strings[0]
    groups[curr_group_key] = [curr_group_key]
    
    for i in range(1, len(strings)):
        prev = curr
        curr = strings[i]
        if ratio(curr, curr_group_key) >= tol or ratio(curr, prev) >= tol:
            groups[curr_group_key].append(curr)
        else:
            groups[curr] = [curr]
            curr_group_key = curr
              
    return groups

# coolio
def stem_group(strings):
    
    groups = {}
    ps = PorterStemmer()
    
    for i in range(len(strings)):
        curr = strings[i]
        curr_stem = ps.stem(curr)
        
        if groups.get(curr_stem) == None:
            groups[curr_stem] = [curr]
        else:
            groups[curr_stem].append(curr)
              
    return groups
