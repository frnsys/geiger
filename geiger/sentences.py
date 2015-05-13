from nltk import word_tokenize


class Sentence():
    def __init__(self, body, comment):
        self.body = body
        self.comment = comment


def prefilter(sentence):
    """
    Ignore sentences for which this returns False.
    """
    tokens = word_tokenize(sentence.lower())
    if not tokens:
        return False

    first_word = tokens[0]
    first_char = first_word[0]
    final_char = tokens[-1][-1]

    # Filter out short sentences.
    if len(tokens) <= 12:
        return False

    # No questions
    elif '?' in sentence:
        return False

    # No quotes
    elif any((c in {'"\'‘’“”'}) for c in sentence):
        return False

    # Should begin with an uppercase (catches improperly tokenized sentences)
    elif not sentence[0].isupper():
        return False

    # The following rules are meant to filter out sentences
    # which may require extra context.
    elif first_char in ['"', '(', '\'', '*', '“', '‘', ':']:
        return False
    elif first_word in ['however', 'but', 'so', 'for', 'or', 'and', 'thus', 'therefore', 'also', 'firstly', 'secondly', 'thirdly']:
        return False
    # TO DO perhaps this should only check if these are in the first n words
    elif set(tokens).intersection({'he', 'she', 'it', 'they', 'them', 'him', 'her', 'their', 'i'}):
        return False
    elif final_char in ['"', '”', '’', "'"]:
        return False

    return True
