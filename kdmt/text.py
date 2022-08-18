import regex as re
import sys
from kdmt.__sentence_splitter import split_multi, split_single

import unicodedata

def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.

  if char == " " or char == "\t" or char == "\n" or char == "\r" or char == "\xa0" or char== u"\\x000D" or char == u"\\xa333":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False


def is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
          (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False


def case_sensitive_replace(string, old, new):
    """ replace occurrences of old with new, within string
        replacements will match the case of the text it replaces
    """

    def repl(match):
        current = match.group()
        result = ''
        all_upper = True
        for i, c in enumerate(current):
            if i >= len(new):
                break
            if c.isupper():
                result += new[i].upper()
            else:
                result += new[i].lower()
                all_upper = False
        # append any remaining characters from new
        if all_upper:
            result += new[i + 1:].upper()
        else:
            result += new[i + 1:].lower()
        return result

    old_re = r"\b{}\b".format(old)
    old_p_re = r"\b{}\b\.".format(old)
    regex = re.compile(old_re, re.I)
    regex_p = re.compile(old_p_re, re.I)

    # print(f"[{repl}][{string}]")

    string = regex_p.sub(repl, string)
    return regex.sub(repl, string)

def normalize(inputs, remove_space=True, lower=False):
    """preprocess data by removing extra space and normalize data."""
    outputs = inputs
    if remove_space:
        outputs = " ".join(inputs.strip().split())

    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def clean_text(text):
    text = str(text)
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or unicodedata.category(char)[0]=="C" or char in ['\xad', '­', '_x000D_', u'\ufeff']:
            continue
        if is_whitespace(char):
            output.append(" ")
        elif char =='：':
            output.append(': ')
        elif char == [u'\u200b', u'\\u1427']:
            continue
        elif char ==u'\u2026':
            output.append('...')
        elif char ==u'’':
            output.append("'")
        elif char in [u'\\u2019', u'\\xb4']:
            output.append("'")
        elif char ==u'\\u2013':
            output.append('-')
        elif char ==u'f\\xfcr':
            output.append( u'\'s')
        elif char ==u'\\xa':
            output.append(u' x')
        elif char in[u'\\xa0', u'\u3000']:
            output.append(u' ')

        elif char ==u'\\xa333':
            output.append(u'  ')
        elif char ==u'x000D':
            output.append( u'\n')
        else:
            output.append(char)

    return "".join(output)



def split_text_to_sentences(text, multi_line=False):
    """
     multi_line: Default= False. split `text` at sentence terminals and at newline chars.
     Option2: multi_line=True Sentences may contain non-consecutive (single) newline chars, while consecutive newline chars
    ("paragraph separators") always split sentences
    """

    if multi_line==False:
        return split_single(text, simple_split=False)
    else:
        return list(split_multi(text))




def remove_punctuations(text, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    if sys.version_info < (3,):
        if isinstance(text, unicode):  # noqa: F821
            translate_map = {
                ord(c): unicode(split) for c in filters  # noqa: F821
            }
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = str.maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = {c: split for c in filters}
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)
    return text.strip()


def ngram(text, n=1, stride=None):
    if stride == None:
        stride = n
    return [text[i:i + n] for i in range(0, len(text), stride)]

if __name__=="__main__":
    test="""His dense and friendly comments have guided many programmers on the linux kernel mailing list. Overview of Visual FoxPro training options We offer practical, hands-on training for programmers 
    of Microsoft."""

    print(split_text_to_sentences(test, multi_line=True))
