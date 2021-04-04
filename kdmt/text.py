import regex as re
import unicodedata

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


def clean_unicode(text):

    text = text.replace(u'\\xa333', u' ')
    text = text.replace(u'\\u2019', u'\'')
    text = text.replace('\xad', '')
    text = text.replace('­', '')
    text = text.replace(u'\\xb4', u'\'')
    text = text.replace(u'\\xa0', u' ')
    text = text.replace(u'f\\xfcr', u'\'s')
    text = text.replace(u'\\xa', u' x')
    text = text.replace('_x000D_', '')
    text = text.replace(u'x000D', u'\n')
    text = text.replace(u'.à', u' a')
    text = text.replace(u'\ufeff', u'')
    text = text.replace(u'\u3000',u' ')

    return text
