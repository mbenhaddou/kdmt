import os, re




def __sub_patterns(patterns, text):
    """
    Apply re.sub to bunch of (pattern, repl)
    """
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text

def left_pad(input, length, fill_character=' '):
    """
    Returns a string, which will be padded on the left with characters if necessary.
    If the input string is longer than the specified length, it will be returned unchanged.

    >>> left_pad('foo', 5)
    '  foo'

    >>> left_pad('foobar', 6)
    'foobar'

    >>> left_pad('toolong', 2)
    'toolong'

    >>> left_pad(1, 2, '0')
    '01'

    >>> left_pad(17, 5, 0)
    '00017'

    :param input:
    :param length: The return string's desired length.
    :param fill_character:
    "rtype str:
    """

    return str(input).rjust(length, str(fill_character))


def right_pad(input, length, fill_character=' '):
    """
    Returns a string, which will be padded on the left with characters if necessary.
    If the input string is longer than the specified length, it will be returned unchanged.

    >>> right_pad('foo', 5)
    'foo  '

    >>> right_pad('foobar', 6)
    'foobar'

    >>> right_pad('toolong', 2)
    'toolong'

    >>> right_pad(1, 2, '0')
    '10'

    >>> right_pad(17, 5, 0)
    '17000'

    :param input:
    :param length: The return string's desired length.
    :param fill_character:
    "rtype str:
    """

    return str(input).ljust(length, str(fill_character))


def clean_spaces(text):
    """
    Clean double spaces, trailing spaces, heading spaces,
    spaces before punctuations
    """
    patterns = (
        # arguments for re.sub: pattern and repl
        # удаляем пробел перед знаками препинания
        (r' +([\.,?!\)]+)', r'\1'),
        # добавляем пробел после знака препинания, если только за ним нет другого
        (r'([\.,?!\)]+)([^\.!,?\)]+)', r'\1 \2'),
        # убираем пробел после открывающей скобки
        (r'(\S+)\s*(\()\s*(\S+)', r'\1 (\3'),
    )
    # удаляем двойные, начальные и конечные пробелы
    return os.linesep.join(
        ' '.join(part for part in line.split(' ') if part)
        for line in __sub_patterns(patterns, text).split(os.linesep)
    )

def camel_case(string):
    """
    Convert a string identifier to :code:`camelCase`.
    """
    return lcfirst(pascal_case(string))

def concat(strings):
    """
    Concatenate a list of strings into a single string.
    """
    return ''.join(strings)

def contains(string, matches):
    """
    Determine if a string contains any of the given values. *matches* may be a
    single string, or a list of strings.
    """
    return any([m in string for m in ([matches] if isinstance(matches, str) else matches)])

def contains_all(string, matches):
    """
    Determine if a string contains all of the given values.
    """
    return all([m in string for m in matches])

def dashed_case(string):
    """
    Convert a string identifier to :code:`dashed-case`. If the string is in
    :code:`snake_case`, capitalization of words will be preserved.
    """
    return join(split_identifier(string), '-')

def is_whitespace(string):
    """
    Determine if a string contains only whitespace characters or is empty.
    """
    return string.strip() == ''

def join(strings, sep=', ', insertend=False):
    """
    Concatenate a list of strings into a single string by a separating
    delimiter. If *insertend* is given and true, the delimiter is also included
    at the end of the string.
    """
    return sep.join(strings) + (sep if insertend else '')

def lines(string, keepends=False):
    """
    Split a string into a list of strings at newline characters. Unless
    *keepends* is given and true, the resulting strings do not have newlines
    included.
    """
    return string.splitlines(keepends)

def lcfirst(string):
    """
    Convert the first character of a string to lowercase.
    """
    return string[:1].lower() + string[1:]

def pascal_case(string):
    """
    Convert a string identifier to :code:`PascalCase`.
    """
    return concat(map(ucfirst, split_identifier(string)))

def reverse(string):
    """
    Reverse the order of the characters in a string.
    """
    return string[::-1]

def snake_case(string):
    """
    Convert a string identifier to :code:`snake_case`. If the string is in
    :code:`dashed-case`, capitalization of words will be preserved.
    """
    return join(split_identifier(string), '_')

def split_identifier(string):
    """
    Split a string identifier into a list of its subparts.
    """
    return (
        re.split('[ \-_]', string)
            if re.findall('[ \-_]', string)
            else words(re.sub(r'([a-z])([A-Z0-9])', r'\1 \2', string))
    )

def title_case(string):
    """
    Convert a string identifier to :code:`Title Case`.
    """
    return join(map(ucfirst, split_identifier(string)), ' ')

def words(string):
    """
    Split a string into a list of words, which were delimited by one or more
    whitespace characters.
    """
    return re.split('\s+', string)

def ucfirst(string):
    """
    Convert the first character of a string to uppercase.
    """
    return string[:1].upper() + string[1:]

def unlines(lines, newline='\n'):
    """
    Join a list of lines into a single string after appending a terminating
    newline character to each.
    """
    return join(lines, newline, True)

def unwords(words):
    """
    Join a list of words into a single string with separating spaces.
    """
    return join(words, ' ')