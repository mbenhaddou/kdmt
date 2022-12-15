__version__ = '1.1.1'

from kdmt.dateparser.date import DateDataParser
from kdmt.dateparser.conf import apply_settings
from kdmt.dateparser.utils.date_extractor import extract_date_strings_inner, _find_and_replace
_default_parser = DateDataParser()


@apply_settings
def parse(date_string, date_formats=None, languages=None, locales=None,
          region=None, settings=None, detect_languages_function=None):
    """Parse date and time from given date string.

    :param date_string:
        A string representing date and/or time in a recognizably valid format.
    :type date_string: str

    :param date_formats:
        A list of format strings using directives as given
        `here <https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior>`_.
        The parser applies formats one by one, taking into account the detected languages/locales.
    :type date_formats: list

    :param languages:
        A list of language codes, e.g. ['en', 'es', 'zh-Hant'].
        If locales are not given, languages and region are used to construct locales for translation.
    :type languages: list

    :param locales:
        A list of locale codes, e.g. ['fr-PF', 'qu-EC', 'af-NA'].
        The parser uses only these locales to translate date string.
    :type locales: list

    :param region:
        A region code, e.g. 'IN', '001', 'NE'.
        If locales are not given, languages and region are used to construct locales for translation.
    :type region: str

    :param settings:
        Configure customized behavior using settings defined in :mod:`dateparser.conf.Settings`.
    :type settings: dict

    :param detect_languages_function:
        A function for language detection that takes as input a string (the `date_string`) and
        a `confidence_threshold`, and returns a list of detected language codes.
        Note: this function is only used if ``languages`` and ``locales`` are not provided.
    :type detect_languages_function: function

    :return: Returns :class:`datetime <datetime.datetime>` representing parsed date if successful, else returns None
    :rtype: :class:`datetime <datetime.datetime>`.
    :raises:
        ``ValueError``: Unknown Language, ``TypeError``: Languages argument must be a list,
        ``SettingValidationError``: A provided setting is not valid.
    """
    parser = _default_parser

    if languages or locales or region or detect_languages_function or not settings._default:
        parser = DateDataParser(languages=languages, locales=locales,
                                region=region, settings=settings, detect_languages_function=detect_languages_function)

    data = parser.get_date_data(date_string, date_formats)

    if data:
        return data['date_obj']


def find_dates(text, source=False, index=False, strict=False):
    for date_string, indices, captures in extract_date_strings_inner(
            text, strict=strict
    ):

        if date_string is None:
            return None
        as_dt = parse(date_string)
        if as_dt is None:
            date_string=_find_and_replace(date_string, captures)
            as_dt = parse(date_string)
            if as_dt is None:
                continue

        returnables = (as_dt,)
        if source:
            returnables = returnables + (date_string,)
        if index:
            returnables = returnables + (indices,)

        if len(returnables) == 1:
            returnables = returnables[0]
        yield returnables

