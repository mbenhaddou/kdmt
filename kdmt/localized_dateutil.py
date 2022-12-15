import calendar
from dateutil import parser
import locale


def _lower_tuples(lst):
    lst2=[]
    for v  in lst:
        lst2.append((v[0].lower(), v[1].lower()))

    return lst2

class LocaleParserInfo(parser.parserinfo):

    locale_list=[ 'de_DE.UTF-8', 'nl_BE', 'es_ES', 'fr_FR', 'it_IT', 'pt_PT']
    locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
    WEEKDAYS=_lower_tuples(list(zip(calendar.day_abbr, calendar.day_name)))
    MONTHS = _lower_tuples(list(zip(calendar.month_abbr, calendar.month_name))[1:])

    for l in locale_list:
        locale.setlocale(locale.LC_TIME, l)
        MT = _lower_tuples(list(zip(calendar.month_abbr, calendar.month_name))[1:])
        for i, m in enumerate(MT):
            MONTHS[i]+=m
    MONTHS=[tuple(set(M)) for M in MONTHS]
    all_months = [m for month in MONTHS for m in month]
    for l in locale_list:
        locale.setlocale(locale.LC_TIME, l)
        WD = _lower_tuples(list(zip(calendar.day_abbr, calendar.day_name)))
        for i, d in enumerate(WD):
            if d[0] not in all_months:
                WEEKDAYS[i]+=(d[0],)
            if d[1] not in all_months:
                WEEKDAYS[i]+=(d[1],)



    WEEKDAYS=[tuple(set(WD)) for WD in WEEKDAYS]

    HMS = [("h", "hour", "hours", "heure", "heures", "uur", "uren"),
           ("m", "minute", "minutes", "minuten"),
           ("s", "second", "seconds", "seconde", "secondes", "seconden")]




def get_local_from_date_string(date_str, local_list=[ 'fr_FR','nl_BE','de_DE.UTF-8', 'es_ES', 'it_IT', 'pt_PT']):

    string_parts=[t.strip('.,') for t in date_str.split(' ') if len(t)>=2 and t.strip('.,').isalpha()]
    for local in local_list:
        nb_matches=0
        WEEKDAYS = [ w for week in _lower_tuples(list(zip(calendar.day_abbr, calendar.day_name))) for w in week]
        MONTHS = [m for month in _lower_tuples(list(zip(calendar.month_abbr, calendar.month_name))[1:]) for m in month]
        for part in string_parts:
            part_=part.stip('.,')
            if part_ in WEEKDAYS:
                nb_matches+=1

            elif part_ in MONTHS:
                nb_matches+=1

        if nb_matches==len(string_parts) or nb_matches==2:
            return local



if __name__=="__main__":

    from dateutil.parser import parse

    print(parse("'Envoyé : mardi 1 décembre 2020 à 14:51:27 UTC+1'", parserinfo=LocaleParserInfo()))
