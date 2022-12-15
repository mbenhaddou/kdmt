import dateparser

for date_str in dateparser.find_dates('May 10, 2022 ', source=True):
    print(date_str)

