from datetime import timedelta
from datetime import timezone
import time

def currenttz():
    if time.daylight:
        return timezone(timedelta(seconds=-time.altzone),time.tzname[1])
    else:
        return timezone(timedelta(seconds=-time.timezone),time.tzname[0])
