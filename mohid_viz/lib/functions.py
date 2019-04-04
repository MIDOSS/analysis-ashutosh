from datetime import datetime, timedelta
from dateutil.parser import parse

def _convert_timestamp(time):
    """Convert datetime.datetime to string in datetime64[s] format
    :arg time: datetime.datetime object
    :return datetime64: str in datetime64[s] format
    """
    year, month, day, hour, minute, second = str(time.year), str(time.month), str(time.day), str(time.hour), str(time.minute), str(time.second)
    if len(month) < 2:
        month = '0' + month
    if len(day) < 2:
        day = '0' + day
    if len(hour) < 2:
        hour = '0' + hour
    if len(minute) < 2:
        minute = '0' + minute
    if len(second) < 2:
        second = '0' + second
    datetime64 = '{}-{}-{}T{}:{}:{}'.format(year, month, day, hour, minute, second)
    return datetime64
