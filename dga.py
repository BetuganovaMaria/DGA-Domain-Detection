import random
import string
from datetime import datetime


def generate_dga_domains(count, min_length, max_length):
    domains = []
    tlds = [
        # gTLD
        '.com', '.net', '.org', '.info', '.biz', '.co', '.us', '.io', '.me', '.xyz',

        # ccTLD
        '.uk', '.de', '.fr', '.ru', '.cn', '.jp', '.br', '.in', '.au', '.ca', '.es',
        '.it', '.nl', '.se', '.no', '.fi', '.pl', '.ch', '.be', '.at', '.dk', '.il',
        '.mx', '.ar', '.pt', '.nz', '.sg', '.hk', '.my', '.th', '.za', '.tr', '.gr',
        '.cz', '.ro', '.sk', '.hu', '.ee', '.lt', '.lv', '.ua', '.bg', '.hr', '.si',
        '.is', '.ph', '.tw', '.sa', '.ae', '.eg', '.pk', '.kz', '.by', '.id',

        # New gTLD
        '.app', '.blog', '.shop', '.site', '.online', '.tech', '.dev', '.space',
        '.store', '.cloud', '.fun', '.top', '.guru', '.life', '.news', '.live',
        '.media', '.agency', '.pro', '.world', '.edu'
    ]
    allowed_chars = string.ascii_lowercase + "-"

    seed = int(datetime.now().strftime('%Y%m%d'))
    random.seed(seed)

    for _ in range(count):
        length = random.randint(min_length, max_length)
        domain = ''.join(random.choices(allowed_chars, k=length))

        domain = domain.strip('-')
        while '--' in domain:
            domain = domain.replace('--', '-')

        tld = random.choice(tlds)
        domains.append(f'{domain}{tld}')

    return domains
