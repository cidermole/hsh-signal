
#### TODO FIXME broken imports.

import sys
sys.path.append('/mnt/hsh/heartshield-server-backend')

from hrv.classical import time_domain, frequency_domain

def compute_hrv(ibis):
    d = time_domain(ibis*1e3)  # convert sec to ms
    d.update(frequency_domain(ibis*1e3))
    return d
