"""ADR-Sentinel v2 application entry point.

India's ADR reporting rate is below 1%, against a global average of 6-10% 
(WHO/PvPI data). Studies attribute this primarily to two barriers: 
(1) low awareness/detection capability among healthcare professionals, and 
(2) time-consuming, friction-heavy manual reporting processes to PvPI/ADR 
Monitoring Centres. Drug-related hospital admissions range from 0.2% to 22% 
globally (WHO), rising to 10-20% in geriatric patients, most of which are 
preventable with earlier detection. This project targets BOTH barriers, not 
just detection.

Supplied project background; numerical claims need primary-source verification.
"""

import uvicorn

from sentinel import config
from sentinel.api import app

if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level=config.LOG_LEVEL)
