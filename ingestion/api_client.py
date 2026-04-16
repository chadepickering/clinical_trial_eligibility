"""
ClinicalTrials.gov REST API v2 wrapper.

Fetches oncology trials in paginated batches and yields raw JSON records.
Endpoint: https://clinicaltrials.gov/api/v2/studies
"""

import time
from typing import Generator

import requests


class ClinicalTrialsClient:
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, page_size: int = 1000, max_trials: int = 15000):
        self.page_size = page_size
        self.max_trials = max_trials
        self.session = requests.Session()

    def fetch(self) -> Generator[dict, None, None]:
        params = {
            "query.cond": "cancer OR oncology OR neoplasm OR tumor OR carcinoma",
            "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
            "pageSize": self.page_size,
            "format": "json",
            # No "fields" filter — fetch full JSON and extract in parser.py
        }

        fetched = 0
        next_page_token = None

        while fetched < self.max_trials:
            if next_page_token:
                params["pageToken"] = next_page_token

            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            studies = data.get("studies", [])
            for study in studies:
                yield study
                fetched += 1
                if fetched >= self.max_trials:
                    return

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

            time.sleep(0.1)  # respect 10 requests/second rate limit
