"""Tests for ClinicalTrials.gov API client."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from ingestion.api_client import ClinicalTrialsClient


def make_response(studies: list, next_page_token: str = None, status_code: int = 200):
    """Helper — build a mock requests.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {
        "studies": studies,
        **({"nextPageToken": next_page_token} if next_page_token else {}),
    }
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = requests.HTTPError(
            response=mock_resp
        )
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


SAMPLE_STUDY = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT00000001",
            "briefTitle": "A Sample Oncology Trial",
        },
        "statusModule": {"overallStatus": "RECRUITING"},
        "eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria:\n* Age >= 18"},
    }
}


# ---------------------------------------------------------------------------
# Unit tests (mocked — no network)
# ---------------------------------------------------------------------------

@patch("ingestion.api_client.requests.Session")
def test_fetch_returns_studies(mock_session_cls):
    """Client yields one dict per study in the response."""
    session = mock_session_cls.return_value
    session.get.return_value = make_response([SAMPLE_STUDY])

    client = ClinicalTrialsClient(page_size=10, max_trials=10)
    results = list(client.fetch())

    assert len(results) == 1
    assert results[0] == SAMPLE_STUDY


@patch("ingestion.api_client.requests.Session")
def test_fetch_respects_max_trials(mock_session_cls):
    """Client stops after max_trials records even if more exist."""
    session = mock_session_cls.return_value
    # Return 5 studies per page but cap at 3
    session.get.return_value = make_response([SAMPLE_STUDY] * 5)

    client = ClinicalTrialsClient(page_size=5, max_trials=3)
    results = list(client.fetch())

    assert len(results) == 3


@patch("ingestion.api_client.time.sleep")
@patch("ingestion.api_client.requests.Session")
def test_fetch_paginates(mock_session_cls, mock_sleep):
    """Client follows nextPageToken across multiple pages."""
    session = mock_session_cls.return_value
    page1 = make_response([SAMPLE_STUDY] * 2, next_page_token="token_abc")
    page2 = make_response([SAMPLE_STUDY] * 2)
    session.get.side_effect = [page1, page2]

    client = ClinicalTrialsClient(page_size=2, max_trials=10)
    results = list(client.fetch())

    assert len(results) == 4
    assert session.get.call_count == 2
    # Second call should include pageToken param
    second_call_params = session.get.call_args_list[1][1]["params"]
    assert second_call_params["pageToken"] == "token_abc"


@patch("ingestion.api_client.requests.Session")
def test_fetch_stops_without_next_page_token(mock_session_cls):
    """Client terminates cleanly when no nextPageToken is returned."""
    session = mock_session_cls.return_value
    session.get.return_value = make_response([SAMPLE_STUDY])

    client = ClinicalTrialsClient(page_size=10, max_trials=100)
    results = list(client.fetch())

    assert len(results) == 1
    assert session.get.call_count == 1


@patch("ingestion.api_client.requests.Session")
def test_http_error_raises(mock_session_cls):
    """Client raises HTTPError on a non-2xx response."""
    session = mock_session_cls.return_value
    session.get.return_value = make_response([], status_code=503)

    client = ClinicalTrialsClient(page_size=10, max_trials=10)

    with pytest.raises(requests.HTTPError):
        list(client.fetch())


# ---------------------------------------------------------------------------
# Integration test (live network — skipped unless --integration flag passed)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_live_fetch_three_records():
    """Hit the real API and confirm 3 records have expected structure."""
    client = ClinicalTrialsClient(page_size=3, max_trials=3)
    results = list(client.fetch())

    assert len(results) == 3

    for study in results:
        assert "protocolSection" in study
        ps = study["protocolSection"]

        # identificationModule must exist and have a real NCT ID
        nct_id = ps.get("identificationModule", {}).get("nctId", "")
        assert nct_id.startswith("NCT"), f"Unexpected nctId: {nct_id}"

        # statusModule must be present
        assert "statusModule" in ps

        # eligibilityModule and criteria text must be present
        eligibility = ps.get("eligibilityModule", {}).get("eligibilityCriteria", "")
        assert len(eligibility) > 0, f"Empty eligibility text for {nct_id}"
