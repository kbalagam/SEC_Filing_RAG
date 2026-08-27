"""
SEC EDGAR download functions: ticker -> CIK lookup, fetching the most
recent 10-K's metadata, and building the direct document URL.

Uses the free, official data.sec.gov API. No API key needed, but SEC
requires a descriptive User-Agent header identifying you (set via
SEC_USER_AGENT in .env / config.py) or it will block requests.
"""

import requests

from src.config import SEC_HEADERS


def get_ticker_to_cik_map() -> dict:
    """SEC publishes a single JSON file mapping ticker -> CIK for all companies."""
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=SEC_HEADERS)
    resp.raise_for_status()
    data = resp.json()
    return {entry["ticker"]: str(entry["cik_str"]).zfill(10) for entry in data.values()}


def get_recent_10k(cik: str) -> dict | None:
    """Fetch a company's filing history and return metadata for the most recent 10-K."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=SEC_HEADERS)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    forms = recent["form"]
    for i, form in enumerate(forms):
        if form == "10-K":
            return {
                "accession_number": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
                "filing_date": recent["filingDate"][i],
                "cik": cik,
            }
    return None


def build_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Construct the direct URL to the filing's primary document."""
    acc_no_dashes = accession_number.replace("-", "")
    cik_int = str(int(cik))  # SEC drops leading zeros in the archive path
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}/{primary_document}"


def fetch_filing_html(doc_url: str) -> bytes:
    """Download the raw filing HTML content for parsing."""
    resp = requests.get(doc_url, headers=SEC_HEADERS)
    resp.raise_for_status()
    return resp.content
