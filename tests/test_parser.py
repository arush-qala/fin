import builtins
from types import SimpleNamespace
import os
import pytest

import app


def test_download_and_parse_13f_xml(monkeypatch):
    # Load sample XML
    here = os.path.dirname(__file__)
    sample_path = os.path.join(here, 'data', 'sample_13f.xml')
    with open(sample_path, 'rb') as f:
        xml_bytes = f.read()

    # Fake responses: first call is the filing page (with href to /sample.xml),
    # second call returns the XML bytes from SEC host
    def fake_get(url, headers=None, timeout=None):
        if url == 'https://example.com/filing':
            return SimpleNamespace(text='Some filing page content href="/sample.xml" more', content=b'', status_code=200)
        if url == 'https://www.sec.gov/sample.xml':
            return SimpleNamespace(content=xml_bytes, text=xml_bytes.decode('utf-8'), status_code=200)
        return SimpleNamespace(text='', content=b'', status_code=404)

    monkeypatch.setattr(app.requests, 'get', fake_get)

    df, period = app.download_and_parse_13f_xml('https://example.com/filing')
    assert df is not None
    assert period == '2025-09-30'
    assert len(df) == 2
    names = set(df['name'].tolist())
    assert 'Company A' in names
    assert 'Company B' in names
