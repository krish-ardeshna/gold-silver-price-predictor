import pytest


pytestmark = pytest.mark.skip(reason="Streamlit UI tests are deferred while the app UI is intentionally frozen.")


def test_app_ui_placeholder():
    assert True
