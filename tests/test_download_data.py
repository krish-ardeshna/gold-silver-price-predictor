from pathlib import Path
from unittest.mock import patch

import pandas as pd

import src.download_data as download_data


class TestDownloadData:
    @patch("src.download_data.yf.download")
    def test_download_successful(self, mock_download, workspace_tmp_path):
        mock_download.return_value = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "Close": [1500, 1501, 1502, 1503, 1504],
                "Open": [1498, 1499, 1500, 1501, 1502],
            }
        )

        result = download_data.download_futures_data("GC=F", "gold", output_dir=workspace_tmp_path)

        assert result is True
        saved = pd.read_csv(workspace_tmp_path / "gold.csv")
        assert len(saved) == 5
        assert "Close" in saved.columns

    @patch("src.download_data.yf.download")
    def test_download_handles_multiindex_columns(self, mock_download, workspace_tmp_path):
        columns = pd.MultiIndex.from_tuples(
            [
                ("Close", "GC=F"),
                ("Open", "GC=F"),
                ("High", "GC=F"),
            ]
        )
        mock_download.return_value = pd.DataFrame([[1500, 1499, 1502]], columns=columns)

        result = download_data.download_futures_data("GC=F", "gold", output_dir=workspace_tmp_path)

        assert result is True
        saved = pd.read_csv(workspace_tmp_path / "gold.csv")
        assert "Close" in saved.columns

    @patch("src.download_data.yf.download")
    def test_download_returns_false_for_empty_data(self, mock_download, workspace_tmp_path):
        mock_download.return_value = pd.DataFrame()

        result = download_data.download_futures_data("GC=F", "gold", output_dir=workspace_tmp_path)

        assert result is False
        assert not (workspace_tmp_path / "gold.csv").exists()

    @patch("src.download_data.yf.download")
    def test_download_returns_false_on_exception(self, mock_download, workspace_tmp_path):
        mock_download.side_effect = RuntimeError("network issue")

        result = download_data.download_futures_data("GC=F", "gold", output_dir=workspace_tmp_path)

        assert result is False

    def test_download_uses_default_output_dir(self, workspace_tmp_path, monkeypatch):
        monkeypatch.setattr(download_data, "BASE_DIR", str(workspace_tmp_path))
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=3, freq="D"),
                "Close": [1.0, 2.0, 3.0],
            }
        )

        with patch("src.download_data.yf.download", return_value=frame):
            result = download_data.download_futures_data("GC=F", "gold")

        assert result is True
        assert (workspace_tmp_path / "data" / "gold.csv").exists()

    @patch("src.download_data.download_futures_data", side_effect=[True, True])
    def test_main_returns_zero_when_all_downloads_succeed(self, mock_download):
        assert download_data.main() == 0
        assert mock_download.call_count == 2

    @patch("src.download_data.download_futures_data", side_effect=[True, False])
    def test_main_returns_one_when_any_download_fails(self, mock_download):
        assert download_data.main() == 1
        assert mock_download.call_count == 2

    @patch("src.download_data.download_futures_data", side_effect=[False, False])
    def test_main_returns_one_when_all_downloads_fail(self, mock_download):
        assert download_data.main() == 1
        assert mock_download.call_count == 2
