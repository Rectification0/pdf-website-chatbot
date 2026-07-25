import pytest
from unittest.mock import patch, MagicMock
from backend import validate_url, check_robots_txt

class TestUrlValidation:
    def test_valid_url(self):
        url = "https://www.google.com"
        result = validate_url(url)
        assert result["valid"] is True

    def test_invalid_scheme(self):
        url = "ftp://example.com"
        result = validate_url(url)
        assert result["valid"] is False
        assert "Only HTTP and HTTPS" in result["error"]

    def test_missing_scheme(self):
        url = "www.google.com"
        result = validate_url(url)
        assert result["valid"] is False
        assert "Missing http:// or https://" in result["error"]
        
    def test_sensitive_domains(self):
        url = "https://www.paypal.com"
        result = validate_url(url)
        assert result["valid"] is False
        assert "sensitive website" in result["error"]

    def test_local_address(self):
        url = "http://localhost:8000"
        result = validate_url(url)
        assert result["valid"] is False
        assert "local or private network" in result["error"]

class TestRobotsTxt:
    @patch('requests.get')
    def test_check_robots_txt_allowed(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "User-agent: *\nDisallow: /admin"
        mock_get.return_value = mock_response

        url = "https://example.com"
        result = check_robots_txt(url)
        assert result["allowed"] is True

    @patch('requests.get')
    def test_check_robots_txt_disallowed(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "User-agent: *\nDisallow: /"
        mock_get.return_value = mock_response

        url = "https://example.com"
        result = check_robots_txt(url)
        assert result["allowed"] is False
        assert "discourages scraping" in result["warning"]
    
    @patch('requests.get')
    def test_robots_txt_not_found(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        url = "https://example.com"
        result = check_robots_txt(url)
        assert result["allowed"] is True

    @patch('requests.get')
    def test_check_robots_txt_exception(self, mock_get):
        mock_get.side_effect = Exception("Connection error")
        
        url = "https://example.com"
        result = check_robots_txt(url)
        assert result["allowed"] is True
        assert "Could not check" in result.get("warning", "")
