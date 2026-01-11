"""
Unit Tests for AI-Powered PDF & Web Chatbot

This module contains comprehensive unit tests for all major functions
in the chatbot application including URL validation, web scraping,
PDF processing, logging, and AI model interactions.

Run tests with: python -m pytest test_chatbot.py -v
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the functions to test
import sys

sys.path.append(".")

# Mock streamlit to avoid import errors during testing
sys.modules["streamlit"] = Mock()

# Import backend functions after mocking streamlit(
from backend import (
    ChatbotLogger,
    validate_url,
    check_robots_txt,
    scrape_website,
    get_embeddings,
    create_vector_db,
    get_llm,
    extract_llm_response,
)


class TestChatbotLogger:
    """Test cases for the ChatbotLogger class"""

    def setup_method(self):
        """Setup for each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_log_file = os.path.join(self.temp_dir, "test_logs.json")

    def teardown_method(self):
        """Cleanup after each test method"""
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)
        os.rmdir(self.temp_dir)

    @patch("backend.socket.gethostname")
    def test_is_local_deployment_local(self, mock_hostname):
        """Test local deployment detection"""
        mock_hostname.return_value = "my-laptop"
        logger = ChatbotLogger()
        assert logger.is_local == True
        assert logger.deployment_type == "LOCAL"

    @patch("backend.socket.gethostname")
    @patch.dict(os.environ, {"STREAMLIT_SHARING_MODE": "true"})
    def test_is_local_deployment_web(self, mock_hostname):
        """Test web deployment detection"""
        mock_hostname.return_value = "streamlit-server"
        logger = ChatbotLogger()
        assert logger.is_local == False
        assert logger.deployment_type == "WEB"

    def test_log_event_success(self):
        """Test successful event logging"""
        with patch.object(ChatbotLogger, "_initialize_log_file"):
            logger = ChatbotLogger()
            logger.log_file = self.test_log_file

            result = logger.log_event("TEST_EVENT", {"key": "value"})

            assert result == True
            assert os.path.exists(self.test_log_file)

            with open(self.test_log_file, "r") as f:
                data = json.load(f)
                assert len(data["logs"]) == 1
                assert data["logs"][0]["event_type"] == "TEST_EVENT"
                assert data["logs"][0]["details"]["key"] == "value"

    def test_log_error(self):
        """Test error logging functionality"""
        with patch.object(ChatbotLogger, "_initialize_log_file"):
            logger = ChatbotLogger()
            logger.log_file = self.test_log_file

            result = logger.log_error(
                "TEST_ERROR", "Test error message", {"context": "test"}
            )

            assert result == True
            with open(self.test_log_file, "r") as f:
                data = json.load(f)
                log_entry = data["logs"][0]
                assert log_entry["status"] == "error"
                assert log_entry["details"]["error_type"] == "TEST_ERROR"
                assert log_entry["details"]["error_message"] == "Test error message"


class TestURLValidation:
    """Test cases for URL validation functions"""

    def test_validate_url_valid_http(self):
        """Test validation of valid HTTP URL"""
        result = validate_url("http://example.com")
        assert result["valid"] == True

    def test_validate_url_valid_https(self):
        """Test validation of valid HTTPS URL"""
        result = validate_url("https://example.com")
        assert result["valid"] == True

    def test_validate_url_missing_scheme(self):
        """Test validation of URL without scheme"""
        result = validate_url("example.com")
        assert result["valid"] == False
        assert "Missing http://" in result["error"]

    def test_validate_url_invalid_scheme(self):
        """Test validation of URL with invalid scheme"""
        result = validate_url("ftp://example.com")
        assert result["valid"] == False
        assert "HTTP and HTTPS protocols" in result["error"]

    def test_validate_url_localhost_blocked(self):
        """Test that localhost URLs are blocked"""
        result = validate_url("http://localhost:8080")
        assert result["valid"] == False
        assert "local or private network" in result["error"]

    def test_validate_url_private_ip_blocked(self):
        """Test that private IP addresses are blocked"""
        result = validate_url("http://192.168.1.1")
        assert result["valid"] == False
        assert "local or private network" in result["error"]

    def test_validate_url_sensitive_domain_blocked(self):
        """Test that sensitive domains are blocked"""
        sensitive_urls = [
            "https://mybank.com",
            "https://login.example.com",
            "https://admin.site.com",
            "https://medical.records.com",
        ]

        for url in sensitive_urls:
            result = validate_url(url)
            assert result["valid"] == False
            assert "sensitive website" in result["error"]

    def test_validate_url_malformed(self):
        """Test validation of malformed URLs"""
        result = validate_url("not-a-url")
        assert result["valid"] == False


class TestRobotsTxt:
    """Test cases for robots.txt checking"""

    @patch("backend.requests.get")
    def test_check_robots_txt_allows_scraping(self, mock_get):
        """Test robots.txt that allows scraping"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "User-agent: *\nDisallow: /admin"
        mock_get.return_value = mock_response

        result = check_robots_txt("https://example.com")
        assert result["allowed"] == True

    @patch("backend.requests.get")
    def test_check_robots_txt_disallows_scraping(self, mock_get):
        """Test robots.txt that disallows scraping"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "User-agent: *\nDisallow: /"
        mock_get.return_value = mock_response

        result = check_robots_txt("https://example.com")
        assert result["allowed"] == False
        assert "discourages scraping" in result["warning"]

    @patch("backend.requests.get")
    def test_check_robots_txt_not_found(self, mock_get):
        """Test when robots.txt doesn't exist"""
        mock_get.side_effect = Exception("Not found")

        result = check_robots_txt("https://example.com")
        assert result["allowed"] == True
        assert "Could not check robots.txt" in result.get("warning", "")


class TestWebScraping:
    """Test cases for web scraping functionality"""

    @patch("backend.validate_url")
    @patch("backend.check_robots_txt")
    @patch("backend.requests.get")
    def test_scrape_website_success(self, mock_get, mock_robots, mock_validate):
        """Test successful website scraping"""
        # Mock validation and robots.txt check
        mock_validate.return_value = {"valid": True}
        mock_robots.return_value = {"allowed": True}

        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <main>
                    <h1>Main Content</h1>
                    <p>This is test content.</p>
                </main>
                <script>console.log('remove me');</script>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        result = scrape_website("https://example.com")

        assert result["success"] == True
        assert result["title"] == "Test Page"
        assert "Main Content" in result["content"]
        assert "This is test content" in result["content"]
        assert "console.log" not in result["content"]  # Script should be removed

    @patch("backend.validate_url")
    def test_scrape_website_invalid_url(self, mock_validate):
        """Test scraping with invalid URL"""
        mock_validate.return_value = {"valid": False, "error": "Invalid URL"}

        result = scrape_website("invalid-url")

        assert result["success"] == False
        assert result["error"] == "Invalid URL"

    @patch("backend.validate_url")
    @patch("backend.check_robots_txt")
    def test_scrape_website_robots_disallow(self, mock_robots, mock_validate):
        """Test scraping when robots.txt disallows"""
        mock_validate.return_value = {"valid": True}
        mock_robots.return_value = {"allowed": False, "warning": "Robots.txt disallows"}

        result = scrape_website("https://example.com")

        assert result["success"] == False
        assert "Robots.txt disallows" in result["error"]

    @patch("backend.validate_url")
    @patch("backend.check_robots_txt")
    @patch("backend.requests.get")
    def test_scrape_website_http_error(self, mock_get, mock_robots, mock_validate):
        """Test scraping with HTTP error"""
        mock_validate.return_value = {"valid": True}
        mock_robots.return_value = {"allowed": True}
        mock_get.side_effect = Exception("Connection failed")

        result = scrape_website("https://example.com")

        assert result["success"] == False
        assert "Connection failed" in result["error"]
