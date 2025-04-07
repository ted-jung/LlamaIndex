# =============================================================================
# MCP env
# Created: 7, Apr 2025
# Updated: 7, Apr 2025
# Writer: Ted Jung
# Description:
#   MCP Configuration for ClickHouse server
#   Handles all environment variable configuration with sensible defaults
#   and type conversion.
# =============================================================================

"""Environment configuration for the MCP ClickHouse server.

This module handles all environment variable configuration with sensible defaults
and type conversion.
"""

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ClickHouseConfig:
    """Configuration for ClickHouse connection settings.

    This class handles all environment variable configuration with sensible defaults
    and type conversion. It provides typed methods for accessing each configuration value.

    Required environment variables:
        CLICKHOUSE_HOST: The hostname of the ClickHouse server
        CLICKHOUSE_USER: The username for authentication
        CLICKHOUSE_PASSWORD: The password for authentication

    Optional environment variables (with defaults):
        CLICKHOUSE_PORT: The port number (default: 8443 if secure=True, 8123 if secure=False)
        CLICKHOUSE_SECURE: Enable HTTPS (default: true)
        CLICKHOUSE_VERIFY: Verify SSL certificates (default: true)
        CLICKHOUSE_CONNECT_TIMEOUT: Connection timeout in seconds (default: 30)
        CLICKHOUSE_SEND_RECEIVE_TIMEOUT: Send/receive timeout in seconds (default: 300)
        CLICKHOUSE_DATABASE: Default database to use (default: None)
    """

    def __init__(self):
        """Initialize the configuration from environment variables."""
        self._validate_required_vars()


    def get_client_config(self) -> dict:
        """Get the configuration dictionary for clickhouse_connect client.

        Returns:
            dict: Configuration ready to be passed to clickhouse_connect.get_client()
        """
        config = {
            "host": os.getenv("CLICKHOUSE_HOST"),
            "port": os.getenv("CLICKHOUSE_PORT"),
            "username": os.getenv("CLICKHOUSE_USER"),
            "password": os.getenv("CLICKHOUSE_PASSWORD"),
            "secure": os.getenv("CLICKHOUSE_SECURE"),
            "verify": os.getenv("CLICKHOUSE_VERIFY"),
            "database": os.getenv("CLICKHOUSE_DATABASE"),
            "connect_timeout": os.getenv("CLICKHOUSE_CONNECT_TIMEOUT", "30"),
            "send_receive_timeout": os.getenv("CLICKHOUSE_SEND_RECEIVE_TIMEOUT", "300"),
            "client_name": "mcp_clickhouse",
        }

        # Add optional database if set
        # if self.database:
        #     config["database"] = self.database

        return config

    def _validate_required_vars(self) -> None:
        """Validate that all required environment variables are set.

        Raises:
            ValueError: If any required environment variable is missing.
        """
        missing_vars = []
        for var in ["CLICKHOUSE_HOST", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD"]:
            if var not in os.environ:
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

# Global instance for easy access
ch_config = ClickHouseConfig()