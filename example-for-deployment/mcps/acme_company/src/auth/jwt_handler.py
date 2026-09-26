"""Create and verify the JWTs"""

import datetime
import logging
import typing as t

import jwt
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken

from src.config import settings

logger = logging.getLogger(__name__)


def generate_token(subject: str, time_to_expire: int | float | None = 1) -> str:
    """Generate a JWT for MCP authentication; `time_to_expire` is in hours."""
    if not subject:
        raise ValueError("Subject must be a non-empty string.")
    if time_to_expire is not None and time_to_expire <= 0:
        raise ValueError("time_to_expire must be a positive number.")
    now = datetime.datetime.now(datetime.timezone.utc)
    payload: dict[str, t.Any] = {"iat": now, "sub": subject, "scope": "user"}
    if time_to_expire is not None:
        payload["exp"] = now + datetime.timedelta(hours=time_to_expire)
    return jwt.encode(payload, settings.MCP_SEED, algorithm="HS256")


class AuthTokenHandler:
    """Token verifier for MCPServer: accepts tokens signed with MCP_SEED."""

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            claims = jwt.decode(token, settings.MCP_SEED, algorithms=["HS256"])
        except jwt.PyJWTError as error:
            logger.warning("Rejected token: %s", error)
            return None
        return AccessToken(
            token=token, client_id=claims["sub"], subject=claims["sub"],
            scopes=claims.get("scope", "").split(), expires_at=claims.get("exp"),
        )


def current_employee() -> str:
    """The employee ID of the token that made this request."""
    token = get_access_token()
    if token is None or not token.subject:
        raise PermissionError("Not authenticated")
    return token.subject


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a JWT token.")
    parser.add_argument("--hours", type=float, default=None, help="Expiration in hours - omit for no expiration")
    parser.add_argument("--subject", required=True, help="Token subject (employee ID)")
    args = parser.parse_args()
    print(generate_token(args.subject, args.hours))
