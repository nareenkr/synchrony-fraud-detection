"""Privacy-preserving identifier transformations used at the system boundary."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress


class Pseudonymizer:
    """Create stable, namespace-separated keyed pseudonyms.

    HMAC prevents dictionary attacks that would be possible with a plain identifier hash.
    Namespaces prevent equal raw values in separate identifier domains from linking.
    """

    def __init__(self, secret: str | bytes) -> None:
        key = secret.encode("utf-8") if isinstance(secret, str) else secret
        if len(key) < 16:
            raise ValueError("Pseudonym key must contain at least 16 bytes")
        self._key = key

    def pseudonymize(self, namespace: str, value: str) -> str:
        clean_namespace = namespace.strip().lower()
        clean_value = value.strip()
        if not clean_namespace or not clean_value:
            raise ValueError("Namespace and identifier must be non-empty")
        message = f"{clean_namespace}\x00{clean_value}".encode()
        digest = hmac.new(self._key, message, hashlib.sha256).hexdigest()
        return f"{clean_namespace}_{digest[:24]}"


def coarse_ip_network(value: str) -> str:
    """Return a privacy-reduced /24 IPv4 or /48 IPv6 network string."""

    address = ipaddress.ip_address(value)
    prefix = 24 if address.version == 4 else 48
    return str(ipaddress.ip_network(f"{address}/{prefix}", strict=False))
