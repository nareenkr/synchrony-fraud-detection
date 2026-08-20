"""Real-time behavioral state adapters.

The in-memory implementation is the local-development adapter.  Production
deployments with more than one API worker should implement
:class:`RealtimeStateStore` with an external, atomic store (for example
Redis); callers do not depend on implementation-specific primitives.
"""

from .base import RealtimeStateStore, StateSnapshot
from .memory import MemoryRealtimeStateStore
from .redis import RedisRealtimeStateStore

__all__ = [
    "MemoryRealtimeStateStore",
    "RealtimeStateStore",
    "RedisRealtimeStateStore",
    "StateSnapshot",
]
