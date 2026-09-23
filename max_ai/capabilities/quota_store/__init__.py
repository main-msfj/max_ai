"""Where user quotas are counted. The limits live in ``QuotaLimits``."""

from .local import LocalQuotaStore
from .mongodb import MongoDBQuotaStore

__all__ = ["LocalQuotaStore", "MongoDBQuotaStore"]
