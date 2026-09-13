"""Source adapters for the multi-source news data pipeline."""

from .ag_news import load_ag_news_splits
from .bbc_news import load_bbc_news
from .huffpost import load_huffpost_world_news

__all__ = ["load_ag_news_splits", "load_bbc_news", "load_huffpost_world_news"]
