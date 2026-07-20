import logging

import aiomysql

from app.config import settings

logger = logging.getLogger(__name__)

_pool: aiomysql.Pool | None = None


async def init_db_pool() -> None:
    global _pool
    _pool = await aiomysql.create_pool(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        db=settings.mysql_db,
        charset="utf8mb4",
        autocommit=False,
        minsize=2,
        maxsize=10,
    )
    logger.info("MySQL pool created: %s:%s/%s", settings.mysql_host, settings.mysql_port, settings.mysql_db)


async def close_db_pool() -> None:
    global _pool
    if _pool:
        _pool.close()
        await _pool.wait_closed()
        _pool = None
        logger.info("MySQL pool closed")


def get_db_pool() -> aiomysql.Pool | None:
    return _pool
