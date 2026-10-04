"""Database repositories for data access.

Only generic data-access helpers live here.  ORM models are defined
exclusively in ``storage.models`` — do **not** redeclare models in this
module: mapping the same table name twice on the shared ``Base`` metadata
raises ``sqlalchemy.exc.InvalidRequestError`` at import time.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.database import Base


class Repository:
    """Base repository class."""
    
    def __init__(self, session: AsyncSession, model: type[Base]):
        self.session = session
        self.model = model
    
    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get record by ID."""
        result = await self.session.get(self.model, id)
        return self._to_dict(result) if result else None
    
    async def get_all(self) -> List[Dict[str, Any]]:
        """Get all records."""
        result = await self.session.execute(select(self.model))
        return [self._to_dict(r) for r in result.scalars().all()]
    
    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new record."""
        record = self.model(**data)
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return self._to_dict(record)
    
    async def update(self, id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update record."""
        record = await self.session.get(self.model, id)
        if not record:
            return None
        for key, value in data.items():
            setattr(record, key, value)
        await self.session.commit()
        await self.session.refresh(record)
        return self._to_dict(record)
    
    async def delete(self, id: str) -> bool:
        """Delete record."""
        record = await self.session.get(self.model, id)
        if not record:
            return False
        await self.session.delete(record)
        await self.session.commit()
        return True
    
    def _to_dict(self, record: Any) -> Dict[str, Any]:
        """Convert SQLAlchemy model to dict."""
        return {c.name: getattr(record, c.name) for c in record.__table__.columns}