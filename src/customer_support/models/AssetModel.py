from sqlalchemy import select

from .BaseDataModel import BaseDataModel
from .db_schemas import Asset


class AssetModel(BaseDataModel):
     def __init__(self, db_client: object):
        super().__init__(db_client)
        self.collection= db_client

     @classmethod
     async def create_instance(cls, db_client:object):
        instance = cls(db_client=db_client)
        return instance
     
     async def create_asset(self, asset: Asset):
          async with self.db_client() as session:
                      async with session.begin():
                          session.add(asset)
                      await session.commit()
                      await session.refresh(asset)
                      return asset
      
     async def get_all_project_assets(self,asset_project_id: str, asset_type: str):
          async with self.db_client() as session:
                      async with session.begin():
                          stmt= select(Asset).where(
                                Asset.asset_project_id == asset_project_id,
                                Asset.asset_type == asset_type
                                )
                          result = await session.execute(stmt)
                          records = result.scalars().all()
          return records

     
     async def get_asset_by_asset_id(self,asset_project_id: int, asset_id: int):
          """Look an asset up by its PRIMARY KEY — the value /ingest hands back.

          get_asset_by_id below matches on asset_name despite its name, and a name is
          not unique: there is no constraint on asset_name, so uploading the same
          filename twice makes its scalar_one_or_none raise MultipleResultsFound. The
          primary key cannot be ambiguous, so this is the lookup /process prefers.
          """
          async with self.db_client() as session:
               async with session.begin():
                      stmt= select(Asset).where(
                                Asset.asset_project_id == asset_project_id,
                                Asset.asset_id == asset_id
                                )
                      result = await session.execute(stmt)
                      record = result.scalar_one_or_none()
               return record

     async def get_asset_by_id(self,asset_project_id: str, asset_name: str):
          async with self.db_client() as session:
               async with session.begin():
                      stmt= select(Asset).where(
                                Asset.asset_project_id == asset_project_id,
                                Asset.asset_name == asset_name
                                )
                      result = await session.execute(stmt)
                      record = result.scalar_one_or_none()
               return record
                    

     