from typing import List, Optional
from pydantic import BaseModel

class MetadataFeature(BaseModel):
    class_name: str
    file_path: str

class FeatureData(BaseModel):
    id: str
    vector: List[float]
    metadata: MetadataFeature
