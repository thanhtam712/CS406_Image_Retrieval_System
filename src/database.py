import os
from dotenv import load_dotenv
from schemas import FeatureData
from pymilvus import MilvusClient, DataType

load_dotenv()

# print(os.getenv("URI_DATABASE"))
# print(os.getenv("TOKEN_DATABASE"))

client = MilvusClient(
    # uri=os.getenv("URI_DATABASE"),
    uri="http://192.168.20.150:19530",
    # token=os.getenv("TOKEN_DATABASE")
    token="root:Milvus"
)

def create_collection_if_not_exists(collection_name: str, dim: int):
    if not client.has_collection(collection_name):
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=255)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128}
        )
        
        client.create_collection(collection_name, dim=dim, schema=schema)
        client.create_index(collection_name, index_params)
        print(f"Collection {collection_name} created.")

def insert_data(data: list[FeatureData], collection_name: str ="image_retrieval_cs406"):
    if not data:
        return
    
    dim = len(data[0].vector)
    create_collection_if_not_exists(collection_name, dim)

    res = client.insert(
        collection_name=collection_name,
        data=[d.dict() for d in data]
    )

    return res

def delete_data(ids: list[str], collection_name: str ="image_retrieval_cs406"):
    res = client.delete(
        collection_name=collection_name,
        ids=ids
    )

    return res

def search_vector(vector: list[float], top_k: int =10, collection_name: str ="image_retrieval_cs406"):
    client.load_collection(collection_name)
    res = client.search(
        collection_name=collection_name,
        data=[vector],
        anns_field="vector",
        search_params={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["metadata"]
    )

    return res
