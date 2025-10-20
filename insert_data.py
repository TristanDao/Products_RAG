from embedding import EmbeddingModel
from vector_db import VectorDatabase
import uuid
from typing import Set
import pandas as pd


def build_combine_row(row):
    combine = f"Tên sản phẩm: {row['title']}\n"
    combine += f"Mô tả: {row['product_specs']}\n"
    combine += f"Giá: {row['current_price']}\n"
    combine += f"Ưu đãi: {row['product_promotion']}\n"
    combine += f"Màu sắc: {row['color_options']}\n"
    return combine


def get_existing_titles(vector_db: VectorDatabase, collection_name: str) -> Set[str]:
    existing_titles = set()
    try:
        if vector_db.db_type == "mongodb":
            db = vector_db.client.get_database("vector_db")
            collection = db[collection_name]
            existing_docs = collection.find({}, {"title": 1})
            existing_titles = {doc["title"] for doc in existing_docs}

        elif vector_db.db_type == "chromadb":
            collection = vector_db.client.get_collection(name=collection_name)
            results = collection.get()
            if results and 'metadatas' in results:
                existing_titles = {meta.get('title') for meta in results['metadatas'] if meta.get('title')}

        elif vector_db.db_type == "qdrant":
            scroll_result = vector_db.client.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True
            )
            for point in scroll_result[0]:
                if 'title' in point.payload:
                    existing_titles.add(point.payload['title'])

        elif vector_db.db_type == "supabase":
            response = vector_db.client.table(collection_name).select("title").execute()
            existing_titles = {row['title'] for row in response.data}
    except Exception as e:
        print(f"Note: {e}")
    return existing_titles


def insert_batch(vector_db: VectorDatabase, batch_data: list, collection_name: str):
    try:
        vector_db.insert(data=batch_data, collection_name=collection_name)
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    CSV_FILE = "hoanghamobile.csv"
    COLLECTION_NAME = "products"
    DB_TYPE = "mongodb"
    EMBEDDING_PROVIDER = "huggingface"
    # EMBEDDING_MODEL = "gemini-embedding-001"
    BATCH_SIZE = 50
    RESET_COLLECTION = True

    print("=" * 70)
    print(f"{'DATA INSERTION TO VECTOR DATABASE':^70}")
    print("=" * 70)
    print(f"Database Type: {DB_TYPE}")
    print(f"Embedding Provider: {EMBEDDING_PROVIDER}")
    # print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Collection Name: {COLLECTION_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 70 + "\n")

    df = pd.read_csv(CSV_FILE)
    df['information'] = df.apply(build_combine_row, axis=1)

    #Initialize Vector Database
    print(f"Connecting to {DB_TYPE} database...")
    try:
        vector_db = VectorDatabase(db_type=DB_TYPE)
        print(f"Connected successfully\n")
    except Exception as e:
        print(f"Connection failed {e}")
        return
    
    #Initialize Embedding Model
    # print(f"Initializing {EMBEDDING_MODEL} embedding model ...")
    try:
        embedding = EmbeddingModel(provider=EMBEDDING_PROVIDER)
        vector_size = embedding.get_vector_size()
        print(f"Model loaded (vector size: {vector_size})\n")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    if RESET_COLLECTION:
        print(f"Resetting collection: {COLLECTION_NAME}")
        try:
            vector_db.delete_collection(COLLECTION_NAME)
            print(f"Collection deleted")
        except Exception as e:
            print(f"Note (delete): {e}")

    try:
        vector_db.create_collection(COLLECTION_NAME, vector_size=vector_size)
        print(f"Collection created: {COLLECTION_NAME}\n")
    except Exception as e:
        print(f"Note (create): {e}\n")

    print("Checking for existing documents...")
    existing_titles = get_existing_titles(vector_db, COLLECTION_NAME)
    print(f"Found {len(existing_titles)} existing documents\n")

    print("=" * 70)
    print(f"{'INSERTING DOCUMENTS':^70}")
    print("=" * 70 + "\n")

    inserted_count = 0
    skipped_count = 0
    error_count = 0
    batch_data = []
    batch_num = 1

    for index, row in df.iterrows():
        title = row['title']

        if title in existing_titles:
            skipped_count += 1
            if skipped_count % 50 == 0:
                print(f"Skipped {skipped_count} existing documents...")
            continue

        doc = row['information']

        try:
            embedding_vector = embedding.encode_single(doc)
            doc_id = hash(title) % (10 ** 10) if vector_db.db_type == "qdrant" else str(uuid.uuid4())

            document_data = {
                "id": doc_id,
                "title": title,
                "embedding": embedding_vector,
                "information": doc,
                "metadata": {
                    "title": title,
                    "index": index,
                    "source": CSV_FILE
                }
            }

            batch_data.append(document_data)

            if len(batch_data) >= BATCH_SIZE:
                success, error = insert_batch(vector_db, batch_data, COLLECTION_NAME)
                if success:
                    inserted_count += len(batch_data)
                    existing_titles.update([data['title'] for data in batch_data])
                    print(f"Batch {batch_num}: Inserted {len(batch_data)} documents "
                          f"(Total: {inserted_count}/{len(df) - skipped_count})")
                else:
                    error_count += len(batch_data)
                    print(f"Batch {batch_num}: Batch - {error}")

                batch_data = []
                batch_num += 1

        except Exception as e:
            print(f"Error processing '{title}': {e}")
            error_count += 1
            continue

    if batch_data:
        success, error = insert_batch(vector_db, batch_data, COLLECTION_NAME)
        if success:
            inserted_count += len(batch_data)
            print(f"Final batch: Inserted {len(batch_data)} documents "
                  f"(Total: {inserted_count}/{len(df) - skipped_count})")
        else:
            error_count += len(batch_data)
            print(f"Final batch: Failed - {error}")

    print("\n" + "=" * 70)
    print(f"{'SUMMARY':^70}")
    print("=" * 70)
    print(f"Total records in CSV:      {len(df):>6}")
    print(f"Already existed:           {skipped_count:>6}")
    print(f"Successfully inserted:     {inserted_count:>6}")
    print(f"Errors:                    {error_count:>6}")
    print("-" * 70)
    print(f"Final total in database:   {len(existing_titles) + inserted_count:>6}")
    print("=" * 70 + "\n")

    try:
        info = vector_db.get_collection_info(COLLECTION_NAME)
        print(f"📊 Collection '{COLLECTION_NAME}' info:")
        print(f"   Total documents: {info.get('count', 'N/A')}")
        if 'vector_size' in info:
            print(f"   Vector size: {info['vector_size']}")
    except Exception as e:
        print(f"Note: Could not retrieve collection info - {e}")

    print()
    if inserted_count > 0:
        print(f"✅ Successfully inserted {inserted_count} new documents!")
    elif skipped_count > 0:
        print(f"⏭️  All {skipped_count} documents already exist in database.")
    else:
        print("⚠️  No documents were processed.")

    print("\n" + "=" * 70)
    print("✨ Process completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()