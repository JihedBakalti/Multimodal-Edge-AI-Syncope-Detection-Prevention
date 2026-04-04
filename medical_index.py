import os
import sys
import warnings
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

load_dotenv()



# ======================================================
# 1. CONFIGURATION
# ======================================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "hannibal_kb"
EMBEDDING_SIZE = 384  # all-MiniLM-L6-v2

# ======================================================
# 2. LOAD DATA
# ======================================================

documents = []
dataset_dir = "dataset"

if not os.path.exists(dataset_dir):
    print(f"[WARN] Directory '{dataset_dir}' not found!")
else:
    print(f"[INFO] Scanning '{dataset_dir}' for documents...")
    for filename in os.listdir(dataset_dir):
        filepath = os.path.join(dataset_dir, filename)
        
        # Process Text Files
        if filename.lower().endswith(".txt"):
            try:
                print(f"  [TXT] Loading {filename}...")
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                    
                    # Custom splitting for Medical Documents
                    if "=== DOCUMENT" in text:
                        # Split by "=== DOCUMENT" and ignore the first empty chunk if any
                        raw_chunks = text.split("=== DOCUMENT")
                        chunks = []
                        for chunk in raw_chunks:
                            if not chunk.strip():
                                continue
                            
                            # Re-add header for context if needed, or just clean up
                            # extracting the title line
                            lines = chunk.strip().split('\n')
                            if not lines: continue
                            
                            # The first line is likely "X ===" or "X ===\nTITLE"
                            # Let's just keep the whole chunk text as the content
                            # We prepended "DOCUMENT" so the split consumed it.
                            # We can reconstruct a clean document ID or just use the text.
                            
                            content = "DOCUMENT " + chunk.strip() # Re-add the tag for context
                            chunks.append(content)
                            
                        print(f"    - Custom Split: Found {len(chunks)} documents.")
                        
                    else:
                        # Fallback to double newline
                        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
                        print(f"    - Standard Split: Found {len(chunks)} chunks.")

                    for chunk in chunks:
                        documents.append({
                            "content": chunk,
                            "source": filename
                        })
            except Exception as e:
                print(f"    [ERROR] Failed to load {filename}: {e}")

        # Process PDF Files
        elif filename.lower().endswith(".pdf"):
            try:
                import pypdf
                print(f"  [PDF] Loading {filename}...")
                reader = pypdf.PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
                
                # Chunking
                chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
                for chunk in chunks:
                    if len(chunk) > 50: # Filter small noise
                        documents.append({
                            "content": chunk,
                            "source": filename
                        })
                print(f"    - Loaded {len(chunks)} chunks.")
            except Exception as e:
                print(f"    [ERROR] Failed to load {filename}: {e}")

if not documents:
    print("[ERROR] No documents to index. Exiting.")
    sys.exit(1)

# ======================================================
# 3. EMBEDDING MODEL
# ======================================================

print("[INFO] Loading local embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("[INFO] Model loaded.\n")

def get_embedding(text):
    return model.encode(text).tolist()

# ======================================================
# 4. CONNECT TO QDRANT
# ======================================================

print(f"[INFO] Connecting to Qdrant at {QDRANT_HOST}...")
if QDRANT_API_KEY:
    client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(url=QDRANT_HOST)

# Recreate collection
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBEDDING_SIZE, distance="Cosine")
)
print(f"[INFO] Collection '{COLLECTION_NAME}' created/reset.\n")

# ======================================================
# 5. INDEX DATA
# ======================================================

points = []
print("[INFO] Indexing...")

for idx, doc in enumerate(documents):
    vector = get_embedding(doc["content"])
    points.append(PointStruct(
        id=idx,
        vector=vector,
        payload={"content": doc["content"], "source": doc["source"]}
    ))

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"[INFO] Indexed {len(points)} documents into '{COLLECTION_NAME}'.\n")

# ======================================================
# 6. TEST
# ======================================================

query = "Tell me about the Battle of Cannae"
print(f"[TEST] Query: '{query}'")
q_vec = get_embedding(query)
results = client.query_points(collection_name=COLLECTION_NAME, query=q_vec, limit=1).points

if results:
    print(f"[RESULT] {results[0].payload['content'][:200]}...")
else:
    print("[INFO] No results found.")
