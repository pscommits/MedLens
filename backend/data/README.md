# `backend/data/` — Persistent data layer

This folder is where the **ChromaDB vector store** lives.

## Required layout

```
backend/data/
└── chroma_store/        ← your existing chroma_store goes here
    ├── chroma.sqlite3
    └── <uuid-folders>/
```

## How to set it up

You already have a populated `chroma_store` from your earlier work.
Copy the **entire folder** here:

```bash
# from the project root
cp -r /path/to/your/existing/chroma_store backend/data/chroma_store
```

Or on Windows (PowerShell):

```powershell
Copy-Item -Recurse C:\path\to\your\existing\chroma_store .\backend\data\chroma_store
```

## Verifying it works

After starting the backend (`./run.sh` or `run.bat`), watch the logs.
On the first analysis request you should see:

```
[retrieval_agent] Opening ChromaDB at /.../backend/data/chroma_store ...
[retrieval_agent] Collection 'medical_knowledge' ready (N passages).
```

If you see a `FileNotFoundError`, your `chroma_store` is not in the expected
place. Either move it here, or set `CHROMA_PATH=/absolute/path` in `backend/.env`.

## Collection name

The retrieval agent looks for a collection named **`medical_knowledge`**
(matches your existing setup). If yours is named differently, set:

```
CHROMA_COLLECTION=your_collection_name
```

in `backend/.env`.

## Rebuilding from scratch

If you need to rebuild the index (e.g. after losing the original), see
`scripts/build_index.py` in the project root.
