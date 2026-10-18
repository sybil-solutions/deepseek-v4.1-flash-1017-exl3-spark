"""sha256 of every downloaded file against the LFS oid Hugging Face records for it (the checkpoint at its pinned
revision). Prints "bad [...]" with the mismatches; finish.sh requires "bad []".

  python -I verify.py [/data/DeepSeek-V4.1-Flash-1017]
"""
import hashlib
import os
import sys

from huggingface_hub import HfApi

REPO, REV = "deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170"
root = sys.argv[1] if len(sys.argv) > 1 else "/data/DeepSeek-V4.1-Flash-1017"
bad = []
for f in HfApi().list_repo_tree(REPO, revision=REV, recursive=True):
    if getattr(f, "lfs", None) is None:
        continue
    h = hashlib.sha256()
    with open(os.path.join(root, f.path), "rb") as fh:
        while b := fh.read(64 << 20):
            h.update(b)
    ok = h.hexdigest() == f.lfs.sha256
    print(f.path, "ok" if ok else "MISMATCH", flush=True)
    if not ok:
        bad.append(f.path)
print("bad", bad)
