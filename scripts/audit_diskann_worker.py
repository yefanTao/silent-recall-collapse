"""Run DiskANN without loading Faiss's conflicting OpenMP runtime."""
import json
import sys
import numpy as np
import diskannpy
job = json.load(open(sys.argv[1], encoding="utf-8"))
if job["operation"] == "build":
    kwargs = job["kwargs"]
    kwargs["data"] = np.load(job["input"])
    diskannpy.build_memory_index(**kwargs)
else:
    index = diskannpy.StaticMemoryIndex(**job["index"])
    result = index.batch_search(np.load(job["input"]), **job["kwargs"])
    np.savez(job["output"], identifiers=result.identifiers, distances=result.distances)
