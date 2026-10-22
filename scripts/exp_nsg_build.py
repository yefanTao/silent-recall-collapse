"""Measure clean and duplicate-injected NSG build times in bounded child processes."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time


def build(path, n, d, ratio, seed, threads):
    import faiss
    import numpy as np
    faiss.omp_set_num_threads(threads)
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, d)).astype('float32')
    faiss.normalize_L2(base)
    sink = rng.standard_normal((1, d)).astype('float32')
    faiss.normalize_L2(sink)
    x = np.concatenate([base, np.repeat(sink, int(n * ratio), axis=0)])
    start = time.perf_counter()
    index = faiss.IndexNSGFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.build_type = 1
    index.add(x)
    Path(path).write_text(json.dumps({'status': 'complete', 'seconds': time.perf_counter()-start,
                                    'n': len(x), 'faiss': faiss.__version__}), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-base', type=int, default=20000)
    parser.add_argument('--dimension', type=int, default=128)
    parser.add_argument('--sink-ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--timeout-seconds', type=float, default=60)
    args = parser.parse_args()
    if args.n_base < 100 or args.dimension < 1 or args.sink_ratio <= 0 or args.timeout_seconds <= 0:
        parser.error('Require n-base >= 100 and positive dimension, sink ratio and timeout')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty or new')
    args.output.mkdir(parents=True, exist_ok=True)
    context = mp.get_context('spawn')
    rows = []
    for name, ratio in [('clean', 0.0), ('duplicate', args.sink_ratio)]:
        path = args.output / f'{name}.json'
        process = context.Process(target=build, args=(str(path), args.n_base, args.dimension,
                                                    ratio, args.seed, args.threads))
        process.start()
        process.join(args.timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            row = {'status': 'timeout', 'limit_seconds': args.timeout_seconds}
        elif process.exitcode != 0:
            row = {'status': 'failed', 'exit_code': process.exitcode}
        else:
            row = json.loads(path.read_text())
        row.update(case=name, sink_base_ratio=ratio)
        rows.append(row)
        print(json.dumps(row), flush=True)
    config = vars(args).copy()
    config['output'] = str(args.output)
    (args.output / 'nsg_build.json').write_text(json.dumps({'config': config, 'rows': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
