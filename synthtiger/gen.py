"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import itertools
import inspect
import os
import random
import sys
from copy import deepcopy
from multiprocessing import Process, Queue

import imgaug
import numpy as np
import yaml


def read_template(path, name, config=None):
    path = os.path.abspath(path)
    root = os.path.dirname(path)
    module = os.path.splitext(os.path.basename(path))[0]
    sys.path.append(root)
    template = getattr(__import__(module), name)(config)
    sys.path.remove(root)
    del sys.modules[module]
    return template


def read_config(path):
    with open(path, "r", encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    return config


def generator(
    path,
    name,
    config=None,
    count=None,
    worker=0,
    seed=None,
    retry=True,
    verbose=False,
    tasks=None,
    shard_corpus=True,
    max_attempts=None,
    return_retry_context=False,
    output_root=None,
    save_in_worker=False,
    compact_data=False,
):
    counter = range(count) if count is not None else itertools.count()
    if tasks is None:
        tasks = _task_generator(seed)
    else:
        tasks = iter(tasks)

    if worker > 0:
        task_queue = Queue(maxsize=worker)
        data_queue = Queue(maxsize=worker)
        pre_count = min(worker, count) if count is not None else worker
        post_count = count - pre_count if count is not None else None

        for worker_idx in range(worker):
            _run(
                _worker,
                (
                    path,
                    name,
                    config,
                    output_root,
                    worker_idx,
                    worker,
                    task_queue,
                    data_queue,
                    retry,
                    verbose,
                    shard_corpus,
                    max_attempts,
                    save_in_worker,
                    compact_data,
                ),
            )
        for _ in range(pre_count):
            task_queue.put(next(tasks))

        for idx in counter:
            task_idx, data, retry_context = data_queue.get()
            if post_count is None or idx < post_count:
                task_queue.put(next(tasks))
            if return_retry_context:
                yield task_idx, data, retry_context
            else:
                yield task_idx, data
    else:
        _configure_runtime_determinism()
        template = read_template(path, name, config)

        for _ in counter:
            task = next(tasks)
            task_idx, task_seed, retry_context = _parse_task(task)
            data = _generate(
                template,
                task_seed,
                retry,
                verbose,
                initial_retry_context=retry_context,
                max_attempts=max_attempts,
                task_idx=task_idx,
            )
            if return_retry_context:
                yield task_idx, data, retry_context
            else:
                yield task_idx, data


def get_global_random_states():
    states = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "imgaug": imgaug.random.get_global_rng().state,
    }
    return states


def set_global_random_states(states):
    random.setstate(states["random"])
    np.random.set_state(states["numpy"])
    imgaug.random.get_global_rng().state = states["imgaug"]


def set_global_random_seed(seed=None):
    random.seed(seed)
    np.random.set_state(np.random.RandomState(np.random.MT19937(seed)).get_state())
    imgaug.random.seed(seed)


def _run(func, args):
    proc = Process(target=func, args=args)
    proc.daemon = True
    proc.start()
    return proc


def _configure_runtime_determinism():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import cv2

        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def _task_generator(seed):
    random_generator = random.Random(seed)
    task_idx = -1

    while True:
        task_idx += 1
        task_seed = random_generator.getrandbits(128)
        yield task_idx, task_seed


def _inject_corpus_shard_config(config, shard_index, num_shards):
    if config is None:
        return None

    worker_config = deepcopy(config)
    corpus = worker_config.get("corpus")
    if not isinstance(corpus, dict):
        return worker_config

    args = corpus.get("args")
    if not isinstance(args, list):
        return worker_config

    for arg in args:
        if isinstance(arg, dict):
            arg["shard_index"] = shard_index
            arg["num_shards"] = num_shards

    return worker_config


def _worker(
    path,
    name,
    config,
    output_root,
    worker_idx,
    worker_count,
    task_queue,
    data_queue,
    retry,
    verbose,
    shard_corpus,
    max_attempts,
    save_in_worker,
    compact_data,
):
    _configure_runtime_determinism()
    if shard_corpus:
        worker_config = _inject_corpus_shard_config(config, worker_idx, worker_count)
    else:
        worker_config = config
    if save_in_worker:
        if worker_config is None:
            worker_config = {}
        else:
            worker_config = deepcopy(worker_config)
        worker_config["annotation_output"] = False
    template = read_template(path, name, worker_config)
    worker_tmp_dir = None
    gt_worker_file = None
    fail_worker_file = None
    coords_worker_file = None
    glyph_coords_worker_file = None

    if save_in_worker and output_root is not None:
        template.init_save(output_root)
        worker_tmp_dir = os.path.join(output_root, ".worker_meta")
        os.makedirs(worker_tmp_dir, exist_ok=True)
        gt_worker_path = os.path.join(worker_tmp_dir, f"gt_worker_{worker_idx}.txt")
        fail_worker_path = os.path.join(worker_tmp_dir, f"fail_worker_{worker_idx}.txt")
        gt_worker_file = open(gt_worker_path, "w", encoding="utf-8", buffering=1)
        fail_worker_file = open(fail_worker_path, "w", encoding="utf-8", buffering=1)

        if getattr(template, "coord_output", False):
            coords_worker_path = os.path.join(
                worker_tmp_dir, f"coords_worker_{worker_idx}.txt"
            )
            coords_worker_file = open(
                coords_worker_path, "w", encoding="utf-8", buffering=1
            )
        if getattr(template, "glyph_coord_output", False):
            glyph_coords_worker_path = os.path.join(
                worker_tmp_dir, f"glyph_coords_worker_{worker_idx}.txt"
            )
            glyph_coords_worker_file = open(
                glyph_coords_worker_path, "w", encoding="utf-8", buffering=1
            )

    while True:
        task = task_queue.get()
        task_idx, task_seed, retry_context = _parse_task(task)
        data = _generate(
            template,
            task_seed,
            retry,
            verbose,
            initial_retry_context=retry_context,
            max_attempts=max_attempts,
            task_idx=task_idx,
        )
        if data is not None and save_in_worker and output_root is not None:
            template.save(output_root, data, task_idx)
            image_key = _make_image_key(task_idx)
            gt_worker_file.write(f"{task_idx}\t{image_key}\t{data['label']}\n")
            if coords_worker_file is not None:
                coords, _ = _build_coord_strings(data["bboxes"], data["glyph_bboxes"])
                coords_worker_file.write(f"{task_idx}\t{image_key}\t{coords}\n")
            if glyph_coords_worker_file is not None:
                _, glyph_coords = _build_coord_strings(
                    data["bboxes"], data["glyph_bboxes"]
                )
                glyph_coords_worker_file.write(
                    f"{task_idx}\t{image_key}\t{glyph_coords}\n"
                )
            if compact_data:
                data = {"status": "success"}
        elif data is None and save_in_worker and output_root is not None:
            label = retry_context.get("fixed_label", "")
            fail_worker_file.write(
                f"{task_idx}\t{label}\tmax_attempts_exceeded\n"
            )
        data_queue.put((task_idx, data, retry_context))


def _make_image_key(task_idx):
    shard = str(task_idx // 10000)
    return os.path.join("images", shard, f"{task_idx}.jpg")


def _build_coord_strings(bboxes, glyph_bboxes):
    coords = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
    coords = "\t".join([",".join(map(str, map(int, coord))) for coord in coords])
    glyph_coords = [[x, y, x + w, y + h] for x, y, w, h in glyph_bboxes]
    glyph_coords = "\t".join(
        [",".join(map(str, map(int, coord))) for coord in glyph_coords]
    )
    return coords, glyph_coords


def _parse_task(task):
    if isinstance(task, dict):
        return task["task_idx"], task["task_seed"], dict(task.get("retry_context", {}))

    if len(task) == 2:
        task_idx, task_seed = task
        return task_idx, task_seed, {}
    if len(task) == 3:
        task_idx, task_seed, retry_context = task
        if retry_context is None:
            retry_context = {}
        return task_idx, task_seed, dict(retry_context)

    raise RuntimeError("Task must be (task_idx, task_seed[, retry_context])")


def _generate(
    template,
    seed,
    retry,
    verbose,
    initial_retry_context=None,
    max_attempts=None,
    task_idx=None,
):
    states = get_global_random_states()
    set_global_random_seed(seed)
    data = None
    retry_context = dict(initial_retry_context or {})
    attempts = 0
    if (
        "fixed_label" in retry_context
        and hasattr(template, "corpus")
        and hasattr(template.corpus, "_update_sampling_orders")
    ):
        # Keep fallback corpus sampling deterministic per task,
        # regardless of worker count or prior tasks.
        template.corpus._update_sampling_orders()
    supports_retry_context = (
        "retry_context" in inspect.signature(template.generate).parameters
    )

    while True:
        attempts += 1
        try:
            if supports_retry_context:
                data = template.generate(retry_context=retry_context)
            else:
                data = template.generate()
        except:
            if verbose:
                task_text = "unknown" if task_idx is None else str(task_idx)
                if retry and (max_attempts is None or attempts < max_attempts):
                    if max_attempts is None:
                        print(
                            f"Task {task_text} failed, retrying "
                            f"(attempt {attempts})."
                        )
                    else:
                        print(
                            f"Task {task_text} failed, retrying "
                            f"({attempts}/{max_attempts})."
                        )
                else:
                    if max_attempts is None:
                        print(
                            f"Task {task_text} failed permanently "
                            f"after {attempts} attempts."
                        )
                    else:
                        print(
                            f"Task {task_text} failed permanently "
                            f"({attempts}/{max_attempts})."
                        )
            if retry and (max_attempts is None or attempts < max_attempts):
                continue
            data = None
        break

    set_global_random_states(states)
    return data
