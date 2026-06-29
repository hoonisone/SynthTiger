"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import argparse
import os
import pprint
import random
import time

import synthtiger


def _iter_component_labels(component):
    if not hasattr(component, "_counts") or not hasattr(component, "_get_text"):
        raise RuntimeError(
            "Corpus component does not support deterministic corpus iteration"
        )

    for key, count in enumerate(component._counts):
        for idx in range(count):
            yield component._get_text(key, idx)


def _iter_corpus_labels(template):
    corpus = getattr(template, "corpus", None)
    if corpus is None:
        raise RuntimeError("Template has no corpus component")

    if hasattr(corpus, "components"):
        weights = getattr(corpus, "weights", [1] * len(corpus.components))
        for component_idx, component in enumerate(corpus.components):
            if component_idx < len(weights) and weights[component_idx] <= 0:
                continue
            yield from _iter_component_labels(component)
        return

    yield from _iter_component_labels(corpus)


def _count_corpus_labels(template):
    corpus = getattr(template, "corpus", None)
    if corpus is None:
        raise RuntimeError("Template has no corpus component")

    if hasattr(corpus, "components"):
        weights = getattr(corpus, "weights", [1] * len(corpus.components))
        total = 0
        for component_idx, component in enumerate(corpus.components):
            if component_idx < len(weights) and weights[component_idx] <= 0:
                continue
            if not hasattr(component, "_counts"):
                raise RuntimeError(
                    "Corpus component does not support deterministic corpus iteration"
                )
            total += sum(component._counts)
        return total

    if not hasattr(corpus, "_counts"):
        raise RuntimeError("Corpus component does not support deterministic iteration")
    return sum(corpus._counts)


def _planned_task_generator_from_corpus(template, seed):
    random_generator = random.Random(seed)
    for task_idx, label in enumerate(_iter_corpus_labels(template)):
        task_seed = random_generator.getrandbits(128)
        shard = str(task_idx // 10000)
        image_key = os.path.join("images", shard, f"{task_idx}.jpg")
        retry_context = {
            "fixed_label": label,
            "label_attempt": 0,
            "image_key": image_key,
        }
        yield task_idx, task_seed, retry_context


def _read_sorted_rows(path, min_fields):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fp:
        for raw in fp:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            parts = raw.split("\t")
            if len(parts) < min_fields:
                continue
            rows.append(parts)
    rows.sort(key=lambda parts: int(parts[0]))
    return rows


def _merge_worker_temp_files(output_root, worker_count):
    tmp_dir = os.path.join(output_root, ".worker_meta")
    if not os.path.isdir(tmp_dir):
        return

    gt_rows = []
    fail_rows = []
    coords_rows = []
    glyph_coords_rows = []

    for worker_idx in range(worker_count):
        gt_rows.extend(
            _read_sorted_rows(
                os.path.join(tmp_dir, f"gt_worker_{worker_idx}.txt"),
                min_fields=3,
            )
        )
        fail_rows.extend(
            _read_sorted_rows(
                os.path.join(tmp_dir, f"fail_worker_{worker_idx}.txt"),
                min_fields=3,
            )
        )
        coords_rows.extend(
            _read_sorted_rows(
                os.path.join(tmp_dir, f"coords_worker_{worker_idx}.txt"),
                min_fields=3,
            )
        )
        glyph_coords_rows.extend(
            _read_sorted_rows(
                os.path.join(tmp_dir, f"glyph_coords_worker_{worker_idx}.txt"),
                min_fields=3,
            )
        )

    gt_rows.sort(key=lambda parts: int(parts[0]))
    fail_rows.sort(key=lambda parts: int(parts[0]))
    coords_rows.sort(key=lambda parts: int(parts[0]))
    glyph_coords_rows.sort(key=lambda parts: int(parts[0]))

    with open(os.path.join(output_root, "gt.txt"), "w", encoding="utf-8") as fp:
        for _, image_key, label, *rest in gt_rows:
            del rest
            fp.write(f"{image_key}\t{label}\n")

    with open(os.path.join(output_root, "fail_case.txt"), "w", encoding="utf-8") as fp:
        fp.write("task_idx\tlabel\treason\n")
        for task_idx, label, reason, *rest in fail_rows:
            del rest
            fp.write(f"{task_idx}\t{label}\t{reason}\n")

    if coords_rows:
        with open(os.path.join(output_root, "coords.txt"), "w", encoding="utf-8") as fp:
            for _, image_key, coords, *rest in coords_rows:
                del rest
                fp.write(f"{image_key}\t{coords}\n")

    if glyph_coords_rows:
        with open(
            os.path.join(output_root, "glyph_coords.txt"), "w", encoding="utf-8"
        ) as fp:
            for _, image_key, glyph_coords, *rest in glyph_coords_rows:
                del rest
                fp.write(f"{image_key}\t{glyph_coords}\n")

    for name in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, name))
    os.rmdir(tmp_dir)


def run(args):
    config = None
    if args.config is not None:
        config = synthtiger.read_config(args.config)

    pprint.pprint(config)

    synthtiger.set_global_random_seed(args.seed)
    template = synthtiger.read_template(args.script, args.name, config)
    corpus_total = _count_corpus_labels(template)
    if args.count is None:
        planned_count = corpus_total
        planned_tasks = _planned_task_generator_from_corpus(template, args.seed)
    else:
        planned_count = args.count
        if planned_count > corpus_total:
            raise RuntimeError(
                f"count ({planned_count}) is larger than corpus size ({corpus_total})"
            )
        planned_tasks = _planned_task_generator_from_corpus(template, args.seed)

    max_attempts = int(getattr(template, "max_attempts_per_label", 3))
    if max_attempts < 1:
        raise RuntimeError("max_attempts_per_label must be >= 1")

    generator = synthtiger.generator(
        args.script,
        args.name,
        config=config,
        count=planned_count,
        worker=args.worker,
        seed=args.seed,
        retry=True,
        verbose=args.verbose,
        tasks=planned_tasks,
        shard_corpus=False,
        max_attempts=max_attempts,
        return_retry_context=True,
        output_root=args.output,
        save_in_worker=(args.worker > 0 and args.output is not None),
        compact_data=(args.worker > 0 and args.output is not None),
    )

    worker_direct_save = args.worker > 0 and args.output is not None
    fail_case_file = None
    if args.output is not None and not worker_direct_save:
        template.init_save(args.output)
        fail_case_path = os.path.join(args.output, "fail_case.txt")
        fail_case_file = open(fail_case_path, "w", encoding="utf-8")
        fail_case_file.write("task_idx\tlabel\treason\n")

    generated = 0
    dropped = 0
    next_task_idx = 0
    pending = {}

    for task_idx, data, retry_context in generator:
        pending[task_idx] = (data, retry_context)

        while next_task_idx in pending:
            ordered_data, ordered_retry_context = pending.pop(next_task_idx)
            if ordered_data is None:
                dropped += 1
                print(f"Dropped task {next_task_idx} after {max_attempts} attempts")
                if worker_direct_save:
                    pass
                elif fail_case_file is not None:
                    label = ordered_retry_context.get("fixed_label", "")
                    fail_case_file.write(
                        f"{next_task_idx}\t{label}\tmax_attempts_exceeded\n"
                    )
            else:
                if args.output is not None:
                    if worker_direct_save:
                        pass
                    else:
                        template.save(args.output, ordered_data, generated)
                generated += 1
                print(f"Generated {generated} data (task {next_task_idx})")
            next_task_idx += 1

    if args.output is not None:
        if worker_direct_save:
            _merge_worker_temp_files(args.output, args.worker)
        else:
            fail_case_file.close()
            template.end_save(args.output)

    print(
        f"Done. planned={planned_count}, generated={generated}, dropped={dropped}, "
        f"max_attempts={max_attempts}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        type=str,
        help="Directory path to save data.",
    )
    parser.add_argument(
        "-c",
        "--count",
        metavar="NUM",
        type=int,
        default=None,
        help="Number of planned tasks. If omitted, iterate entire corpus once.",
    )
    parser.add_argument(
        "-w",
        "--worker",
        metavar="NUM",
        type=int,
        default=0,
        help="Number of workers. If 0, It generates data in the main process. [default: 0]",
    )
    parser.add_argument(
        "-s",
        "--seed",
        metavar="NUM",
        type=int,
        default=42,
        help="Random seed. [default: 42]",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print error messages while generating data.",
    )
    parser.add_argument(
        "script",
        metavar="SCRIPT",
        type=str,
        help="Script file path.",
    )
    parser.add_argument(
        "name",
        metavar="NAME",
        type=str,
        help="Template class name.",
    )
    parser.add_argument(
        "config",
        metavar="CONFIG",
        type=str,
        nargs="?",
        help="Config file path.",
    )
    args = parser.parse_args()

    pprint.pprint(vars(args))

    return args


def main():
    start_time = time.time()
    args = parse_args()
    run(args)
    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds elapsed")


if __name__ == "__main__":
    main()
