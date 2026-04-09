import os
import re
import csv

def parse_input_log(args, tuning_info):
    # save unique lines in hipblaslt log
    unique_log_name = args.output_path + '/unique_' + args.input_file.split('/')[-1]
    f_out = open(unique_log_name, 'w')
    max_gemms = getattr(args, "max_gemms", -1)
    unique_count = 0

    with open(args.input_file, 'r') as f:
        for line in f:
            if 'hipblaslt-bench' not in line:
                continue

            if line in tuning_info:
                tuning_info[line]['count'] += 1
                continue
            else:
                if max_gemms >= 0 and unique_count >= max_gemms:
                    continue
                tuning_info[line] = {}
                tuning_info[line]['count'] = 1
                f_out.write(line)
                unique_count += 1
            matches = re.findall(r"-(m|n|k)\s+(\d+)", line)
            matches.extend(re.findall(r"--(lda|ldb|ldc|ldd)\s+(\d+)", line))
            matches.extend(re.findall(r"--(a_type|b_type|c_type|d_type)\s+(\S+)", line))
            tuning_info[line].update({key: value for key, value in matches})

    return unique_log_name

def resolve_bench_command(args):
    bench_path = getattr(args, "bench_path", "")
    if bench_path:
        return bench_path

    env_path = os.environ.get("HIPBLASLT_BENCH_PATH", "")
    if env_path:
        return env_path

    return "hipblaslt-bench"

def resolve_bench_library_dir(args):
    bench_library_dir = getattr(args, "bench_library_dir", "")
    if bench_library_dir:
        return bench_library_dir

    env_dir = os.environ.get("HIPBLASLT_LIBRARY_DIR", "")
    if env_dir:
        return env_dir

    bench_command = resolve_bench_command(args)
    if os.path.sep not in bench_command:
        return ""

    bench_dir = os.path.dirname(os.path.abspath(bench_command))
    candidates = [
        os.path.join(os.path.dirname(bench_dir), "library"),
        os.path.join(os.path.dirname(os.path.dirname(bench_dir)), "library"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    return ""

def resolve_tensile_libpath(args):
    tensile_libpath = getattr(args, "tensile_libpath", "")
    if tensile_libpath:
        return tensile_libpath

    return os.environ.get("HIPBLASLT_TENSILE_LIBPATH", "")

def build_runtime_env(args):
    env = os.environ.copy()

    bench_library_dir = resolve_bench_library_dir(args)
    if bench_library_dir:
        existing_ld_library_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            bench_library_dir
            if not existing_ld_library_path
            else f"{bench_library_dir}:{existing_ld_library_path}"
        )

    tensile_libpath = resolve_tensile_libpath(args)
    if tensile_libpath:
        env["HIPBLASLT_TENSILE_LIBPATH"] = tensile_libpath

    return env

def parse_hipblaslt_output(output, line, tuning_info, mode):
    outputs = output.split('\n')
    print(output)
    if mode == 'baseline':
        latency = float(outputs[-5].split(',')[-1])
    else:
        latency = float(outputs[-5].split(',')[-2])
    solution_idx = outputs[-4].split(':')[-1]
    if mode == 'baseline':
        tuning_info[line].update({"baseline_latency(us)": latency})
        tuning_info[line].update({"baseline_solution_idx": solution_idx})
    elif mode == 'tuning':
        tuning_info[line].update({"tuned_latency(us)": latency})
        tuning_info[line].update({"tuned_solution_idx": solution_idx})
        ratio = tuning_info[line]['baseline_latency(us)'] / latency * 100
        ratio = round(ratio, 2)
        tuning_info[line].update({"baseline/tuned": f"{ratio}%"})

def export_csv(input_file, tuning_info, args):
    fieldnames = ['m', 'n', 'k', 'lda', 'ldb', 'ldc', 'ldd', 'a_type', 'b_type', 'c_type', 'd_type', 'count', 'baseline_latency(us)', 'baseline_solution_idx', 'tuned_latency(us)', 'tuned_solution_idx', 'baseline/tuned']
    tuning_info_list = []

    with open(input_file, 'r') as f:
        for line in f:
            tuning_info_list.append(tuning_info[line])

    with open(args.output_path + '/tuning_result.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tuning_info_list)

def dynamic_iters(input_cmd, cold_iters, iters, tuning_info):
    if cold_iters == -1 or iters == -1:
        mnk = (float(tuning_info[input_cmd]['m']) * float(tuning_info[input_cmd]['n']) * float(tuning_info[input_cmd]['k'])) / (1024*1024)
        if mnk <= 4000:
            cold_iters, iters = 200, 100
        elif mnk <= 80000:
            cold_iters, iters = 50, 20
        else:
            cold_iters, iters = 10, 2
    return cold_iters, iters

def convert_command(input_cmd, args, tuning_info, mode):
    requested_solution = args.requested_solution
    iters = args.iters
    cold_iters = args.cold_iters
    bench_command = resolve_bench_command(args)

    output_cmd = input_cmd.strip()
    cold_iters, iters = dynamic_iters(input_cmd, cold_iters, iters, tuning_info)
    output_cmd = re.sub(r"^\s*hipblaslt-bench\b", bench_command, output_cmd, count=1)

    output_cmd = re.sub(f'--algo_method\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'--solution_index\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'--requested_solution\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'--cold_iters\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'--iters\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'--rotating\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'--aux_type\s+\S+', '', output_cmd)
    output_cmd = re.sub(f'\s{2,}', '', output_cmd)

    output_cmd = output_cmd + f" --algo_method heuristic"
    output_cmd = output_cmd + f" --cold_iters {cold_iters}"
    output_cmd = output_cmd + f" --iters {iters}"
    output_cmd = output_cmd + f" --rotating 512 --flush --print_kernel_info"
    if mode == "baseline":
        output_cmd = output_cmd + f" --requested_solution 1"
    else:
        output_cmd = output_cmd + f" --requested_solution {requested_solution}"
        output_cmd = output_cmd + f" --skip_slow_solution_ratio 0.8"
        if (args.swizzleA == True) or (args.swizzleB == True):
            output_cmd = re.sub("--transA\s+\S+", "--transA T", output_cmd)
            if args.swizzleA == True:
                output_cmd = output_cmd + f" --swizzleA"
            if args.swizzleB == True:
                output_cmd = output_cmd + f" --swizzleB"

    return output_cmd
