import argparse
import sys


sys.path.append(".")
from base_classes import ImageToImageBenchmark, TurboImageToImageBenchmark  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        choices=[
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            "stabilityai/sdxl-turbo",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_cpu_offload", action="store_true")
    parser.add_argument("--run_compile", action="store_true")
    args = parser.parse_args()

    benchmark_pipe = ImageToImageBenchmark(args) if "turbo" not in args.ckpt else TurboImageToImageBenchmark(args)
    benchmark_pipe.benchmark(args)
