import shutil
import time
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from multiprocessing import TimeoutError
from multiprocessing.pool import Pool
from typing import Any, Optional, Dict
import tyro
import collections
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, Future
import multiprocessing

from gsplat.strategy import DefaultStrategy
from simple_trainer import Config, main as run_gsplat_training


# --- Custom Multiprocessing Pool ---

class NonDaemonPool(Pool):
    """A process pool that ensures worker processes are non-daemonic."""
    def Process(self, *args, **kwds):
        proc = super().Process(*args, **kwds)
        try:
            proc.daemon = False
        except Exception:
            for attr in ("_daemonic", "_daemon", "_is_daemon"):
                try:
                    setattr(proc, attr, False)
                    break
                except Exception:
                    continue
        return proc

# --- Staging Manager ---
class StagingManager:
    """Manages the staging of scan data to a temporary directory."""
    def __init__(self, tmp_dir: Path, logger: logging.Logger):
        self.base_tmp_dir = tmp_dir
        self.logger = logger
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count(), thread_name_prefix="StagingThread")
        
        self.logger.info(f"Initializing staging manager at {self.base_tmp_dir}")
        self.base_tmp_dir.mkdir(parents=True, exist_ok=True)

    def _stage_scan_task(self, scan_path: Path, staging_subdir: Path) -> Path:
        """The actual task of copying files for a single scan."""
        staged_scan_path = staging_subdir / scan_path.name
        
        if staged_scan_path.exists():
            shutil.rmtree(staged_scan_path)
        staged_scan_path.mkdir(parents=True)

        self.logger.info(f"Staging '{scan_path.name}' to '{staged_scan_path}'...")
        
        # Directories to copy from the source scan folder.
        dirs_to_copy = ["images", "masks", "sparse", "overlays"]
        for dir_name in dirs_to_copy:
            source_dir = scan_path / dir_name
            if source_dir.is_dir():
                try:
                    shutil.copytree(source_dir, staged_scan_path / dir_name, symlinks=True)
                except Exception as e:
                    self.logger.error(f"Failed to copy {source_dir} for '{scan_path.name}': {e}")
                    raise
        
        self.logger.info(f"Finished staging '{scan_path.name}'.")
        return staged_scan_path

    def stage_async(self, scan_path: Path, staging_subdir: Path) -> Future:
        """Asynchronously stages a scan and returns a future."""
        return self.executor.submit(self._stage_scan_task, scan_path, staging_subdir)

    def cleanup(self):
        """Shuts down the thread pool and cleans up temporary directories."""
        self.logger.info("Shutting down staging manager and cleaning up temp files...")
        self.executor.shutdown(wait=True)
        try:
            shutil.rmtree(self.base_tmp_dir)
            self.logger.info(f"Removed temporary directory: {self.base_tmp_dir}")
        except OSError as e:
            self.logger.error(f"Error removing tmp directory {self.base_tmp_dir}: {e}")

# --- Configuration ---

@dataclass
class OrchestratorConfig:
    """Configuration for the parallel gsplat training orchestrator."""

    base_dir: Optional[Path] = field(default=None, metadata={"help": "Base directory containing the 'scan_*' folders to be processed."})
    
    # gsplat training config. We can override any setting from simple_trainer.py here
    # For example: --trainer.load_images_in_memory
    trainer: Config = field(default_factory=Config)

    gpu_ids: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7],
        metadata={"help": "A list of GPU IDs to use for parallel jobs."}
    )

    jobs_per_gpu: int = field(
        default=1,
        metadata={"help": "Number of parallel jobs to run on each GPU."}
    )
    
    time_limit_seconds: int = field(
        default= 3600, # 1 hour
        metadata={"help": "Time limit in seconds for each individual job."}
    )

    tmp_dir: Optional[Path] = field(default=None, metadata={"help": "Temporary directory for staging data. Defaults to system temp."})

    # These will be populated in main after base_dir is known
    aggregate_render_dir: Path = field(init=False)
    aggregate_ply_dir: Path = field(init=False)
    aggregate_mesh_dir: Path = field(init=False)


# --- Worker Function ---

def run_worker_process(args: tuple[Path, Path, int, OrchestratorConfig]) -> tuple[str, str, Any | None]:
    """
    A wrapper function that runs in each parallel process.
    It redirects all output to a dedicated log file.
    """
    original_scan_path, staged_scan_path, gpu_id, config = args
    scan_name = original_scan_path.name
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    log_file_path = original_scan_path / "gsplat.log"
    result_dir = original_scan_path / "3DGS"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True)

    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)
    
    try:
        with open(log_file_path, 'w') as log_file:
            log_fd = log_file.fileno()
            os.dup2(log_fd, original_stdout_fd)
            os.dup2(log_fd, original_stderr_fd)

            for name in logging.root.manager.loggerDict:
                logger = logging.getLogger(name)
                logger.handlers.clear()
                logger.propagate = True
            logging.getLogger().handlers.clear()

            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                force=True
            )

            logging.info(f"Worker process for '{scan_name}' starting on GPU {gpu_id} (PID: {os.getpid()}).")
            logging.info("All output from this process is now being redirected to this log file.")

            # --- Run gsplat training ---
            logging.info("--- Starting gsplat training ---")
            
            # Update trainer config with paths for this specific job
            trainer_config = config.trainer
            trainer_config.data_dir = str(staged_scan_path)
            trainer_config.result_dir = str(result_dir)

            run_gsplat_training(local_rank=0, world_rank=0, world_size=1, cfg=trainer_config)
            
            logging.info("--- gsplat training COMPLETED ---")
            
            # --- Post-processing ---
            logging.info("--- Starting post-processing ---")

            # Copy render
            source_render_dir = result_dir / "renders"
            if source_render_dir.exists():
                render_files = sorted(source_render_dir.glob("*.png"))
                if render_files:
                    last_image = render_files[-1]
                    destination_path = config.aggregate_render_dir / f"{scan_name}{last_image.suffix}"
                    shutil.copy(last_image, destination_path)
                    logging.info(f"Copied render to {destination_path}")
                else:
                    logging.warning(f"No render image found for scan '{scan_name}'.")

            # Copy PLY
            source_ply_dir = result_dir / "ply"
            if source_ply_dir.exists():
                ply_files = sorted(source_ply_dir.glob("*.ply"))
                if ply_files:
                    last_ply = ply_files[-1]
                    destination_path = config.aggregate_ply_dir / f"{scan_name}.ply"
                    shutil.copy(last_ply, destination_path)
                    logging.info(f"Copied PLY to {destination_path}")
                else:
                    logging.warning(f"No PLY file found for scan '{scan_name}'.")

            # Copy Mesh
            source_mesh_file = result_dir / "mesh" / "recon_29999.ply"
            if source_mesh_file.exists():
                destination_path = config.aggregate_mesh_dir / f"{scan_name}.ply"
                shutil.copy(source_mesh_file, destination_path)
                logging.info(f"Copied mesh to {destination_path}")
            else:
                logging.warning(f"No mesh file found for scan '{scan_name}'.")
            
            logging.info(f"Worker for '{scan_name}' completed successfully.")
            return scan_name, "success", None
    
    except Exception as e:
        logging.exception(f"WORKER FAILED: An unhandled exception occurred in the worker for '{scan_name}'.")
        return scan_name, "failed", str(e)

    finally:
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


# --- Main Application ---

def main(config: OrchestratorConfig, logger: logging.Logger):
    """Main function to orchestrate the parallel processing of scans."""

    if config.base_dir is None:
        logger.critical("Error: `base_dir` must be specified.")
        sys.exit(1)

    # --- Setup Paths and Directories ---
    config.aggregate_render_dir = config.base_dir / "all_renders"
    config.aggregate_ply_dir = config.base_dir / "all_splats"
    config.aggregate_mesh_dir = config.base_dir / "all_meshes"
    config.aggregate_render_dir.mkdir(exist_ok=True)
    config.aggregate_ply_dir.mkdir(exist_ok=True)
    config.aggregate_mesh_dir.mkdir(exist_ok=True)

    base_dir = config.base_dir
    logger.info("--- Starting Parallel gsplat Training (Multiprocessing) ---")
    
    # --- State Management Setup ---
    completed_tracker_file = base_dir / "completed_scans.txt"
    failed_tracker_file = base_dir / "failed_scans.txt"
    completed_tracker_file.touch()
    failed_tracker_file.touch()

    # --- Build Scan Queue ---
    try:
        completed_scans = set(completed_tracker_file.read_text().splitlines())
    except IOError as e:
        logger.error(f"Could not read completed scans file: {e}")
        completed_scans = set()
    
    logger.info(f"Loaded {len(completed_scans)} completed scans to be skipped.")

    all_scans_found = sorted(p for p in base_dir.glob("scan_*") if p.is_dir())
    scans_queue_all = [p for p in all_scans_found if p.name not in completed_scans]
    
    if not scans_queue_all:
        logger.info("No new scans to process. All tasks are complete.")
        sys.exit(0)

    total_scans_to_process = len(scans_queue_all)
    logger.info(f"Found {total_scans_to_process} new scans to process.")
    
    # --- Staging and Parallel Job Management ---
    if config.tmp_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        config.tmp_dir = Path(tempfile.gettempdir()) / f"gsplat_orchestrator_{timestamp}"
    
    staging_manager = StagingManager(config.tmp_dir, logger)
    
    scans_processed_count = 0
    start_time_total = time.time()

    scans_to_process_q = collections.deque(scans_queue_all)
    job_slots = config.gpu_ids * config.jobs_per_gpu
    free_gpus = collections.deque(job_slots)
    
    staging_jobs: Dict[int, Dict[str, Any]] = {}  # gpu_id -> {future, original_scan_path}
    active_jobs: Dict[Future, Dict[str, Any]] = {} # mp_future -> {original_scan_path, gpu_id, start_time}

    with NonDaemonPool(processes=len(job_slots)) as pool:
        try:
            while scans_processed_count < total_scans_to_process:
                # 1. Launch new staging jobs for any free GPUs
                while free_gpus and scans_to_process_q:
                    gpu_id = free_gpus.popleft()
                    original_scan_path = scans_to_process_q.popleft()
                    
                    gpu_staging_dir = config.tmp_dir / f"gpu_{gpu_id}"
                    staging_future = staging_manager.stage_async(original_scan_path, gpu_staging_dir)
                    staging_jobs[gpu_id] = {
                        "future": staging_future,
                        "original_scan_path": original_scan_path,
                    }

                # 2. Check for completed staging jobs and launch training
                for gpu_id in list(staging_jobs.keys()):
                    job_info = staging_jobs[gpu_id]
                    if job_info["future"].done():
                        try:
                            staged_scan_path = job_info["future"].result()
                            original_scan_path = job_info["original_scan_path"]
                            scan_name = original_scan_path.name
                            logger.info(f"🚀 Staging for '{scan_name}' complete. Starting training job on GPU {gpu_id}...")
                            
                            worker_args = (original_scan_path, staged_scan_path, gpu_id, config)
                            mp_future = pool.apply_async(run_worker_process, args=(worker_args,))
                            active_jobs[mp_future] = {
                                "original_scan_path": original_scan_path,
                                "gpu_id": gpu_id,
                                "start_time": time.time(),
                            }
                            del staging_jobs[gpu_id]

                        except Exception as e:
                            original_scan_path = job_info["original_scan_path"]
                            scan_name = original_scan_path.name
                            logger.error(f"❌ Staging for '{scan_name}' FAILED. Reason: {e}")
                            with open(failed_tracker_file, "a") as f:
                                f.write(f"{scan_name} (failed: staging - {e})\n")
                            scans_processed_count += 1
                            free_gpus.append(gpu_id)
                            del staging_jobs[gpu_id]

                # 3. Check for completed training jobs
                done_futures = []
                for future, job_info in active_jobs.items():
                    try:
                        result = future.get(timeout=0.01)
                        scans_processed_count += 1
                        duration = time.time() - job_info["start_time"]
                        scan_name = job_info["original_scan_path"].name
                        gpu_id = job_info["gpu_id"]
                        
                        _, status, error_message = result
                        if status == "success":
                            logger.info(f"✅ Job for '{scan_name}' on GPU {gpu_id} finished SUCCESSFULLY in {duration / 60:.2f} mins. ({scans_processed_count}/{total_scans_to_process})")
                            with open(completed_tracker_file, "a") as f:
                                f.write(f"{scan_name}\n")
                        else:
                            logger.error(f"❌ Job for '{scan_name}' on GPU {gpu_id} FAILED in {duration / 60:.2f} mins. Reason: {error_message} ({scans_processed_count}/{total_scans_to_process})")
                            with open(failed_tracker_file, "a") as f:
                                f.write(f"{scan_name} (failed: {error_message})\n")
                        
                        done_futures.append(future)
                        free_gpus.append(gpu_id)

                    except TimeoutError:
                        duration = time.time() - job_info["start_time"]
                        if duration > config.time_limit_seconds:
                            scan_name = job_info["original_scan_path"].name
                            gpu_id = job_info["gpu_id"]
                            logger.error(f"⏰ Job for '{scan_name}' on GPU {gpu_id} TIMED OUT after {duration / 60:.2f} mins. ({scans_processed_count}/{total_scans_to_process})")
                            with open(failed_tracker_file, "a") as f:
                                f.write(f"{scan_name} (failed: timeout)\n")
                            done_futures.append(future)
                            free_gpus.append(gpu_id)
                            scans_processed_count += 1
                    except Exception as e:
                        scan_name = job_info["original_scan_path"].name
                        logger.error(f"An unexpected error occurred while checking job '{scan_name}': {e}")
                        done_futures.append(future)
                        free_gpus.append(job_info["gpu_id"])
                        scans_processed_count += 1

                for future in done_futures:
                    del active_jobs[future]

                if scans_processed_count >= total_scans_to_process:
                    break
                
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user. Terminating pool...")
            pool.terminate()
            pool.join()
            sys.exit(130)
        finally:
            staging_manager.cleanup()

    # --- Final Report ---
    total_duration = time.time() - start_time_total
    logger.info("---")
    logger.info(f"All {scans_processed_count} scans have been processed.")
    if scans_processed_count > 0:
        average_duration = total_duration / scans_processed_count
        logger.info(f"📊 Total time for {scans_processed_count} scans: {total_duration / 60:.2f} minutes.")
        logger.info(f"📊 Average job time: {average_duration / 60:.2f} minutes.")
    logger.info("All tasks are complete.")


if __name__ == "__main__":
    # Set the start method to 'spawn' for CUDA compatibility
    multiprocessing.set_start_method('spawn', force=True)

    # --- Setup Orchestrator Logging ---
    orchestrator_logger = logging.getLogger("orchestrator")
    orchestrator_logger.setLevel(logging.INFO)
    orchestrator_logger.propagate = False
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    orchestrator_logger.addHandler(handler)
    
    # --- Suppress gsplat logger in main process ---
    gsplat_logger = logging.getLogger("gsplat")
    gsplat_logger.setLevel(logging.WARNING)

    # Disable tqdm in the orchestrator process
    os.environ['TQDM_DISABLE'] = '1'

    # Create the default configuration for the trainer based on run_gsplat.sh
    default_trainer_config = Config(
        load_images_in_memory=True,
        load_images_to_gpu=True,
        optimize_foreground=True,
        use_masks=True,
        disable_viewer=True,
        disable_video=True,
        init_type="visual_hull",
        hull_voxel_size=256,
        hull_images_per_camera=2,
        hull_camera_stride=2,
        hull_sample_from_first_quarter=True,
        max_steps=12000,
        save_steps=[12000],
        ply_steps=[12000],
        eval_steps=[12000],
        test_every=0,
        data_factor=1,
        random_bkgd=True,
        strategy=DefaultStrategy(refine_stop_iter=6000, verbose=False),
        use_rade=True,
        rade_step=6000,
    )

    default_config = OrchestratorConfig(trainer=default_trainer_config)
    
    cfg = tyro.cli(
        OrchestratorConfig,
        default=default_config,
        description="Default orchestrator configuration with settings from run_gsplat.sh.",
    )
    main(cfg, orchestrator_logger)