import json
import os
import shutil
from datetime import datetime
import logging


def backup_data(database_path: str, output_directory: str, reports_directory: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Creating data backup")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backups/backup_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)

    if os.path.exists(database_path):
        shutil.copy2(database_path, f"{backup_dir}/database_backup.db")

    if os.path.exists(output_directory):
        shutil.copytree(output_directory, f"{backup_dir}/output", dirs_exist_ok=True)

    if os.path.exists(reports_directory):
        shutil.copytree(reports_directory, f"{backup_dir}/reports", dirs_exist_ok=True)

    backup_info = {
        'timestamp': timestamp,
        'database_size': os.path.getsize(database_path) if os.path.exists(database_path) else 0,
        'files_backed_up': len(os.listdir(backup_dir)),
        'backup_size': sum(os.path.getsize(os.path.join(backup_dir, f))
                           for f in os.listdir(backup_dir)
                           if os.path.isfile(os.path.join(backup_dir, f)))
    }

    with open(f"{backup_dir}/backup_info.json", 'w') as file_handle:
        json.dump(backup_info, file_handle, indent=2)

    logger.info(f"Backup created: {backup_dir}")


