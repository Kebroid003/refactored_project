DEFAULT_CONFIG = {
    'database_path': 'data/analytics.db',
    'input_directory': 'data/input/',
    'output_directory': 'data/output/',
    'reports_directory': 'reports/',
    'log_file': 'logs/processor.log',
    'email_server': 'smtp.gmail.com',
    'email_port': 587,
    'email_user': 'analytics@company.com',
    'email_password': 'secure_password_123',
    'api_base_url': 'https://api.dataservice.com/v1',
    'api_key': 'sk-1234567890abcdef',
    'chunk_size': 10000,
    'max_file_size': 100 * 1024 * 1024,
    'backup_enabled': True,
    'encryption_key': 'my_secret_key_2024'
}


def overlay_config(base_config: dict, overrides: dict | None) -> dict:
    if not overrides:
        return base_config
    merged = base_config.copy()
    merged.update(overrides)
    return merged


