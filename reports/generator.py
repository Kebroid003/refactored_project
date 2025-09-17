from datetime import datetime
import os
import pandas as pd
import logging
from database.schemas import insert_report_log


def generate_text_summary(summary_stats: dict, connection, reports_directory: str) -> str:
    logger = logging.getLogger(__name__)
    logger.info("Generating summary text report")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(reports_directory, exist_ok=True)
    report_path = f"{reports_directory}/summary_report_{timestamp}.txt"

    with open(report_path, 'w') as file_handle:
        file_handle.write("DATA PROCESSING SUMMARY REPORT\n")
        file_handle.write("=" * 50 + "\n\n")
        file_handle.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file_handle.write("PROCESSING STATISTICS:\n")
        file_handle.write("-" * 25 + "\n")
        for key, value in summary_stats.items():
            file_handle.write(f"{key}: {value}\n")

    insert_report_log(connection, 'summary', report_path, 'completed')
    logger.info(f"Report generated: {report_path}")
    return report_path


def generate_html_report(summary_stats: dict) -> str:
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Processing Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .stats-table {{ border-collapse: collapse; width: 100%; }}
            .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .stats-table th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Data Processing Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="section">
            <h2>Processing Summary</h2>
            <table class="stats-table">
                <tr><th>Metric</th><th>Value</th></tr>
                {''.join([f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in summary_stats.items()])}
            </table>
        </div>
    </body>
    </html>
    """
    return html


def generate_detailed_html(summary_stats: dict, connection, reports_directory: str) -> str:
    logger = logging.getLogger(__name__)
    logger.info("Generating detailed HTML report")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(reports_directory, exist_ok=True)
    report_path = f"{reports_directory}/detailed_report_{timestamp}.html"

    html_content = generate_html_report(summary_stats)
    with open(report_path, 'w') as file_handle:
        file_handle.write(html_content)

    insert_report_log(connection, 'detailed', report_path, 'completed')
    logger.info(f"Report generated: {report_path}")
    return report_path


