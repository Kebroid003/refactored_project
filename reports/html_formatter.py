from datetime import datetime


def render_summary_html(summary_stats: dict) -> str:
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


