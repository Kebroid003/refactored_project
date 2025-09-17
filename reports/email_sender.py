import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging


def send_email_report(report_path: str, recipient_email: str, email_user: str, email_password: str, email_server: str, email_port: int) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Sending report to {recipient_email}")

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = recipient_email
    msg['Subject'] = "Data Processing Report"

    body = (
        "Dear User,\n\n"
        "Please find attached the latest data processing report.\n\n"
        "Best regards,\n"
        "Data Processing System\n"
    )
    msg.attach(MIMEText(body, 'plain'))

    with open(report_path, "rb") as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename= {os.path.basename(report_path)}')
    msg.attach(part)

    server = smtplib.SMTP(email_server, email_port)
    server.starttls()
    server.login(email_user, email_password)
    server.sendmail(email_user, recipient_email, msg.as_string())
    server.quit()

    logger.info("Email sent successfully")


