with open('anthro_report.md', 'r') as report:
    template_contents = report.read()

report_content = template_contents.replace("{{report_contents_body}}", 'lol')

