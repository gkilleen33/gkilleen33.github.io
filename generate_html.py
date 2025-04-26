import pandas as pd
from pylatexenc.latex2text import LatexNodes2Text

def convert_latex_to_html(latex_text):
    """Convert LaTeX to plain text or basic HTML."""
    if pd.isna(latex_text):
        return ""
    # Convert LaTeX to plain text (basic HTML-safe text)
    plain_text = LatexNodes2Text().latex_to_text(latex_text)
    # Return the converted text
    return plain_text

def generate_html():
    # File paths for the header, footer, and Excel sheet
    header_path = "research-header.html"
    footer_path = "research-footer.html"
    excel_path = "papers.xlsx"
    output_file = "research.html"
    
    # Read the header and footer files
    with open(header_path, "r") as f:
        header = f.read()

    with open(footer_path, "r") as f:
        footer = f.read()

    # Read the Excel file
    working_papers = pd.read_excel(excel_path, sheet_name="Working Papers", engine="openpyxl")
    publications = pd.read_excel(excel_path, sheet_name="Publications", engine="openpyxl")
    work_in_progress = pd.read_excel(excel_path, sheet_name="Work in Progress", engine="openpyxl")

    # Start building the HTML content
    html_content = header

    # Add Working Papers section
    html_content += "        <h2>Working Papers</h2>\n"
    for _, row in working_papers.iterrows():
        html_content += f"        <div class='paper'>\n"
        html_content += f"            <div class='paper-title'>{row['Title']}</div>\n"
        if not pd.isna(row.get('Authors')):
            html_content += f"            <div class='paper-authors'>{row['Authors']}</div>\n"
        if not pd.isna(row.get('Abstract')) or not pd.isna(row.get('Link')):
            html_content += f"            <div class='button-row'>\n"
            if not pd.isna(row.get('Abstract')):
                html_content += f"            <button class='toggle-button' onclick='toggleAbstract(this)'>Abstract</button>\n"
            if not pd.isna(row.get('Link')):
                html_content += f"            <a class='pdf-button' href='{row['Link']}' target='_blank'>Paper</a>\n"
            html_content += f"            </div>\n"
            if not pd.isna(row.get('Abstract')):
                if row['Latex'] == 1:
                    abstract_html = convert_latex_to_html(row['Abstract'])
                else:
                    abstract_html = row['Abstract']
                html_content += f"            <div class='abstract'><p>{abstract_html}</p></div>\n"
        html_content += "        </div>\n"

    # Add Publications section
    html_content += "        <h2>Publications</h2>\n"
    for _, row in publications.iterrows():
        html_content += f"        <div class='paper'>\n"
        html_content += f"            <div class='paper-title'>{row['Title']}</div>\n"
        if not pd.isna(row.get('Authors')):
            html_content += f"            <div class='paper-authors'>{row['Authors']}</div>\n"
        if not pd.isna(row.get('Abstract')) or not pd.isna(row.get('Link')):
            html_content += f"            <div class='button-row'>\n"
            if not pd.isna(row.get('Abstract')):
                html_content += f"            <button class='toggle-button' onclick='toggleAbstract(this)'>Abstract</button>\n"
            if not pd.isna(row.get('Link')):
                html_content += f"            <a class='pdf-button' href='{row['Link']}' target='_blank'>Paper</a>\n"
            html_content += f"            </div>\n"
            if not pd.isna(row.get('Abstract')):
                if row['Latex'] == 1:
                    abstract_html = convert_latex_to_html(row['Abstract'])
                else:
                    abstract_html = row['Abstract']
                html_content += f"            <div class='abstract'><p>{abstract_html}</p></div>\n"
        html_content += "        </div>\n"

    # Add Work in Progress section
    html_content += "        <h2>Work in Progress</h2>\n"
    for _, row in work_in_progress.iterrows():
        html_content += f"        <div class='paper'>\n"
        html_content += f"            <div class='paper-title'>{row['Title']}</div>\n"
        if not pd.isna(row.get('Authors')):
            html_content += f"            <div class='paper-authors'>{row['Authors']}</div>\n"
        if not pd.isna(row.get('Abstract')) or not pd.isna(row.get('Link')):
            html_content += f"            <div class='button-row'>\n"
            if not pd.isna(row.get('Abstract')):
                html_content += f"            <button class='toggle-button' onclick='toggleAbstract(this)'>Abstract</button>\n"
            if not pd.isna(row.get('Link')):
                html_content += f"            <a class='pdf-button' href='{row['Link']}' target='_blank'>Paper</a>\n"
            html_content += f"            </div>\n"
            if not pd.isna(row.get('Abstract')):
                if row['Latex'] == 1:
                    abstract_html = convert_latex_to_html(row['Abstract'])
                else:
                    abstract_html = row['Abstract']
                html_content += f"            <div class='abstract'><p>{abstract_html}</p></div>\n"
        html_content += "        </div>\n"

    # Append the footer
    html_content += footer

    # Write the final HTML content to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    generate_html()
