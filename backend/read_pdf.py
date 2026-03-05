"""Extract text from hackathon problem statement PDF."""
from PyPDF2 import PdfReader

pdf_path = r"c:\Users\harir\Downloads\ALL\projects\plantiq\69997ffba83f5_problem_statement\Hackathon_Problem Statement.pdf"

reader = PdfReader(pdf_path)
print(f"Total pages: {len(reader.pages)}")
print("=" * 80)

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    print(f"\n--- PAGE {i+1} ---")
    print(text)
    print()
