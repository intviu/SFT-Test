hp_pdf_path = "Harry_Potter.pdf"

chapters = split_into_chapters(hp_pdf_path)

# 2. Clean up the text in each chapter by replacing unwanted characters (e.g., '\t') with spaces.
#    This ensures the text is consistent and easier to process downstream.
chapters = replace_t_with_space(chapters)

# 3. Print the number of chapters extracted to verify the result.
print(len(chapters))

loader = PyPDFLoader(hp_pdf_path)
document = loader.load()

# 2. Clean the loaded document by replacing unwanted characters (e.g., '\t') with spaces
document_cleaned = replace_t_with_space(document)

# 3. Extract a list of quotes from the cleaned document as Document objects
book_quotes_list = extract_book_quotes_as_documents(document_cleaned)