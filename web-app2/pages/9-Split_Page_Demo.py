from re import split
import streamlit as st
import PyPDF2

def display_pdf(file):
    pdf_reader = PyPDF2.PdfFileReader(file)
    num_pages = pdf_reader.numPages
    for page_num in range(num_pages):
        page = pdf_reader.getPage(page_num)
        st.write(page.extract_text())


    

def main():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    col1, col2 = st.columns([2, 8])  # Set the ratio of the columns to 20:80
    with col1:
        st.header("Open a Resume")
        st.write("Content for the left side of the page.")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            ted = display_pdf(uploaded_file)
            ted = "ted"

    with col2:
        st.header("Right Column")
        if uploaded_file is not None:
            st.write("ted")
            st.header("PDF Content"+col1.write(ted))

        st.write("Content for the right side of the page.")




main()