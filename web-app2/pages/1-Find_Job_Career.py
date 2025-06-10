# =============================================================================
# Crawl web with BeautifulSoup and do a query on LLM
# Created: 19, Feb 2025
# Updated: 18, Mar 2025
# Writer: Ted, Jung
# Description: 
#   Scrap web page -> Indexing -> PromptTemplate -> Query
# =============================================================================



from narwhals import String
import streamlit as st
import smtplib


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.settings import Settings
from llama_index.core.indices import VectorStoreIndex
from llama_index.core import PromptTemplate


from IPython.display import Markdown, display




# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))


# Do a Query to find what you want
def ted_query(str_context, str_query):

    # Turn documents into an index for querying
    my_document = Document(text=str_context)
    documents = [my_document]
    ted_index = VectorStoreIndex.from_documents(documents)

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") 
    # Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5") 
    llm = OpenAI(model="gpt-4.1-nano", timeout=720.0)

    # print(str_context)

    # Creae a PromptTemplate
    qa_prompt_tmpl_str = """\
        You are an expert of the data analyst. You are given a list of job,country pairs. 
        Your task is to extract the following information:

        Context Input Data is below:
        ---------------------
        {context_str}
        ---------------------

        Given the context and Output direction not prior knowledge,
        Do not remove any information from the context.

        Answer the query.

        Query: {query_str}
        Answer: 
    """

    # Create an index by template with mapped variables
    # Answer the query against an index
    template_var_mappings = {"context_str": f"{str_context}",  "query_str": f"{str_query}"}
    prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str, template_var_mappings)

    answer_engine = ted_index.as_query_engine(llm = llm, prompt_tmpl=prompt_tmpl, response_mode="compact")

    display_prompt_dict(answer_engine.get_prompts())
    res = (answer_engine.query(str_query))

    return res



# Get sources from the web using headless browser
def ted_source_data(url) -> String:

    job_country_string = ""
    options = Options()
    options.add_argument("--headless")    # Run Chrome in headless mode
    options.add_argument("--disable-gpu") # Recommended for headless

    driver = webdriver.Chrome(options=options)
    driver.get(url)


    # Wait for the data to load (important!)
    # Example: Wait for an element to appear (replace with your selector, mb-16, mb-1, text-ne~~~)
    # CSS? replace with your web site
    try:
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, ".mb-16")) 
        WebDriverWait(driver, 10).until(element_present) # Wait up to 3 seconds
        html = driver.page_source                        # Get the updated HTML
        soup = BeautifulSoup(html, "html.parser")

        data_elements1 = soup.find_all("div", class_="mb-1")
        data_elements2 = soup.find_all("div", class_= "text-neutral-200 text-base font-normal")
        
        for element1, element2 in zip(data_elements1, data_elements2):
            role = element1.text.strip()
            country = element2.text.strip()
            job_country_string += f'{{"Job":"{role}","Country":"{country}"}},'
        
        return job_country_string
    
    except Exception as e:
        print(f"Error waiting for element: {e}")

    finally:
        driver.quit()



# need to update
def email_to(message):
    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)
    # start TLS for security
    s.starttls()
    # Authentication
    s.login("aaa@bbb.com", "pass")
    # message to be sent
    message = f"{message}"
    # sending the mail
    s.sendmail("aaa@bbb.com", "aaa@bbb.com", message)
    # terminating the session
    s.quit()



# Main
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    if st.button('Clear All Cache'):
        st.cache_data.clear()
        st.success('All cache cleared!')

    if st.button('Find Job'):    
        career_response = ted_source_data('https://clickhouse.com/company/careers')

        message = ted_query(career_response, """
            Which job has posted for which country? 
                            
            ### Output Direction. 
            1. Make a summary table with columns(Position, Country, Sum of Position, Continent) order by Position. 
            2. Make a summary table group by Country.
            3. Show Korea position status at the end. 

            """
        )

        st.write(message.response)

    # email_to(message)    

