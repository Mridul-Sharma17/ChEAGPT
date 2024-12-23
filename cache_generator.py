import os
import joblib
import re
import numpy as np
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY_4"))

# Create local cache folder
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Global variables
EMBEDDINGS = None
VECTOR_STORE = None

def initialize_resources():
    """Initialize global resources (embeddings and vector store)."""
    global EMBEDDINGS, VECTOR_STORE
    
    EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    VECTOR_STORE = FAISS.load_local(
        "rulebook_vector_store", 
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )
    print("Successfully initialized embeddings and vector store")

def get_semantic_key(question: str) -> str:
    """Generate a semantic cache key using embeddings."""
    global EMBEDDINGS
    vector = EMBEDDINGS.embed_query(question)
    semantic_key = hash(tuple(np.round(vector, decimals=5)))
    return f"semantic_search_{semantic_key}"

def save_cache_to_disk(cache_name: str, data):
    """Save cache data using joblib."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    joblib.dump(data, cache_path)

def load_cache_from_disk(cache_name: str):
    """Load cache data using joblib if it exists."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    if os.path.exists(cache_path):
        return 1
    return None

def make_cache(question: str):
    """Efficient Semantic Caching for Query Similarity Search."""
    global VECTOR_STORE
    
    # Step 1: Semantic Cache (using embeddings)
    semantic_key = get_semantic_key(question)
    cached_docs = load_cache_from_disk(semantic_key)
    if cached_docs is not None:
        return None  # Return if cached

    # Step 2: Perform Search and Cache
    docs = VECTOR_STORE.similarity_search(question)
    save_cache_to_disk(semantic_key, docs)
    return None

def main():
    # Initialize global resources
    initialize_resources()
    
    questions = """
    What's the exact difference between DE and STEM electives?,,, Can a STEM elective be counted as DE?,,, If I take more than required DE credits, can they be counted as STEM?,,, How do pre-registration and regular registration differ for electives?,,, What happens if I miss pre-registration but want to take a high-demand elective?,,, Can I pre-register for multiple sections of the same course?,,, What's the difference between course withdrawal and course drop?,,, If I withdraw from a course, does it affect my minimum credit requirement?,,, How is CPI calculated if I have W grades?,,, Can I convert a regular course to ALC after registration?,,, What's the deadline for converting a course to ALC?,,, If I fail an ALC, does it affect my CPI?,,, Can ALCs be used to fulfill graduation requirements?,,, How many ALCs can I take in one semester?,,, Is there a limit on total ALCs during my program?,,, What's the difference between Honours and Minor in terms of credit counting?,,, Can Honours courses be counted towards Minor requirements?,,, If I start a Minor but don't complete it, what happens to the credits?,,, Can Minor courses be counted as STEM electives?,,, What's the process for converting Minor courses to regular electives?,,, How does waitlist work for Minor courses?,,, If I get a course through waitlist, can I still drop it?,,, What happens if I fail a Minor course?,,, Can I repeat a Minor course to improve my grade?,,, How are Minor courses factored into overall CPI?,,, What's the difference between guided study and self-study courses?,,, Can I take guided study for core courses?,,, How many professors need to approve a guided study course?,,, What's the evaluation process for guided study courses?,,, Can guided study courses be counted as DEs?,,, What's the difference between SLP and guided study?,,, Can I do SLP under multiple professors?,,, How is the grade decided for group projects in SLP-IDP?,,, What happens if one team member drops SLP-IDP mid-semester?,,, Can I switch projects during SLP?,,, What's the difference between DD conversion and IDDDP?,,, Can I apply for both DD and IDDDP?,,, What happens to my Honours credits if DD conversion is rejected?,,, How does CPI calculation change after DD conversion?,,, Can I take additional Masters courses during DD?,,, What's the difference between Joint Masters and regular DD?,,, How are credits transferred in Joint Masters program?,,, Can I apply for Joint Masters after starting DD?,,, What happens if I don't maintain required CPI in Joint Masters?,,, How does funding work for Joint Masters program?,,, What's the difference between MCM and fee remission in terms of benefits?,,, Can I apply for both MCM and fee remission?,,, How is the income limit calculated for scholarships?,,, What documents are accepted as income proof?,,, Can scholarship status affect course registration?,,, What's the process if scholarship is approved after fee payment?,,, How does reimbursement work for scholarships?,,, Can I get multiple named scholarships?,,, How are named scholarship recipients selected?,,, What happens if I lose scholarship eligibility mid-semester?,,, How does the FR grade affect scholarship status?,,, What's the difference between FR and W grade in terms of scholarship?,,, Can I keep my scholarship during semester exchange?,,, How does semester exchange affect graduation timeline?,,, What happens if exchange courses don't match exactly with IITB courses?,,, Can I take extra credits during exchange?,,, How are exchange grades converted to IITB grades?,,, What happens if I fail a course during exchange?,,, Can I do exchange in my final semester?,,, How does exchange affect placement eligibility?,,, What's the process for course mapping approval in exchange?,,, Can I change mapped courses after reaching exchange university?,,, How does NPTEL credit transfer work?,,, Can NPTEL courses be counted as core courses?,,, What's the minimum grade required for NPTEL credit transfer?,,, How are NPTEL grades converted to IITB grades?,,, Can I take NPTEL courses during regular semester?,,, What happens if I fail NPTEL exam after course completion?,,, Can I repeat NPTEL courses?,,, How many times can I attempt NPTEL exam?,,, What's the difference between summer and regular semester courses?,,, Can I take advanced courses in summer?,,, How does summer course registration priority work?,,, What happens if summer course is cancelled after registration?,,, Can I take summer courses at other IITs?,,, How does summer course affect scholarship status?,,, What's the minimum attendance for summer courses?,,, Can I take more than one summer course simultaneously?,,, What's the difference between retagging and course conversion?,,, When is the best time to do retagging?,,, Can I retag courses after minimum credit completion?,,, What happens if I miss retagging deadline?,,, How does retagging affect graduation requirements?,,, Can I retag courses between departments?,,, What's the process for inter-departmental retagging approval?,,, How does course substitution differ from retagging?,,, What's the difference between Category I and II students?,,, How does category affect course registration?,,, Can category change during the program?,,, How does backlog affect category status?,,, What's the process for category improvement?,,, Can I take additional courses to improve category?,,, How does category affect scholarship status?,,, What's the difference between department and institute backlogs?,,, How do backlogs affect DD conversion eligibility?,,, Can I clear backlogs through NPTEL?,,, What's the maximum number of attempts for clearing backlogs?,,, How do backlogs affect placement eligibility?,,, Can I take advanced courses with backlogs?,,, What's the process for backlog course registration?,,, How does project evaluation work?,,, Can project grades be challenged?,,, What's the process for project extension?,,, How are group projects evaluated individually?,,, Can I change project type mid-semester?,,, What's the difference between project and thesis?,,, How is project work verified?,,, Can I do multiple projects under same professor?,,, What's the process for project collaboration?,,, How are interdisciplinary projects evaluated?,,, What's the difference between research and industry projects?,,, Can I convert course project to SLP?,,, How does project credit calculation work?,,, What's the minimum project duration?,,, Can I take courses during project semester?,,, How does project affect scholarship status?,,, What's the process for project funding?,,, Can I get stipend for projects?,,, How does project publication work?,,, What's the process for project patent?,,, Can I continue project after graduation?,,, How does leave affect project timeline?,,, What's the process for project report submission?,,, Can I change project domain?,,, How does project affect placement eligibility?,,, What's the difference between department and institute seminars?,,, Can seminar credits be counted as DEs?,,, How are seminar grades decided?,,, What's the minimum attendance for seminars?,,, Can I take multiple seminars in one semester?,,, How does seminar registration work?,,, What's the process for seminar presentation?,,, Can I change seminar topic?,,, How are group seminars evaluated?,,, What's the process for seminar report submission?,,, If I fail a HASMED elective, can I take a STEM elective instead?,,, Can I convert STEM to HASMED electives?,,, How many 700-level courses can I take?,,, What's the process for taking PhD level courses?,,, Can I audit PhD courses?,,, How does course audit affect CPI?,,, What's the difference between audit and sit-through?,,, Can I convert audit to credit course?,,, How many courses can I sit through?,,, What's the process for sit-through approval?,,, Can I get certificate for sit-through courses?,,, How does sit-through attendance work?,,, Can I take exams in sit-through courses?,,, What's the process for course feedback?,,, How does feedback affect grade?,,, Can I submit feedback after results?,,, What's the process for grade appeal?,,, How long does grade appeal take?,,, Can I see my answer sheets?,,, What's the process for copy checking?,,, How are relative grades calculated?,,, Can I know my absolute marks?,,, What's the minimum passing mark?,,, How does grace mark policy work?,,, Can I get grace marks for multiple courses?,,, What's the process for makeup examination?,,, How many makeup exams can I take?,,, Can I take makeup exam for improvement?,,, What's the process for medical makeup?,,, How does makeup exam affect grade?,,, Can I reject makeup exam grade?,,, What's the process for special examination?,,, How many special exams can I take?,,, Can I take special exam for improvement?,,, What's the process for supplementary examination?,,, How many supplementary exams can I take?,,, Can I take supplementary exam for improvement?,,, What's the process for re-examination?,,, How many re-exams can I take?,,, Can I take re-exam for improvement?,,, What's the difference between makeup and supplementary exams?,,, How does exam clash resolution work?,,, What's the process for exam rescheduling?,,, Can I take exam in different slot?,,, How does medical certificate submission work?,,, What's the deadline for medical certificate?,,, Can I get medical leave during exams?,,, What's the process for exam leave?,,, How does exam leave affect attendance?,,, Can I take exam from hospital?,,, What's the process for online examination?,,, How does online exam proctoring work?,,, Can I take online exam from home?,,, What's the process for technical issues during online exam?,,, How does online exam submission work?,,, Can I get extra time in online exam?,,, What's the process for online exam appeal?,,, How does online exam grading work?,,, Can I take online makeup exam?,,, What's the process for online exam registration?,,, How does online exam schedule work?,,, Can I change online exam slot?,,, What's the process for online exam preparation?,,, How does online exam monitoring work?,,, Can I use multiple devices in online exam?,,, What's the process for online exam feedback?,,, How does online exam security work?,,, Can I take break during online exam?,,, What's the process for online exam technical support?,,, How does online exam result declaration work?,,, Can I get online exam certificate?,,, What's the process for online exam grievance?,,, How does online exam attendance work?,,, Can I take online exam without webcam?,,, What's the process for online exam mock test?,,, How does online exam time management work?,,, Can I use calculator in online exam?,,, What's the process for online exam question paper?,,, How does online exam answer submission work?,,, Can I submit online exam late?,,, What's the process for online exam extension?,,, How does online exam backup work?,,, Can I retake online exam?,,, What's the process for online exam verification?,,, How does online exam certification work?
    """
    
    for question in questions.split(",,,"):
        query_modifier = ". Provide a comprehensive and informative explanation. If there is any relevant link or reference, please provide that as well."
        modified_query = question + query_modifier
        make_cache(modified_query)

if __name__ == "__main__":
    main()