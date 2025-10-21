import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

## --- 1. Prepare the Job Data (Knowledge Base) ---
# Replace this with your actual, comprehensive Lambton College Job FAQ data.
faq_text = """
Q: Who can participate in co-op?
A: Only students who are registered in a co-op program and meet the co-op eligibility criteria as stated on their program map can
participate in co-op opportunities.
Q: What are the benefits of being in a co-op program?
A: There are many benefits to being a co-op student. The most important being the hands-on, related experience you gain during your
co-op term. The experience you gain may help you secure a full-time job after graduation and help you develop contacts in your
profession.
Q: How do I know if I am eligible for co-op?
A: For students to be eligible for a co-op work term, they must meet the following criteria. During their active job search term (term
before going out on co-op), students will be advised by the college if they are eligible for a co-op work term based on the following
eligibility criteria:
• Have successfully completed all courses in their previous academic terms; and
• Be enrolled in all courses required in the third term; and
• Have a minimum program cumulative GPA of 2.8 or greater (or as specified for specific programs); and
• Have a valid Co-op Work Permit (for international students); and
• Fees paid in Full.
Q: Do I need a co-op work permit to participate in co-op?
A: Yes, as an international student, you MUST possess a valid Co-op Work Permit to accept and participate in co-op positions.
Students must apply for a co-op work permit well in advance of seeking or securing co-op employment or participating in a co-op
work term.
Q: Will I be placed in a co-op position?
A: Students are not placed or matched with an employer. Students in programs with co-op work experiences complete preemployment courses prior to their job search term within their programs to prepare them to market themselves to employers.
Along with the assistance and guidance from their Co-op Advisors and job developers, students are responsible to conduct their
own job search and compete for available co-op positions posted on the myCareer System or found by conducting their own job
search outside of the myCareer System to obtain an approved self developed co-op position.
Q: What is the myCareer System?
A: Students can search, view, and apply for available co-op opportunities during their active job search term in the myCareer System.
The college connects with employers to promote programs available for a co-op work term. Any available opportunities are posted
and promoted through the myCareer System for eligible students to view and compete for.
Q: Is there a deadline date to secure a co-op position?
A: Yes. Please consult your Co-op Advisor for important co-op dates. This information will be shared by your Co-op Advisor and is
specific to your program, intake, and study location, and will be communicated to students several times during the term before
your Co-op. Students must submit a Work Term Record through the myCareer System by the submission deadline date for review
by their Co-op Advisor.
Q: Will I be guaranteed a co-op position?
A: Lambton College does not guarantee co-op positions for students in any co-op programs. The most important factor in a student’s
success is the effectiveness of their job search.
Students who are fully engaged and committed to an active job search have normally done well in securing meaningful co-ops.
Q: Can students find and develop their own co-ops?
A: Yes, students can use their own networks and connections to find their own co-op positions across Canada. Student-developed
positions must be submitted to their co-op advisor for assessment. To allow sufficient time for proper assessment of self-developed
positions, positions must be submitted by the date provided by their Co-op Advisor during their active job search term.
Q: How do I protect myself from a potential job scam?
A: Use caution and good judgement when applying to job postings, attending interviews, and accepting offers of employment. Seek
advice from your Co-op Advisor immediately if you suspect any problems or do not understand what an employer is proposing or
asking of you.
Q: Can I accept a job offer and then decline it for another job offer later?
A: No, once you have accepted a job offer verbally or in writing, you are committed to the employer with whom you've accepted the
offer. If you accept more than one job offer you will fail co-op.
Q: Will I have to move to do my co-op?
A: Students are expected to consider out-of-city opportunities with the possibility of relocation or commuting to increase their co-op prospects. The Co-op & Career Services Centre develops relationships with employers and organizations locally as well as in numerous
communities throughout Ontario and across Canada. Co-op students will also be expected to conduct research into various other communities to learn more about valuable opportunities that exist outside of their home or school area.
Q: Can I do a co-op outside of Ontario or Canada as an International Learner?
A: Yes, you can complete a co-op outside of Ontario, but you are responsible for any additional documentation that may be required
(visa, work permit, insurance coverage, etc.) Lambton College does not approve co-ops outside of Canada for International Students.
Q: Am I responsible for my transportation and accommodations during my co-op work term?
A: Yes, it is the student’s responsibility to arrange for transportation to and from the job. Some locations are not accessible by public
transportation – make sure you have reliable transportation to and from your co-op. Living accommodations are also the student’s responsibility. We encourage students to research transportation and
accommodation options before applying to positions to ensure they can cover these costs before applying to a position.
Q: Will my co-op term be paid?
A: Students are expected to be flexible concerning co-op wages and should approach the work term as a learning experience and an opportunity for growth, rather than looking at it solely as an opportunity for financial gain.
Approved co-op positions and co-op work terms can be paid (at least minimum wage according to each province’s guidelines) or unpaid, commission-based, or otherwise. This depends on employers' preferences, the career field, and on the job market supply
and demand conditions which exist, including specific provincial regulations.
Q: If I complete an unpaid co-op work term, who will provide workplace insurance for me?
A: When you are completing your internship or co-op and not receiving pay from the employer, the college will provide workplace insurance for you if you are injured on the job.
Please contact your Co-op Advisor for further details regarding this process.
Q: What can I do to increase my chances of securing a PAID co-op?
A: Engage and be committed to an active job search
Q: Do I need a criminal record check or a security clearance to apply for co-op positions?
A: Many employers require their employees to complete employment pre-screening assessments including, but not limited to, criminal record checks and security clearances.
Q: Can I apply to as many co-op positions as I want?
A: Yes. Students may apply for as many co-op positions as they wish, but submission of an application (résumé, cover letter, etc.) indicates serious interest in the position. Once you have accepted a position (verbally or in writing) you must stop your job search
and can neither apply for nor accept any further positions.
Q: What is included in my co-op fee?
A: Students in a co-op program are assessed a fee during their academic terms to cover the cost of co-op related activities, supports, and services offered by Co-op & Career Services.
The Co-op & Career Services staff manages the co-op experience process
Q: Do I pay a co-op fee if I find my own job?
A: The financial model used by the co-op program is similar to other arrangements where there is a fee (e.g. health coverage, club memberships, some insurance.) Students pay for the availability of the service regardless of the extent of services utilized. Students who find their own job are still
using services provided by the Co-op & Career Services to assess, approve, track, and monitor their work terms.
Q: Can I receive a refund of the co-op fee if I do not participate in co-op?
A: No. Co-op fees are non-refundable.
"""

## --- 2. Split Text into QA Chunks ---
faq_chunks = [f"Q:{chunk.strip()}" for chunk in faq_text.split("Q:") if chunk.strip()]

## --- 3. Create Embeddings & FAISS Index ---
@st.cache_resource
def setup_rag_system():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(faq_chunks)
    embeddings = np.array(embeddings).astype('float32')  # FAISS requires float32

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return model, index

model, index = setup_rag_system()

## --- 4. RAG Search Function ---

def rag_search(question: str, k: int = 1) -> str:
    q_emb = model.encode([question]).astype('float32')
    D, I = index.search(q_emb, k=k)
    retrieved_chunks = [faq_chunks[i] for i in I[0] if i < len(faq_chunks)]
    
    cleaned_answers = []
    for chunk in retrieved_chunks:

        # Remove leading "Q:" if it exists, then split by "A:"

        if "A:" in chunk:
            parts = chunk.split("A:", 1)
            # Take the answer part only (after 'A:')
            answer_part = parts[1].strip()
            cleaned_answers.append(answer_part)
        else:
            # Fallback if no 'A:' found
            cleaned_answers.append(chunk.replace("Q:", "").strip())
    
    # Return only the answer (or answers if k > 1)

    return "\n\n".join(cleaned_answers)

## --- 5. Streamlit App Layout ---
st.title("Lambton College Coop FAQ Chatbot - RAG")
st.markdown(
    "Ask any question about co-op at Lambton College?"
)

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- User Input ---
user_question = st.text_input("Type your question here:")

if user_question:
    # Search FAQ
    answer = rag_search(user_question)
    
    # Save the Q&A in chat history
    st.session_state.chat_history.append({"question": user_question, "answer": answer})
    

# --- Display Chat History ---
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['question']}")
    st.info(f"**Bot:** {chat['answer']}")

# --- Optional: Show RAG Process Details ---
with st.expander("RAG Process Details (for learning purposes)"):
    if st.session_state.chat_history:
        last_chat = st.session_state.chat_history[-1]
        st.write("Your question was converted into a vector and compared to all FAQ chunks using FAISS.")
        st.write(f"**Original Question:** {last_chat['question']}")
        st.write(f"**Best Matched Chunks:** {last_chat['answer']}")
    else:
        st.write("No questions asked yet.")
