# RAG-Chatbot-
Lambton College Co-op FAQ Chatbot (RAG-Based)
#  Lambton College Co-op FAQ Chatbot (RAG-Based)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to assist Lambton College students with frequently asked questions about co-op jobs. The chatbot combines semantic search with generative AI to deliver accurate, conversational answers based on curated documents and resources.

---

## Purpose

Students often face confusion around co-op eligibility, application timelines, employer expectations, and academic requirements. This chatbot provides instant, reliable answers to common questions, helping students navigate the co-op process with confidence.

---

##  How It Works

- **Retrieval Layer**: Uses vector search to retrieve relevant FAQ content from a pre-indexed knowledge base.
- **Generation Layer**: Combines retrieved context with a language model to generate natural, informative responses.
- **RAG Architecture**: Ensures responses are grounded in official Lambton College documentation while maintaining conversational fluency.

---

## Project Structure

- `rag_chatbot.py`: Core chatbot logic and RAG pipeline
- `data/`: Contains indexed FAQ documents and co-op policy files
- `vector_store/`: Preprocessed embeddings for retrieval
- `README.md`: Project overview 

---

## Example Questions

- "Am I eligible for a co-op placement in my second semester?"
-“Can I accept a job offer and then decline it for another job offer later?”
-“Will I have to move to do my co-op?”
-“Can I do a co-op outside of Ontario or Canada as an International Learner?”
-“Am I responsible for my transportation and accommodations during my co-op work term?”
- "How do I find approved co-op employers?"
- "What GPA is required to apply for a co-op?"
- "Can international students participate in co-op programs?"


---

---
##ChatBot in Action
 <img width="915" height="602" alt="image" src="https://github.com/user-attachments/assets/81ad3478-775d-412e-9812-89bf0387384b" />
 <img width="719" height="529" alt="image" src="https://github.com/user-attachments/assets/7c3d8d4c-ff52-45eb-a5f0-4f3c6b34ac19" />
 <img width="806" height="532" alt="image" src="https://github.com/user-attachments/assets/c9f50146-2d7f-4e0f-9621-d700913301eb" />




 

 
## Notes

- All responses are grounded in Lambton College’s official co-op documentation.
- The chatbot is designed for informational support only and does not replace academic advising.
- Future improvements may include multilingual support, voice input, and integration with Lambton’s student portal.

---

## Contact

For questions or contributions, please reach out to boluyeye@gmail.com or open an issue in this repository.

