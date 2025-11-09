from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from utils import get_llm,load_docs,get_retrive,fetch_pyq_context
from retriver import create_retriver
from vectore_store import create_vectore_store
from prompt import get_study_assistant_template,get_concept_prompt,get_problem_prompt,get_pyq_prompt,get_hindi_conversational_prompt
from text_to_speech import speak_hindi
from agnets import get_extra_context



str_parser=StrOutputParser()

llm=get_llm()


chat_history=[]


def main_rag(query,docs,mode):
    if mode == "1":
        prompt=get_concept_prompt()

    elif mode == "2":
        prompt=get_problem_prompt()

    elif mode == "3":
        print("\n🌐 Searching for real past-year questions...\n")
        web_context = fetch_pyq_context(query)

        vectore_store=create_vectore_store(docs)

        retriver=create_retriver(vectore_store)

        pyq_prompt=get_pyq_prompt()

        pyq_chain= pyq_prompt | llm | str_parser
        pyq_result=pyq_chain.invoke({'context':web_context,'question':query,"chat_history": chat_history,'context1': retriver | RunnableLambda(get_retrive)})

        chat_history.append({'user': query, 'bot': pyq_result})

        return pyq_result

    elif mode == "4":
        prompt=get_hindi_conversational_prompt()
        vectore_store=create_vectore_store(docs)

        retriver=create_retriver(vectore_store)

        chat_history.append({'user':query})

        parallel_chain=RunnableParallel({
            'context': retriver | RunnableLambda(get_retrive),
            'question': RunnablePassthrough(),
            'chat_history':RunnableLambda(lambda x: chat_history),
            'context1': RunnableLambda(get_extra_context)
        })

        main_chain= parallel_chain | prompt | llm | str_parser

        result=main_chain.invoke(query)
        chat_history.append({'bot':result})

        print("\n💡 Final Answer:")
        print(result)
        
        speak_hindi(result)

        return
           

    else:
        prompt=get_study_assistant_template()

    vectore_store=create_vectore_store(docs)

    retriver=create_retriver(vectore_store)

    chat_history.append({'user':query})

    parallel_chain=RunnableParallel({
        'context': retriver | RunnableLambda(get_retrive),
        'question': RunnablePassthrough(),
        'chat_history':RunnableLambda(lambda x: chat_history),
        'context1': RunnableLambda(get_extra_context)
    })

    main_chain= parallel_chain | prompt | llm | str_parser

    result=main_chain.invoke(query)
    chat_history.append({'bot':result})

    return result


if __name__=="__main__":
    print("Welcome to EduRAG — Learn Smarter. Solve Faster. Understand Deeper.\n")
    doc_path=input("Enter file path (.txt, .pdf, .csv, .docx, web_url, .md): ")

    
    print("\n📄 Loading external documents...")
    docs=load_docs(doc_path)
    print(f"✅ Loaded {len(docs)} external document(s).")

    mode = input("""
🎓  Choose your Learning Mode in EduRAG:

 1️⃣  Concept Mode — Deep conceptual understanding  
 2️⃣  Problem Mode — Real-world problem-solving  
 3️⃣  PYQ Mode — Explore real past-year exam questions  
 4️⃣  Voice Mode — Listen to audio-based explanations 🎧
💡  Tip: Just press [Enter] to continue in Default Mode.
→  Enter your choice:""")
    
    while True:
        query = input("\n🔎 Ask a question (or type 'exit' to quit): ").strip()

        if query.lower() == "exit":
            break
        print("\n⏳ Generating answer...")

        result=main_rag(query,docs,mode)

        if result :
            print("\n💡 Final Answer:")
            print(result)
        
        









