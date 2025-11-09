from flask import Flask, render_template, request
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from utils import get_llm, load_docs, get_retrive, fetch_pyq_context
from retriver import create_retriver
from vectore_store import create_vectore_store
from prompt import (
    get_study_assistant_template,
    get_concept_prompt,
    get_problem_prompt,
    get_pyq_prompt,
    get_hindi_conversational_prompt
)
from text_to_speech import speak_hindi
from agnets import get_extra_context
import os
import threading

# Flask app setup
app = Flask(__name__)


UPLOAD_DIR = "file"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global variables
chat_history = []
llm = get_llm()
str_parser = StrOutputParser()


import time
def delayed_speak(result):
    time.sleep(0.04)
    speak_hindi(result)


# ------------------ Core EduRAG Logic ------------------ #
def main_rag(query, docs, mode):
    if mode == "1":
        prompt = get_concept_prompt()

    elif mode == "2":
        prompt = get_problem_prompt()

    elif mode == "3":
        print("\n🌐 Searching for real past-year questions...\n")
        web_context = fetch_pyq_context(query)
        vectore_store = create_vectore_store(docs)
        retriver = create_retriver(vectore_store)
        pyq_prompt = get_pyq_prompt()
        pyq_chain = pyq_prompt | llm | str_parser

        pyq_result = pyq_chain.invoke({
            'context': web_context,
            'question': query,
            "chat_history": chat_history,
            'context1': retriver | RunnableLambda(get_retrive)
        })

        chat_history.append({'user': query, 'bot': pyq_result})
        return pyq_result

    elif mode == "4":
        prompt = get_hindi_conversational_prompt()
        vectore_store = create_vectore_store(docs)
        retriver = create_retriver(vectore_store)
        chat_history.append({'user': query})

        parallel_chain = RunnableParallel({
            'context': retriver | RunnableLambda(get_retrive),
            'question': RunnablePassthrough(),
            'chat_history': RunnableLambda(lambda x: chat_history),
            'context1': RunnableLambda(get_extra_context)
        })

        main_chain = parallel_chain | prompt | llm | str_parser
        result = main_chain.invoke(query)
        chat_history.append({'bot': result})

        threading.Thread(target=delayed_speak, args=(result,), daemon=True).start()
        return result

    else:
        prompt = get_study_assistant_template()

    vectore_store = create_vectore_store(docs)
    retriver = create_retriver(vectore_store)
    chat_history.append({'user': query})

    parallel_chain = RunnableParallel({
        'context': retriver | RunnableLambda(get_retrive),
        'question': RunnablePassthrough(),
        'chat_history': RunnableLambda(lambda x: chat_history),
        'context1': RunnableLambda(get_extra_context)
    })

    main_chain = parallel_chain | prompt | llm | str_parser
    result = main_chain.invoke(query)
    chat_history.append({'bot': result})

    return result


# ------------------ Flask Routes ------------------ #
@app.route("/", methods=["GET", "POST"])
def edu_rag():
    if request.method == "POST":
        query = request.form.get("query")
        mode = request.form.get("mode", "default")
        uploaded_file = request.files.get("file")

        if not query and not uploaded_file:
            return render_template("index.html", error_message="Please provide a question or upload a document.")

        # Handle file upload
        docs = []
        temp_path = None
        try:
            if uploaded_file:
                filename = uploaded_file.filename.lower()
                temp_path = os.path.join(UPLOAD_DIR, filename)
                uploaded_file.save(temp_path)
                print(f"✅ File saved: {filename}")
                docs = load_docs(temp_path)

            if not docs:
                return render_template("index.html", error_message="No valid content extracted from file.")

            print(f"📄 Loaded {len(docs)} document(s). Processing query: {query}")
            result = main_rag(query, docs, mode)

            return render_template("index.html", result=result, question=query, mode=mode)

        except ValueError as e:
            print(f"❌ ValueError: {e}")
            return render_template("index.html", error_message=str(e))

        except Exception as e:
            print(f"⚠️ Unexpected Error: {e}")
            return render_template("index.html", error_message="An unexpected error occurred during processing. Please try again.")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"🧹 Cleaned up temporary file: {temp_path}")

    return render_template("index.html")


# ------------------ Flask Entry ------------------ #
if __name__ == "__main__":
    try:
        app.run(debug=True, use_reloader=False)
    except OSError:
        print("⚠️ Server shutdown handled gracefully.")