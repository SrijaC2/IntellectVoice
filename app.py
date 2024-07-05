import os
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from sentence_splitter import SentenceSplitter
from gtts import gTTS
from dotenv import load_dotenv
from io import BytesIO
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'  
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
llm = ChatGoogleGenerativeAI(model="gemini-pro")

def get_response(input_text,num_return_sequences):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    global db
    if 'pdf-file' not in request.files:
        return jsonify({'result': 'No file added'}), 400

    file = request.files['pdf-file']

    if file.filename == '':
        return jsonify({'result': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        loader = PyPDFLoader(filepath)
        pages = loader.load_and_split()

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(pages, embeddings)

        return jsonify({'result': 'File uploaded successfully'})
    else:
        return jsonify({'result': 'File type not allowed'}), 400
    

@app.route('/process-pdf/summarize', methods=['POST'])
def summarize_pdf():
    query = "Summarize the text"  
    docs = db.similarity_search(query)
    content = "\n".join([x.page_content for x in docs])
    input_text = "\nContext:" + content + "\nUser prompt:\n" + query
    result = llm.invoke(input_text)
    return jsonify({'result': result.content})

@app.route('/process-pdf/paraphrase', methods=['POST'])
def paraphrase_pdf():  
    query = "Paraphrase the text"  
    docs = db.similarity_search(query)
    content = "\n".join([x.page_content for x in docs])
    splitter = SentenceSplitter(language='en')
    sentence_list = splitter.split(content)
    paraphrase = []
    for i in sentence_list: 
        a = get_response(i,1)
        paraphrase.append(a[0])
    result = ' '.join(paraphrase)
    return jsonify({'result': result})

  
@app.route('/process-pdf/keywords', methods=['POST'])
def extract_keywords_pdf():
    query = "Extract main key words from the text"  
    docs = db.similarity_search(query)
    content = "\n".join([x.page_content for x in docs])
    qa_prompt = "Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.----------------"
    input_text = qa_prompt + "\nContext:" + content + "\nUser question:\n" + query + "Give result in bullet points eg. \n- keyword1\n- keyword2"
    result = llm.invoke(input_text)
    return jsonify({'result': result.content})

@app.route('/download-audio', methods=['POST'])
def download_audio():
  text = request.form['text']
  audio_data = BytesIO()
  tts = gTTS(text)
  tts.write_to_fp(audio_data)
  audio_data.seek(0)
  return send_file(audio_data, mimetype='audio/mpeg', as_attachment=True, download_name='audio.mp3')

@app.route('/ask-question', methods=['POST'])
def extract_answer():
    data = request.get_json() 
    query = data.get('question')
    docs = db.similarity_search(query)
    content = "\n".join([x.page_content for x in docs])
    qa_prompt = "Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.----------------"
    input_text = qa_prompt + "\nContext:" + content + "\nUser question:\n" + query
    result = llm.invoke(input_text)
    return jsonify({'result': result.content})

if __name__ == '__main__':
    app.run(debug=True,threaded=True)
