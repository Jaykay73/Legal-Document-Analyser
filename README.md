# 📑 AI-powered Legal Document Analyzer

An interactive **Streamlit application** that helps users analyze and search through **legal contracts in PDF format**.
The app uses **state-of-the-art NLP embeddings** to extract clauses from contracts and allows users to query them with semantic search.

---

## 🚀 Features

* 📂 **Upload any PDF contract** and extract its text.
* ✂️ **Smart text chunking** into clauses/paragraphs for precise analysis.
* 🔍 **Semantic search** using [SentenceTransformers](https://www.sbert.net/):

  * Search for terms like *"payment terms"*, *"termination clause"*, or *"liability"*.
  * Retrieves clauses that are **semantically similar**, not just keyword matches.
* 📊 **Customizable settings** in the sidebar:

  * Choose embedding model (`all-MiniLM-L6-v2` or `multi-qa-MiniLM-L6-cos-v1`).
  * Control maximum chunk size, similarity threshold, and number of results.
* 🧠 **Cosine similarity scoring** to rank and filter results.
* 💾 **Download matching clauses as CSV** for record-keeping or further review.
* 📖 **Document preview** with extracted page-by-page text.
* ⚡ **Caching for speed** with `st.cache_resource` and `st.cache_data`.

---

## 🛠️ Tech Stack

* [**Python 3.9+**](https://www.python.org/)
* [**Streamlit**](https://streamlit.io/) – interactive frontend
* [**SentenceTransformers**](https://www.sbert.net/) – embeddings for semantic search
* [**PyPDF**](https://pypi.org/project/pypdf/) – PDF text extraction
* [**NumPy**](https://numpy.org/) – vector operations
* [**Pandas**](https://pandas.pydata.org/) – tabular results export

---

## 📂 Project Structure

```
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/Jaykay73/Legal-Document-Analyser.git
cd Legal-Document-Analyser
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

**requirements.txt** should include:

```
streamlit
sentence-transformers
pypdf
numpy
pandas
```

---

## ▶️ Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Then open your browser at: [http://localhost:8501](http://localhost:8501/)

---

## 📖 How It Works

1. **PDF Upload**

   * Upload a PDF contract (must contain extractable text).
   * If the PDF is scanned, OCR is required first (not handled here).

2. **Text Extraction & Chunking**

   * Splits document into paragraphs and further splits long text into manageable chunks.
   * Retains page numbers for context.

3. **Embeddings & Search**

   * Each chunk is converted into an embedding vector using SentenceTransformers.
   * When a query is entered, its embedding is compared against chunk embeddings using cosine similarity.

4. **Results**

   * Returns ranked list of clauses with scores.
   * Results above the chosen **similarity threshold** are displayed.
   * User can expand, preview, and export as CSV.

---

## ⚙️ Settings

* **Embedding model** – Choose between small/fast and QA-optimized models.
* **Max chunk characters** – Controls how text is split into pieces.
* **Similarity threshold** – Adjust to filter more or fewer results.
* **Max results to show** – Limit number of clauses returned.

---

## 📝 Example Queries

* `"termination clause"`
* `"payment schedule"`
* `"confidentiality agreement"`
* `"dispute resolution"`

---

## 📌 Notes & Limitations

* ❌ Does not handle scanned PDFs without text (use OCR beforehand).
* ⚡ Large documents may take longer to process.
* 📊 Similarity results depend on embedding model chosen.

---

## 🚀 Deployment

You can deploy this app easily on:

* **[Streamlit Cloud](https://streamlit.io/cloud)**
* **[Render](https://render.com/)**
* **[Heroku](https://www.heroku.com/)**
* **Docker** (optional)

Example Docker setup:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 👨‍💻 Author

Developed by **John Aledare** ( Machine Learning | AI | Data Science).

Feel free to connect on [X](https://www.X.com/Jermaine_73) or contribute to the project!
