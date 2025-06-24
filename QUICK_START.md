# Quick Start guide
## 🚀 Installation Guide

### 1. Download and Install from Release

1. Go to the [Releases](https://github.com/QuipQuill/quip-quill/releases) page.
2. Download the latest `zip` or `tar.gz` archive.
3. Extract the archive and navigate into the project folder:

   ```bash
   unzip yourproject-<version>.zip    # or tar -xzf yourproject-<version>.tar.gz
   cd yourproject-<version>
   ```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate     # for Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Neo4j Database Setup

1. Install Neo4j (e.g., from the [official website](https://neo4j.com/download/)).
2. Start Neo4j and create two databases:

   * `staticdb` — stores static knowledge (preloaded facts and entities).
   * `dynamicdb` — stores dynamically generated entities and facts during runtime.

---

## ⚙️ Configuration File Setup

In the `config.yaml` file, specify the following:

```yaml
agentgraph:
  generated_mode: True         # Automatically create entities based on chat
  provider: google_genai       # Model provider: 'openai' or 'google_genai'
  load_entities: False         # Skip loading entities from documents on startup
```

---

## 🧾 Environment Variables Setup

Create a `.env` file in the project root and set the following variables:

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Paths
WORKING_DIR=/path/to/project_root
DATA_PATH=/path/to/documents

# LLM configuration
RAG_LLM_MODEL=gpt-4
RAG_LLM_MODEL_SUMMARIZER=gpt-3.5-turbo
RAG_BASE_URL=https://api.openai.com/v1

# API Keys
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_genai_key
```

---

### ✅ All Set!

You can now run the application or use the CLI tools included in the project.
