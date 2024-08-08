# CogniChain: AI-Powered Research Assistant

CogniChain combines "Cognition" (the mental action of acquiring knowledge) and "Chain" (representing the linked nature of the research process). This project is an AI-powered research assistant that automates the process of gathering, analyzing, and summarizing information on various topics. It uses multiple sources including Google, Wikipedia, and arXiv to collect data, and leverages AI models for analysis and content generation.

## Features

- Automated research on user-specified topics
- Multi-source information gathering (Google, Wikipedia, arXiv)
- AI-powered content analysis and summarization
- Markdown file generation for each researched topic
- Obsidian-compatible note creation with aliases and links
- Automatic suggestion of related topics for continued research
- Checkpoint system for resuming interrupted research sessions
- Vector database (ChromaDB) integration for efficient information retrieval

## Requirements

- Python 3.7+
- Ollama server running (for AI model inference)
- ChromaDB
- Various Python libraries (see `requirements.txt`)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/EmreOzdemiroglu/CogniChain.git
   cd CogniChain
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up Ollama server and ensure it's running at the specified URL in the script.

4. Update the `OLLAMA_URL` variable in the script with your Ollama server's IP address.

## Usage

Run the main script:
  ```
python main.py
   ```

Follow the prompts to:
1. Enter an initial research topic
2. Specify the number of related topics to generate
3. Choose the desired output format (Markdown or Obsidian notes)
4. Wait for the research assistant to gather and analyze the information
5. Review the generated output and use the suggested related topics for continued research
