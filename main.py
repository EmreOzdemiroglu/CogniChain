import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
import signal
import sys
import arxiv
from urllib.parse import quote

OLLAMA_URL = "http://192.168.1.74:11434"  # Replace this with your remote server's IP

# ChromaDB için kalıcı depolama dizini belirle
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

# Dizinin var olduğundan emin ol
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)

# Chroma client'ı başlat
client = chromadb.Client(Settings(
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    is_persistent=True
))

# Koleksiyonu al veya oluştur
collection = client.get_or_create_collection("obsidian_notes")

CHECKPOINT_FILE = "research_checkpoint.json"

def internet_search(query, num_results=5):
    url = f"https://www.google.com/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('div', class_='g')
    return [result.get_text() for result in results[:num_results]]

def wikipedia_search(query, languages=['en', 'de', 'fr', 'es', 'it']):
    results = {}
    for lang in languages:
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "srprop": "snippet"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data['query']['search']:
                title = data['query']['search'][0]['title']
                snippet = data['query']['search'][0]['snippet']
                results[lang] = f"{title}: {snippet}"
            else:
                # If no direct match, try a more general search
                params["srsearch"] = f"{query} topic"
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                if data['query']['search']:
                    title = data['query']['search'][0]['title']
                    snippet = data['query']['search'][0]['snippet']
                    results[lang] = f"{title}: {snippet}"
        except requests.exceptions.RequestException as e:
            print(f"Error during {lang} Wikipedia search: {str(e)}")
    
    return results if results else "No results found in any language."

def get_wikipedia_topics(query):
    prompt = f"""Suggest 5 Wikipedia article titles related to the topic "{query}" that would be useful for research. 
    The titles should be in English and directly related to the main topic. 
    Include both specific and general terms.
    Format your response as a simple list of 5 titles, each on a new line."""
    
    response = chat_with_llm(prompt)
    topics = [title.strip() for title in response.split('\n') if title.strip()]
    
    # Add some general terms related to the query
    general_terms = [query, f"{query} concept", f"{query} in culture", f"{query} history", f"{query} examples"]
    
    return list(set(topics + general_terms))  # Remove duplicates

def chat_with_llm(prompt, model="mistral-nemo", max_tokens=None):
    try:
        json_data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 4096}  # Reset context for each new chat
        }
        if max_tokens:
            json_data["options"]["num_predict"] = max_tokens
        
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=json_data)
        response.raise_for_status()
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        return f"Error communicating with LLM: {str(e)}"

def generate_embedding_ollama(text):
    try:
        response = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
            "model": "mxbai-embed-large",
            "prompt": text
        })
        response.raise_for_status()
        return np.array(response.json()['embedding'])
    except requests.exceptions.RequestException as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def find_relevant_topics(current_content, top_n=3):
    current_embedding = generate_embedding_ollama(current_content)
    
    if current_embedding is None:
        return []
    
    # Convert NumPy array to list
    current_embedding_list = current_embedding.tolist()
    
    # Chroma'da benzer dokümanları ara
    results = collection.query(
        query_embeddings=[current_embedding_list],  # Pass as a list of lists
        n_results=top_n,
        include=["metadatas", "distances", "documents"]
    )
    
    relevant_topics = []
    for i in range(len(results['ids'][0])):
        topic = results['metadatas'][0][i]['query']
        filename = results['metadatas'][0][i]['filename']
        similarity = 1 - results['distances'][0][i]  # Mesafeyi benzerliğe çevir
        snippet = results['documents'][0][i][:200] + "..."  # İlk 200 karakter
        relevant_topics.append({
            "topic": topic,
            "filename": filename,
            "similarity": similarity,
            "snippet": snippet
        })
    
    return relevant_topics

def generate_aliases(query):
    prompt = f"""Generate 3-5 synonyms or closely related terms for "{query}". 
    These will be used as aliases in an Obsidian note.
    Provide only the terms, each on a new line, without any additional text or formatting."""
    
    response = chat_with_llm(prompt)
    aliases = [alias.strip() for alias in response.split('\n') if alias.strip()]
    return aliases

def create_markdown_file(query, content, final_analysis, analogy, collection):
    obsidian_folder = "obsidian"
    if not os.path.exists(obsidian_folder):
        os.makedirs(obsidian_folder)
    
    short_filename = generate_short_filename(query)
    filename = f"{obsidian_folder}/{short_filename}.md"
    
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    aliases = generate_aliases(query)
    aliases_yaml = "---\naliases:\n" + "\n".join(f"  - {alias}" for alias in aliases) + "\n---\n\n"
    
    relevant_docs = generate_relevant_documents(collection, content + final_analysis + analogy)
    links = "links: " + ", ".join(relevant_docs)
    
    summary = beautify_summary(generate_summary(content))
    importance_and_connections = beautify_importance_and_connections(query, content)
    detailed_results = beautify_detailed_results(content)
    beautified_analogy = beautify_analogy(analogy)
    
    try:
        unique_title, unique_content = generate_unique_section(query, content + final_analysis + analogy)
    except Exception as e:
        print(f"Error in generate_unique_section: {str(e)}")
        unique_title = "Unique Aspect of Research"
        unique_content = f"Error generating unique section: {str(e)}"

    markdown_content = f"""created: {current_date}
tags: #research #{query.replace(' ', '')}
{links}

{aliases_yaml}
# {query}

## Summary

{summary}

## Importance and Connections

{importance_and_connections}

## Detailed Research Results

{detailed_results}

## Final Analysis

{final_analysis}

## Analogy

{beautified_analogy}

## {unique_title}

{unique_content}
"""

    # Hata durumunda ek bilgi ekle
    if "Error generating unique section" in unique_content:
        markdown_content += "\n\nNOTE: There was an error generating the unique aspect section. Please check the logs for more details.\n"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    # Embedding oluştur ve Chroma'ya ekle
    embedding = generate_embedding_ollama(content + final_analysis + analogy)
    collection.add(
        documents=[content + final_analysis + analogy],
        embeddings=[embedding.tolist()],
        metadatas=[{
            "filename": short_filename, 
            "query": query, 
            "created": current_date, 
            "aliases": ", ".join(aliases)  # Aliases'ı string olarak ekliyoruz
        }],
        ids=[short_filename]
    )
    
    return filename, short_filename

def generate_short_filename(query):
    prompt = f"""Generate a short, descriptive filename for the following topic. 
    The filename should be:
    1. No longer than 15 characters
    2. Only use lowercase letters, numbers, and underscores
    3. Be relevant to the topic
    4. Avoid common words like 'the', 'a', 'an', etc.

    Topic: {query}

    Filename:"""
    
    response = chat_with_llm(prompt, max_tokens=20)
    
    # Clean the response
    filename = response.strip().lower()
    filename = ''.join(c if c.isalnum() or c == '_' else '_' for c in filename)
    
    # Ensure it's not longer than 15 characters
    return filename[:15]

def generate_summary(content):
    prompt = f"""
    Summarize the following research content. Focus on:
    1. Key findings
    2. Main themes
    3. Significant insights
    
    Use bullet points for clarity and keep it concise yet informative.

    Content to summarize:
    {content}
    """
    return chat_with_llm(prompt)

def beautify_summary(summary):
    prompt = f"""
    Enhance the following summary. Use:
    - Bullet points for key findings
    - Bold for important concepts
    - Ensure it's concise yet informative
    - Add a brief introduction and conclusion

    Summary to enhance:
    {summary}
    """
    return chat_with_llm(prompt)

def beautify_importance_and_connections(query, content):
    prompt = f"""
    Analyze the following research content on "{query}" and create a detailed section about its importance and connections:

    1. Importance:
       - Explain why this research on "{query}" is significant in the broader context.
       - Highlight potential impacts on industry, society, or technology.
       - Discuss any gaps in knowledge this research might fill.

    2. Importance of Each Source:
       - For each source mentioned (Google, Arxiv, Wikipedia):
         a) Explain why the information from this source is important.
         b) Discuss how it contributes to the overall understanding of the topic.
         c) Highlight any unique insights or perspectives this source provides.

    3. Connections:
       - Analyze how information from different sources relates to each other.
       - Identify common themes or contradictions across sources.
       - Explain how these connections contribute to a comprehensive understanding of "{query}".

    4. Synthesis:
       - Provide a brief synthesis of how all these elements (importance, source contributions, and connections) 
         form a cohesive understanding of the topic.

    Use appropriate Markdown syntax, including headers, subheaders, bullet points, 
    and emphasis where necessary. Ensure the content is detailed, informative, and well-structured.

    Content to analyze:
    {content}
    """
    return chat_with_llm(prompt)

def beautify_detailed_results(content):
    prompt = f"""
    Structure and enhance the following detailed research results:
    1. Use headers (##) for main topics
    2. Use subheaders (###) for subtopics
    3. Use bullet points or numbered lists for specific details
    4. Bold important terms or findings
    5. Use blockquotes (>) for significant quotes or findings
    6. Ensure proper formatting and readability

    Content to structure:
    {content}
    """
    return chat_with_llm(prompt)

def beautify_analogy(analogy):
    prompt = f"""
    Enhance the following analogy:
    1. Use bold for key comparisons
    2. Ensure each part of the analogy is clearly explained
    3. Add a brief introduction and conclusion
    4. Use subheaders or bullet points if necessary for clarity

    Analogy to enhance:
    {analogy}
    """
    return chat_with_llm(prompt)

def generate_relevant_documents(collection, current_content, top_n=3):
    current_embedding = generate_embedding_ollama(current_content)
    
    if current_embedding is None:
        return []
    
    current_embedding_list = current_embedding.tolist()
    
    results = collection.query(
        query_embeddings=[current_embedding_list],
        n_results=top_n,
        include=["metadatas", "distances"]
    )
    
    relevant_docs = []
    for i in range(len(results['ids'][0])):
        filename = results['metadatas'][0][i]['filename']
        query = results['metadatas'][0][i]['query']
        similarity = 1 - results['distances'][0][i]  # Mesafeyi benzerliğe çevir
        relevant_docs.append(f"[[{filename}|{query}]]")
    
    return relevant_docs

def extract_relevant_words(filename, original_query):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    
    prompt = f"""Based on the content in the file about "{original_query}", identify three relevant concepts for further research. Format your response as follows:

    [[Concept1]] [[Concept2]] [[Concept3]]

    Then, select the most relevant concept for the next research iteration. Ensure each concept is no longer than 20 characters.

    Content:
    {content}

    Concepts and selection:
    """
    
    response = chat_with_llm(prompt)
    
    # Extract concepts and selected concept
    concepts = re.findall(r'\[\[(.*?)\]\]', response)
    selected_line = [line for line in response.split('\n') if line.strip().startswith("Selected concept:")]
    
    if not concepts or len(concepts) < 3:
        # If not enough concepts, generate new ones
        new_prompt = f"""Generate three new related concepts for "{original_query}" for further research. Format your response as:

        [[Concept1]] [[Concept2]] [[Concept3]]

        Then, select the most relevant concept. Ensure each concept is no longer than 20 characters.

        Concepts and selection:
        """
        new_response = chat_with_llm(new_prompt)
        concepts = re.findall(r'\[\[(.*?)\]\]', new_response)
        selected_line = [line for line in new_response.split('\n') if line.strip().startswith("Selected concept:")]
    
    # Ensure concepts are no longer than 20 characters
    concepts = [concept[:20] for concept in concepts]
    
    # If still no concepts, use fallback
    if not concepts:
        concepts = [f"{original_query[:17]}...", f"New aspect of {original_query[:8]}...", f"Related to {original_query[:9]}..."]
    
    # Ensure we have at least 3 concepts
    while len(concepts) < 3:
        concepts.append(f"More on {original_query[:11]}...")
    
    # Select a concept
    if selected_line and ':' in selected_line[0]:
        selected_concept = selected_line[0].split(':')[1].strip()
        selected_concept = selected_concept[:20]  # Ensure it's not longer than 20 characters
    else:
        selected_concept = concepts[0]
    
    return concepts, selected_concept

def save_checkpoint(current_query, iteration, researched_topics, auto_continue, max_iterations):
    checkpoint = {
        "current_query": current_query,
        "iteration": iteration,
        "researched_topics": researched_topics,
        "auto_continue": auto_continue,
        "max_iterations": max_iterations
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)

def load_checkpoint():
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def signal_handler(sig, frame):
    print("\nProgram durduruluyor. İlerleme kaydediliyor...")
    save_checkpoint(current_query, iteration, researched_topics, auto_continue, max_iterations)
    sys.exit(0)

def arxiv_search(query, max_results=5):
    client = arxiv.Client()
    search = arxiv.Search(
        query = query,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )
    results = []
    for result in client.results(search):
        results.append({
            'title': result.title,
            'summary': result.summary[:200] + "..."  # İlk 200 karakter
        })
    return results

def generate_unique_section(query, full_content):
    prompt = f"""
    Based on the following research content about "{query}", generate a unique section that highlights 
    the most interesting or unexpected aspect of this topic. This section should:

    1. Have a creative and engaging title that captures the essence of the unique aspect.
    2. Contain 2-3 paragraphs of content exploring this aspect in detail.
    3. Relate this unique aspect back to the main topic of "{query}".
    4. If applicable, discuss any potential future implications or areas for further research.

    Format your response EXACTLY as follows:
    TITLE: [Your creative title here]
    CONTENT: [Your 2-3 paragraphs here]

    Research content:
    {full_content}
    """
    
    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": "mistral-nemo",
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 4096}
        })
        response.raise_for_status()
        
        result = response.json()['response']
        
        # Daha sağlam bir ayrıştırma yöntemi
        if "TITLE:" in result and "CONTENT:" in result:
            title_part = result.split("TITLE:")[1].split("CONTENT:")[0].strip()
            content_part = result.split("CONTENT:")[1].strip()
        else:
            # E��er beklenen format yoksa, tüm içeriği kullan
            title_part = "Unique Aspect of Research"
            content_part = result.strip()
        
        return title_part, content_part
    except Exception as e:
        print(f"Error in generate_unique_section: {str(e)}")
        return "Unique Aspect of Research", f"Error generating unique section: {str(e)}\n\nFull response:\n{result}"

def main():
    global current_query, iteration, researched_topics

    signal.signal(signal.SIGINT, signal_handler)

    checkpoint = load_checkpoint()
    if checkpoint:
        print("Önceki araştırma noktası bulundu. Devam etmek ister misiniz? (e/h)")
        if input().lower() == 'e':
            current_query = checkpoint["current_query"]
            iteration = checkpoint["iteration"]
            researched_topics = checkpoint["researched_topics"]
            auto_continue = checkpoint.get("auto_continue", False)
            max_iterations = checkpoint.get("max_iterations", 10)
            print(f"Araştırma '{current_query}' konusundan ve {iteration}. iterasyondan devam ediyor.")
        else:
            os.remove(CHECKPOINT_FILE)
            checkpoint = None
    
    if not checkpoint:
        initial_query = input("Araştırmak istediğiniz ilk konuyu girin: ")
        max_iterations = int(input("Maksimum araştırma iterasyon sayısını girin: "))
        auto_continue = input("Yeni konularla otomatik olarak devam etmek istiyor musunuz? (e/h): ").lower() == 'e'
        
        current_query = initial_query
        iteration = 0
        researched_topics = {}
        # Chroma'dan mevcut konuları al
        all_metadatas = collection.get()['metadatas']
        for metadata in all_metadatas:
            researched_topics[metadata['query'].lower()] = metadata['filename']
        
        # İlk verilen kelime için kontrol
        if current_query.lower() in researched_topics:
            print(f"'{current_query}' daha önce araştırılmış. Yeni bir konu öneriliyor...")
            new_topic_prompt = f"Suggest a related but different topic to '{current_query}' that hasn't been researched yet. Keep it under 20 characters."
            current_query = chat_with_llm(new_topic_prompt).strip()[:20]
            while current_query.lower() in researched_topics:
                print(f"'{current_query}' de daha önce araştırılmış. Başka bir konu öneriliyor...")
                new_topic_prompt = f"Suggest a different topic that hasn't been researched yet. Keep it under 20 characters."
                current_query = chat_with_llm(new_topic_prompt).strip()[:20]
            print(f"Yeni araştırma konusu: {current_query}")
    else:
        if 'max_iterations' not in locals():
            max_iterations = int(input("Maksimum araştırma iterasyon sayısını girin: "))

    while iteration < max_iterations:
        print(f"\nİterasyon {iteration + 1}: '{current_query}' araştırılıyor")
        
        all_results = ""
        summaries = []

        # Google search
        search_results = internet_search(current_query, num_results=5)
        all_results += "Google Search Results:\n"
        for i, result in enumerate(search_results, 1):
            all_results += f"Result {i}: {result[:200]}...\n"
        
        # Arxiv search
        arxiv_results = arxiv_search(current_query, max_results=5)
        all_results += "\nArxiv Search Results:\n"
        for i, result in enumerate(arxiv_results, 1):
            all_results += f"Result {i}: {result['title']} - {result['summary']}\n"

        # Wikipedia search
        wiki_topics = get_wikipedia_topics(current_query)
        all_results += "\nWikipedia Search Results:\n"
        for topic in wiki_topics:
            wiki_results = wikipedia_search(topic)
            if isinstance(wiki_results, dict):
                for lang, result in wiki_results.items():
                    all_results += f"{topic} ({lang.upper()}): {result}\n"
            else:
                all_results += f"{topic}: {wiki_results}\n"

        # Summary of all results
        prompt_summary = f"Summarize the following search results and extract key points:\n\n{all_results}"
        summary = chat_with_llm(prompt_summary)
        summaries.append(summary)
        all_results += f"\nSummary:\n{summary}\n\n"

        # Final analysis
        final_prompt = f"Analyze all the following search results and summaries, and create a comprehensive report:\n\n{all_results}"
        final_analysis = chat_with_llm(final_prompt)

        # Create analogy
        analogy_prompt = f"Create an interesting and explanatory analogy for the topic '{current_query}'."
        analogy = chat_with_llm(analogy_prompt)

        # Create Markdown file
        try:
            filename, short_filename = create_markdown_file(current_query, all_results, final_analysis, analogy, collection)
            print(f"\nResearch results and analysis have been saved in '{filename}'.")
        except Exception as e:
            print(f"Error creating Markdown file: {str(e)}")
            filename = f"error_{current_query.replace(' ', '_')}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Error occurred while creating file for {current_query}:\n{str(e)}")
            print(f"Error details have been saved in '{filename}'.")
            short_filename = filename

        # Add the current query to researched topics (case-insensitive)
        researched_topics[current_query.lower()] = short_filename

        # Extract relevant concepts and select the next query
        concepts, next_query = extract_relevant_words(filename, current_query)
        
        print(f"Generated concepts: {', '.join(concepts)}")
        print(f"Selected concept for next iteration: {next_query}")
        
        # Check if the next query has already been researched (case-insensitive)
        while next_query.lower() in researched_topics and concepts:
            print(f"'{next_query}' has already been researched. Trying the next option...")
            concepts.remove(next_query)
            next_query = concepts[0] if concepts else None

        if not next_query:
            print("No new topics to research. Generating a new topic...")
            new_topic_prompt = f"Suggest a related but different topic to '{current_query}' that hasn't been researched yet. Keep it under 20 characters."
            next_query = chat_with_llm(new_topic_prompt).strip()[:20]

        # Ensure the new topic hasn't been researched (case-insensitive)
        while next_query.lower() in researched_topics:
            print(f"'{next_query}' has already been researched. Generating a new topic...")
            new_topic_prompt = f"Suggest a different topic related to '{current_query}' that hasn't been researched yet. Keep it under 20 characters."
            next_query = chat_with_llm(new_topic_prompt).strip()[:20]

        print(f"Next research topic: {next_query}")

        if not auto_continue:
            user_input = input("Do you want to continue with the next topic? (y/n): ").lower()
            if user_input != 'y':
                break

        current_query = next_query
        iteration += 1
        save_checkpoint(current_query, iteration, researched_topics, auto_continue, max_iterations)

    print("Research process completed.")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

if __name__ == "__main__":
    main()
