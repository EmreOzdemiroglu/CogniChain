
import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import re

OLLAMA_URL = "http://192.168.1.74:11434"  # Replace this with your remote server's IP

def internet_search(query):
    url = f"https://www.google.com/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('div', class_='g')
    return [result.get_text() for result in results[:3]]  # First 3 results

def wikipedia_search(query, languages=['en', 'de', 'fr', 'es']):
    results = {}
    for lang in languages:
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": query
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            page = next(iter(data['query']['pages'].values()))
            extract = page.get('extract', '')
            if extract:
                results[lang] = extract
        except requests.exceptions.RequestException as e:
            print(f"Error during {lang} Wikipedia search: {str(e)}")
    
    return results if results else "No results found in any language."

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

def create_markdown_file(query, content, final_analysis, analogy):
    obsidian_folder = "obsidian"
    if not os.path.exists(obsidian_folder):
        os.makedirs(obsidian_folder)
    
    # Generate a short filename
    short_filename = generate_short_filename(query)
    
    filename = f"{obsidian_folder}/{short_filename}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"tags: #research #{query.replace(' ', '')}\n")
        f.write("alias:\n")
        f.write("links:\n\n")
        f.write(f"# {query}\n\n")
        f.write("## Research Results\n\n")
        f.write(content)
        f.write("\n## Final Analysis\n\n")
        f.write(final_analysis)
        f.write("\n\n## Analogy\n\n")
        f.write(analogy)
    
    return filename

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

def beautify_markdown(filename):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract existing structure
    lines = content.split('\n')
    tags = next((line for line in lines if line.startswith('tags:')), '')
    alias = next((line for line in lines if line.startswith('alias:')), '')
    links = next((line for line in lines if line.startswith('links:')), '')
    title = next((line for line in lines if line.startswith('# ')), '')
    
    # Extract main content
    main_content_start = content.index(title) + len(title)
    main_content = content[main_content_start:].strip()
    
    prompt = f"""Enhance and beautify the following Markdown content while preserving its structure:
    1. Keep the existing tags, alias, and links sections
    2. Preserve the main title
    3. Use headers (## for sub-headers) to organize the content
    4. Use bullet points (- or *) and numbered lists (1. 2. 3.) where appropriate
    5. Highlight important concepts by creating Obsidian links with [[]]
    6. Use **bold** or *italic* for emphasis
    7. Use > for important quotes or notes
    8. Use === highlight === for very important information
    9. Correct any spelling or grammatical errors
    10. Organize the content to be more readable and structured

    Original content:
    {main_content}
    """
    
    beautified_content = chat_with_llm(prompt)
    
    # Reconstruct the file with original structure and beautified content
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{tags}\n{alias}\n{links}\n\n{title}\n\n{beautified_content}")

    print(f"Markdown file has been beautified and saved as {filename}.")

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

def main():
    initial_query = input("Enter the initial topic you want to research: ")
    max_iterations = int(input("Enter the maximum number of research iterations: "))
    auto_continue = input("Do you want to automatically continue with new topics? (y/n): ").lower() == 'y'
    
    researched_topics = set()
    if os.path.exists("researchedtopics.md"):
        with open("researchedtopics.md", "r", encoding="utf-8") as f:
            researched_topics = set(f.read().splitlines())
    
    current_query = initial_query
    for iteration in range(max_iterations):
        if current_query in researched_topics:
            print(f"'{current_query}' has already been researched. Generating a new topic...")
            new_topic_prompt = f"Suggest a related but different topic to '{current_query}' that hasn't been researched yet. Keep it under 20 characters."
            current_query = chat_with_llm(new_topic_prompt).strip()[:20]
            continue
        
        print(f"\nIteration {iteration + 1}: Researching '{current_query}'")
        
        all_results = ""
        summaries = []

        # First internet search
        search_results1 = internet_search(current_query)
        all_results += "First Internet Search Results:\n"
        for i, result in enumerate(search_results1, 1):
            all_results += f"Result {i}: {result[:200]}...\n"
        
        # First summary
        prompt1 = f"Summarize the following search results and extract key points:\n\n{all_results}"
        summary1 = chat_with_llm(prompt1)
        summaries.append(summary1)
        all_results += f"\nSummary 1:\n{summary1}\n\n"

        # Second internet search (with first summary)
        query2 = f"{current_query} {summary1[:100]}"  # Create new query with first 100 characters of the first summary
        search_results2 = internet_search(query2)
        all_results += "Second Internet Search Results:\n"
        for i, result in enumerate(search_results2, 1):
            all_results += f"Result {i}: {result[:200]}...\n"
        
        # Second summary
        prompt2 = f"Summarize the new search results and extract key points:\n\n{all_results}"
        summary2 = chat_with_llm(prompt2)
        summaries.append(summary2)
        all_results += f"\nSummary 2:\n{summary2}\n\n"

        # Third internet search (with first two summaries)
        query3 = f"{current_query} {summary1[:50]} {summary2[:50]}"
        search_results3 = internet_search(query3)
        all_results += "Third Internet Search Results:\n"
        for i, result in enumerate(search_results3, 1):
            all_results += f"Result {i}: {result[:200]}...\n"
        
        # Third summary
        prompt3 = f"Summarize the new search results and extract key points:\n\n{all_results}"
        summary3 = chat_with_llm(prompt3)
        summaries.append(summary3)
        all_results += f"\nSummary 3:\n{summary3}\n\n"

        # Wikipedia search
        wiki_results = wikipedia_search(current_query)
        all_results += "Wikipedia Results:\n"
        if isinstance(wiki_results, dict):
            for lang, result in wiki_results.items():
                all_results += f"{lang.upper()}: {result[:200]}...\n"
        else:
            all_results += wiki_results + "\n"
        
        # Wikipedia summary
        prompt_wiki = f"Summarize the Wikipedia results and extract key points:\n\n{all_results}"
        summary_wiki = chat_with_llm(prompt_wiki)
        summaries.append(summary_wiki)
        all_results += f"\nWikipedia Summary:\n{summary_wiki}\n\n"

        # Final analysis
        final_prompt = f"Analyze all the following search results and summaries, and create a comprehensive report:\n\n{all_results}"
        final_analysis = chat_with_llm(final_prompt)

        # Create analogy
        analogy_prompt = f"Create an interesting and explanatory analogy for the topic '{current_query}'."
        analogy = chat_with_llm(analogy_prompt)

        # Create Markdown file
        filename = create_markdown_file(current_query, all_results, final_analysis, analogy)

        # Beautify Markdown file
        beautify_markdown(filename)

        print(f"\nResearch results and analysis have been saved and beautified in '{filename}'.")

        # Add the current query to researched topics
        researched_topics.add(current_query)
        with open("researchedtopics.md", "a", encoding="utf-8") as f:
            f.write(f"{current_query}\n")

        # Extract relevant concepts and select the next query
        concepts, next_query = extract_relevant_words(filename, current_query)
        
        print(f"Generated concepts: {', '.join(concepts)}")
        print(f"Selected concept for next iteration: {next_query}")
        
        # Check if the next query has already been researched
        while next_query in researched_topics and concepts:
            print(f"'{next_query}' has already been researched. Trying the next option...")
            concepts.remove(next_query)
            next_query = concepts[0] if concepts else None

        if not next_query:
            print("No new topics to research. Generating a new topic...")
            new_topic_prompt = f"Suggest a related but different topic to '{current_query}' that hasn't been researched yet. Keep it under 20 characters."
            next_query = chat_with_llm(new_topic_prompt).strip()[:20]

        print(f"Next research topic: {next_query}")

        if not auto_continue:
            user_input = input("Do you want to continue with the next topic? (y/n): ").lower()
            if user_input != 'y':
                break

        current_query = next_query

    print("Research process completed.")

if __name__ == "__main__":
    main()