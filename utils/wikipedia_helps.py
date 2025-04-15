import wikipedia

def grab_wikipedia_context(query: str, num_pages: int = 3) -> str:
    try:
        # Initial search
        search_results = wikipedia.search(query, results=num_pages)
        if not search_results:
            return "I couldn't find any relevant information on Wikipedia."

        summaries = []
        attempted_titles = []

        for title in search_results[:num_pages]:
            attempted_titles.append(title)
            try:
                page = wikipedia.page(title, auto_suggest=False)
                summaries.append(f"Title: {page.title}\nSummary: {page.summary}")
            except wikipedia.exceptions.DisambiguationError as e:
                # Try to resolve ambiguity
                suggested = wikipedia.suggest(title)
                if suggested and suggested not in attempted_titles:
                    try:
                        page = wikipedia.page(suggested, auto_suggest=False)
                        summaries.append(f"Title: {page.title} (suggested from '{title}')\nSummary: {page.summary}")
                    except Exception as inner_e:
                        summaries.append(f"Title: {suggested} (from '{title}')\nSummary: Skipped due to: {inner_e}")
                else:
                    summaries.append(f"Title: {title} (ambiguous)\nSummary: Skipped due to ambiguity.")
            except Exception as e:
                summaries.append(f"Title: {title}\nSummary: Skipped due to error: {e}")

        context = "\n\n".join(summaries)
        return context
    except Exception as e:
        return f"An error occurred while using Wikipedia: {e}"
