import arxiv
from typing import List, Dict


class PaperAnalyzer:
    def __init__(self):
        self.client = arxiv.Client()

    def search_papers(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search for papers on ArXiv"""
        try:
            search = arxiv.Search(query=query, max_results=max_results)
            papers = []

            for result in self.client.results(search):
                paper = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary,
                    "pdf_url": result.pdf_url,
                    "published": result.published,
                    "categories": result.categories,
                }
                papers.append(paper)

            return papers
        except Exception as e:
            print(f"Error searching papers: {e}")
            return [] 