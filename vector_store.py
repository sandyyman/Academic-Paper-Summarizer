import chromadb
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os


class VectorStore:
    def __init__(self):
        """Initialize ChromaDB and TF-IDF vectorizer"""
        # Create data directory if needed
        if not os.path.exists("./data"):
            os.makedirs("./data")

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./data/chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=512,
            stop_words='english'
        )

    def add_paper_with_summary(self, paper: Dict, generated_summary: str) -> bool:
        """Add paper and its generated summary to vector store"""
        try:
            # Create text for embedding
            text = f"{paper['title']} {paper['summary']} {generated_summary}"
            
            # Generate embedding
            embedding = self.vectorizer.fit_transform([text]).toarray()[0]
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Store in ChromaDB
            self.collection.add(
                ids=[paper["title"]],
                embeddings=[embedding.tolist()],
                documents=[text],
                metadatas=[{
                    "title": paper["title"],
                    "authors": ", ".join(paper["authors"]),
                    "pdf_url": paper["pdf_url"],
                    "published": str(paper["published"]),
                    "original_summary": paper["summary"],
                    "generated_summary": generated_summary
                }]
            )
            return True
        except Exception as e:
            print(f"Error adding paper: {e}")
            return False

    def get_similar_papers(self, query_text: str, n_results: int = 3) -> List[Dict]:
        """Find similar papers"""
        try:
            # Generate query embedding
            query_vector = self.vectorizer.transform([query_text]).toarray()[0]
            
            # Normalize
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=n_results,
                include=["documents", "metadatas"]
            )
            
            # Format results
            similar_papers = []
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]
                similar_papers.append({
                    "title": metadata["title"],
                    "generated_summary": metadata.get("generated_summary", "No summary available"),
                    "original_summary": metadata.get("original_summary", "")
                })
            
            return similar_papers
        except Exception as e:
            print(f"Error finding similar papers: {e}")
            return []

    def get_paper_count(self) -> int:
        """Get total number of stored papers"""
        return self.collection.count()

    def add_papers(self, papers: List[Dict], embeddings: List[List[float]]):
        """
        Add papers to the vector store
        """
        ids = [paper["id"] for paper in papers]
        documents = [paper["summary"] for paper in papers]
        metadatas = [
            {
                "title": paper["title"],
                "authors": ", ".join(paper["authors"]),
                "pdf_url": paper["pdf_url"],
                "published": str(paper["published"]),
            }
            for paper in papers
        ]

        self.collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )
        print(f"✅ Added {len(papers)} papers to vector store")

    def search_similar(
        self, query_embedding: List[float], n_results: int = 3
    ) -> List[Dict]:
        """
        Search for similar papers and return full context
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas"],
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
            )

        return formatted_results

    def analyze_research_trends(self, topic: str) -> Dict:
        """Analyze research trends from stored papers"""
        try:
            similar_papers = self.get_similar_papers(topic, n_results=10)
            
            trends = {
                "topics": {},
                "years": {},
                "authors": {},
                "methodologies": set(),
                "key_findings": []
            }
            
            for paper in similar_papers:
                # Analyze year trends
                year = paper.get("published", "").split("-")[0]
                if year:
                    trends["years"][year] = trends["years"].get(year, 0) + 1
                
                # Extract topics and methodologies using LLM
                if paper.get("generated_summary"):
                    # Add to key findings
                    trends["key_findings"].append({
                        "title": paper["title"],
                        "finding": paper["generated_summary"][:200]
                    })
            
            return trends
        except Exception as e:
            print(f"Error analyzing trends: {e}")
            return {}
