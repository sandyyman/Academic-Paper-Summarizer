from typing import List, Dict

class PaperRecommender:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def get_recommendations(self, user_interests: List[str], recent_papers: List[Dict]) -> List[Dict]:
        """Get personalized paper recommendations"""
        try:
            # Combine user interests and recent paper topics
            combined_query = " ".join(user_interests)
            for paper in recent_papers:
                combined_query += f" {paper['title']} {paper.get('summary', '')}"
            
            # Get recommendations
            recommendations = self.vector_store.get_similar_papers(
                combined_query, n_results=5
            )
            
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return [] 