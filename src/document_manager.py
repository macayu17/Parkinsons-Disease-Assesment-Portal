"""
Document Manager for RAG System
Handles loading, processing, and retrieving medical documents for the RAG system
"""

import os
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2  # For handling PDF files

class DocumentManager:
    """Manages medical documents for the RAG system."""
    
    def __init__(self, docs_dir: str = "medical_docs"):
        """Initialize the document manager.
        
        Args:
            docs_dir: Directory where medical documents are stored
        """
        self.docs_dir = Path(docs_dir)
        self.documents = {}
        self.document_embeddings = {}
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Create docs directory if it doesn't exist
        os.makedirs(self.docs_dir, exist_ok=True)
        
        # For backward compatibility, also check for files directly in the docs_dir
        self.main_dir = self.docs_dir
        
        # Create subdirectories for different document types
        self.papers_dir = self.docs_dir / "papers"
        self.guidelines_dir = self.docs_dir / "guidelines"
        self.textbooks_dir = self.docs_dir / "textbooks"
        
        os.makedirs(self.papers_dir, exist_ok=True)
        os.makedirs(self.guidelines_dir, exist_ok=True)
        os.makedirs(self.textbooks_dir, exist_ok=True)
        
        # Load existing documents if any
        self.load_documents()
    
    def load_documents(self) -> None:
        """Load all documents from the docs directory."""
        self.documents = {}
        
        # Load documents from each subdirectory
        for doc_type, directory in [
            ("paper", self.papers_dir),
            ("guideline", self.guidelines_dir),
            ("textbook", self.textbooks_dir)
        ]:
            # Load text files
            for file_path in directory.glob("*.txt"):
                doc_id = f"{doc_type}_{file_path.stem}"
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract metadata from the first few lines
                metadata = self._extract_metadata(content)
                
                self.documents[doc_id] = {
                    "id": doc_id,
                    "type": doc_type,
                    "content": content,
                    "metadata": metadata,
                    "file_path": str(file_path)
                }
            
            # Load PDF files
            for file_path in directory.glob("*.pdf"):
                doc_id = f"{doc_type}_{file_path.stem}"
                content = self._extract_text_from_pdf(file_path)
                
                # Extract metadata from the first few lines
                metadata = self._extract_metadata(content)
                
                self.documents[doc_id] = {
                    "id": doc_id,
                    "type": doc_type,
                    "content": content,
                    "metadata": metadata,
                    "file_path": str(file_path)
                }
        
        # Also check for files directly in the main directory
        # Text files
        for file_path in self.main_dir.glob("*.txt"):
            doc_id = f"document_{file_path.stem}"
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from the first few lines
            metadata = self._extract_metadata(content)
            
            # Determine document type based on content or filename
            doc_type = "paper"  # Default type
            
            self.documents[doc_id] = {
                "id": doc_id,
                "type": doc_type,
                "content": content,
                "metadata": metadata,
                "file_path": str(file_path)
            }
            
        # PDF files
        for file_path in self.main_dir.glob("*.pdf"):
            doc_id = f"document_{file_path.stem}"
            content = self._extract_text_from_pdf(file_path)
            
            # Extract metadata from the first few lines
            metadata = self._extract_metadata(content)
            
            # Determine document type based on content or filename
            doc_type = "paper"  # Default type
            
            self.documents[doc_id] = {
                "id": doc_id,
                "type": doc_type,
                "content": content,
                "metadata": metadata,
                "file_path": str(file_path)
            }
        
        # Also check for files directly in the main directory
        for file_path in self.main_dir.glob("*.txt"):
            doc_id = f"document_{file_path.stem}"
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from the first few lines
            metadata = self._extract_metadata(content)
            
            # Determine document type based on content or filename
            doc_type = "paper"  # Default type
            
            self.documents[doc_id] = {
                "id": doc_id,
                "type": doc_type,
                "content": content,
                "metadata": metadata,
                "file_path": str(file_path)
            }
        
        # Create document embeddings
        self._create_embeddings()
        
        # Count document types
        doc_counts = {'paper': 0, 'guideline': 0, 'textbook': 0, 'total': len(self.documents)}
        for doc in self.documents.values():
            doc_type = doc.get('type', 'unknown')
            if doc_type in doc_counts:
                doc_counts[doc_type] += 1
        
        print(f"Loaded {doc_counts} medical documents")
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text content from a PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {e}")
            return f"Error extracting text: {str(e)}"
    
    def _extract_metadata(self, content: str) -> Dict:
        """Extract metadata from document content."""
        metadata = {
            "title": "",
            "authors": "",
            "year": "",
            "source": "",
            "keywords": []
        }
        
        # Try to extract metadata from the first 20 lines
        lines = content.split('\n')[:20]
        for line in lines:
            if line.lower().startswith("title:"):
                metadata["title"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("author") or line.lower().startswith("authors"):
                metadata["authors"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("year:"):
                metadata["year"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("source:"):
                metadata["source"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("keywords:"):
                keywords = line.split(":", 1)[1].strip()
                metadata["keywords"] = [k.strip() for k in keywords.split(",")]
        
        return metadata
    
    def _create_embeddings(self) -> None:
        """Create TF-IDF embeddings for all documents."""
        if not self.documents:
            return
        
        # Extract document contents
        doc_ids = list(self.documents.keys())
        doc_contents = [self.documents[doc_id]["content"] for doc_id in doc_ids]
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(doc_contents)
        
        # Store embeddings
        for i, doc_id in enumerate(doc_ids):
            self.document_embeddings[doc_id] = tfidf_matrix[i]
    
    def add_document(self, file_path: str, doc_type: str = "paper") -> str:
        """Add a new document to the collection.
        
        Args:
            file_path: Path to the document file
            doc_type: Type of document (paper, guideline, textbook)
            
        Returns:
            Document ID of the added document
        """
        file_path = Path(file_path)
        
        # Determine target directory
        if doc_type == "paper":
            target_dir = self.papers_dir
        elif doc_type == "guideline":
            target_dir = self.guidelines_dir
        elif doc_type == "textbook":
            target_dir = self.textbooks_dir
        else:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        # Copy file to target directory
        target_path = target_dir / file_path.name
        
        # Read content from source file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Write content to target file
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Extract metadata
        metadata = self._extract_metadata(content)
        
        # Create document ID
        doc_id = f"{doc_type}_{target_path.stem}"
        
        # Add to documents collection
        self.documents[doc_id] = {
            "id": doc_id,
            "type": doc_type,
            "content": content,
            "metadata": metadata,
            "file_path": str(target_path)
        }
        
        # Update embeddings
        self._create_embeddings()
        
        return doc_id
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the collection.
        
        Args:
            doc_id: ID of the document to remove
            
        Returns:
            True if document was removed, False otherwise
        """
        if doc_id not in self.documents:
            return False
        
        # Get file path
        file_path = self.documents[doc_id]["file_path"]
        
        # Remove file
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")
        
        # Remove from documents collection
        del self.documents[doc_id]
        
        # Remove from embeddings
        if doc_id in self.document_embeddings:
            del self.document_embeddings[doc_id]
        
        return True
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for documents matching the query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of matching documents with relevance scores
        """
        if not self.documents:
            return []
        
        # Create query embedding
        query_embedding = self.vectorizer.transform([query])
        
        # Calculate similarity scores
        scores = {}
        for doc_id, doc_embedding in self.document_embeddings.items():
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            scores[doc_id] = similarity
        
        # Sort by similarity score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = self.documents[doc_id].copy()
            doc["relevance"] = float(score)
            results.append(doc)
        
        return results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Get a document by its ID.
        
        Args:
            doc_id: ID of the document
            
        Returns:
            Document dict or None if not found
        """
        return self.documents.get(doc_id)
    
    def get_document_count(self) -> Dict[str, int]:
        """Get count of documents by type.
        
        Returns:
            Dict with counts by document type
        """
        counts = {
            "paper": 0,
            "guideline": 0,
            "textbook": 0,
            "total": len(self.documents)
        }
        
        for doc in self.documents.values():
            counts[doc["type"]] += 1
        
        return counts
    
    def extract_relevant_passages(self, query: str, top_k: int = 3, 
                                 passage_length: int = 500) -> List[Dict]:
        """Extract relevant passages from documents for a query.
        
        Args:
            query: Search query
            top_k: Number of top passages to return
            passage_length: Approximate length of each passage
            
        Returns:
            List of relevant passages with metadata
        """
        # First, get relevant documents
        relevant_docs = self.search_documents(query, top_k=top_k)
        
        passages = []
        for doc in relevant_docs:
            content = doc["content"]
            
            # Split content into paragraphs
            paragraphs = re.split(r'\n\s*\n', content)
            
            # Create passages by combining paragraphs
            current_passage = ""
            for para in paragraphs:
                if len(current_passage) + len(para) <= passage_length:
                    current_passage += para + "\n\n"
                else:
                    # Add current passage to results
                    if current_passage:
                        passages.append({
                            "text": current_passage.strip(),
                            "doc_id": doc["id"],
                            "doc_title": doc["metadata"]["title"],
                            "relevance": doc["relevance"]
                        })
                    current_passage = para + "\n\n"
            
            # Add final passage
            if current_passage:
                passages.append({
                    "text": current_passage.strip(),
                    "doc_id": doc["id"],
                    "doc_title": doc["metadata"]["title"],
                    "relevance": doc["relevance"]
                })
        
        # Sort passages by relevance
        passages.sort(key=lambda x: x["relevance"], reverse=True)
        
        return passages[:top_k]